# offline_rag_app.py — Zaktualizowana wersja 2025/2026
# Zmiany: nowe importy LangChain, persystencja FAISS, ładowanie JSONL,
#         invoke() zamiast run(), wybór profilu sprzętowego (--profil)

import os
import json
import argparse
import fitz        # PyMuPDF
import docx
import gradio as gr

# === NOWE importy LangChain (community split od v1.x+) ===
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA

# ============================================================
#  PROFILE SPRZĘTOWE — wybierasz przy starcie
# ============================================================
PROFILES = {
    "slaby": {
        "opis":        "💻 Słaby komputer (8 GB RAM) — Llama 3.2 8B",
        "model_path":  "Llama-3.2-8B-Instruct-Q4_K_M.gguf",
        "model_url":   "https://huggingface.co/bartowski/Llama-3.2-8B-Instruct-GGUF/resolve/main/Llama-3.2-8B-Instruct-Q4_K_M.gguf",
        "n_ctx":       4096,
        "n_batch":     256,
        "max_tokens":  512,
        "max_orzeczenia": 3_000,
        "max_qa":         1_000,
    },
    "sredni": {
        "opis":        "🖥️ Średni komputer (12–16 GB RAM) — Gemma 4 E4B",
        "model_path":  "gemma-4-e4b-it-Q4_K_M.gguf",
        "model_url":   "https://huggingface.co/bartowski/gemma-4-e4b-it-GGUF/resolve/main/gemma-4-e4b-it-Q4_K_M.gguf",
        "n_ctx":       8192,
        "n_batch":     512,
        "max_tokens":  1024,
        "max_orzeczenia": 7_000,
        "max_qa":         3_000,
    },
    "mocny": {
        "opis":        "🚀 Mocny komputer (16+ GB RAM) — Gemma 4 26B MoE",
        "model_path":  "gemma-4-26b-a4b-it-Q4_K_M.gguf",
        "model_url":   "https://huggingface.co/bartowski/gemma-4-26b-a4b-it-GGUF/resolve/main/gemma-4-26b-a4b-it-Q4_K_M.gguf",
        "n_ctx":       16384,
        "n_batch":     512,
        "max_tokens":  2048,
        "max_orzeczenia": 10_000,
        "max_qa":         5_000,
    },
}

# ==================== USTAWIENIA STAŁE ====================
EMBEDDING_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DOCS_FOLDER      = "docs"
JSONL_ORZECZENIA = "orzeczenia_SN.jsonl"
JSONL_QA         = "fine_tune_qa.jsonl"
FAISS_INDEX_PATH = "faiss_index"
# ==========================================================


def wybierz_profil() -> dict:
    """Parsuje --profil z CLI lub pyta interaktywnie."""
    parser = argparse.ArgumentParser(description="Lokalny Asystent Prawniczy RAG")
    parser.add_argument(
        "--profil",
        choices=["slaby", "sredni", "mocny"],
        help="Profil sprzętowy: slaby | sredni | mocny"
    )
    args, _ = parser.parse_known_args()

    if args.profil:
        return PROFILES[args.profil]

    # Interaktywny wybór jeśli nie podano argumentu
    print("\n" + "="*60)
    print("  Lokalny Asystent RAG — Orzeczenia Sądu Najwyższego")
    print("="*60)
    print("\n📋 Wybierz profil sprzętowy:\n")
    opcje = list(PROFILES.keys())
    for i, klucz in enumerate(opcje, 1):
        p = PROFILES[klucz]
        print(f"  [{i}] {p['opis']}")
        print(f"       Model: {p['model_path']}\n")

    while True:
        try:
            wybor = input("Wpisz numer (1/2/3): ").strip()
            if wybor in ["1", "2", "3"]:
                klucz = opcje[int(wybor) - 1]
                print(f"\n✅ Wybrany profil: {PROFILES[klucz]['opis']}\n")
                return PROFILES[klucz]
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Proszę wpisać 1, 2 lub 3.")


# --- Loader: pliki PDF / DOCX / TXT z folderu docs/ ---
def load_folder_documents(folder_path: str) -> list[Document]:
    documents = []
    if not os.path.isdir(folder_path):
        print(f"  [pominięto] folder '{folder_path}' nie istnieje")
        return documents

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        try:
            if filename.endswith(".pdf"):
                with fitz.open(path) as pdf:
                    text = "\n".join(page.get_text() for page in pdf)
            elif filename.endswith(".docx"):
                doc = docx.Document(path)
                text = "\n".join(p.text for p in doc.paragraphs)
            elif filename.endswith(".txt"):
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            else:
                continue
            documents.append(Document(page_content=text, metadata={"source": filename, "type": "doc"}))
        except Exception as e:
            print(f"  [błąd] {filename}: {e}")

    print(f"  Załadowano {len(documents)} plików z '{folder_path}'")
    return documents


# --- Loader: orzeczenia_SN.jsonl ---
def load_orzeczenia(jsonl_path: str, limit: int | None = None) -> list[Document]:
    documents = []
    if not os.path.isfile(jsonl_path):
        print(f"  [pominięto] plik '{jsonl_path}' nie istnieje")
        return documents

    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                record = json.loads(line)
                content  = record.get("content", "").strip()
                filename = record.get("filename", f"rekord_{i}")
                if content:
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": filename, "type": "orzeczenie_SN"}
                    ))
            except json.JSONDecodeError:
                continue

    print(f"  Załadowano {len(documents)} orzeczeń z '{jsonl_path}'")
    return documents


# --- Loader: fine_tune_qa.jsonl ---
def load_qa_pairs(jsonl_path: str, limit: int | None = None) -> list[Document]:
    documents = []
    if not os.path.isfile(jsonl_path):
        print(f"  [pominięto] plik '{jsonl_path}' nie istnieje")
        return documents

    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                record = json.loads(line)
                instruction = record.get("instruction", "").strip()
                context     = record.get("input", "").strip()
                answer      = record.get("output", "").strip()
                text = f"Pytanie: {instruction}\n\nKontekst:\n{context}\n\nOdpowiedź: {answer}"
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": f"qa_{i}", "type": "qa_pair"}
                    ))
            except json.JSONDecodeError:
                continue

    print(f"  Załadowano {len(documents)} par QA z '{jsonl_path}'")
    return documents


# ==================== MAIN ====================

profil = wybierz_profil()
MODEL_PATH = profil["model_path"]

# Unikalny folder indeksu per profil (żeby nie nadpisywać)
profil_nazwa = next(k for k, v in PROFILES.items() if v is profil)
faiss_path   = f"{FAISS_INDEX_PATH}_{profil_nazwa}"

print(f"\n[1] Ładowanie dokumentów (limit: {profil['max_orzeczenia']} orzeczeń)...")
all_docs = []
all_docs += load_folder_documents(DOCS_FOLDER)
all_docs += load_orzeczenia(JSONL_ORZECZENIA,  limit=profil["max_orzeczenia"])
all_docs += load_qa_pairs(JSONL_QA,            limit=profil["max_qa"])
print(f"  Łącznie dokumentów: {len(all_docs)}")

print("\n[2] Tworzenie chunków...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(all_docs)
print(f"  Łącznie chunków: {len(chunks)}")

print("\n[3] Embeddingi i FAISS...")
embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}   # zmień na "cuda" jeśli masz GPU NVIDIA
)

if os.path.isdir(faiss_path):
    print(f"  Wczytywanie istniejącego indeksu z '{faiss_path}'...")
    vectordb = FAISS.load_local(
        faiss_path,
        embedding,
        allow_dangerous_deserialization=True
    )
else:
    print("  Budowanie nowego indeksu (może potrwać kilka minut)...")
    vectordb = FAISS.from_documents(chunks, embedding)
    vectordb.save_local(faiss_path)
    print(f"  Indeks zapisany w '{faiss_path}'")

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

print(f"\n[4] Ładowanie modelu: {MODEL_PATH}")
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"\n❌ Model '{MODEL_PATH}' nie znaleziony!\n"
        f"Pobierz go z:\n  {profil['model_url']}\n"
        f"i umieść w folderze projektu."
    )

llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.3,
    max_tokens=profil["max_tokens"],
    top_p=0.95,
    n_ctx=profil["n_ctx"],
    n_batch=profil["n_batch"],
    verbose=False
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

print("\n✅ System gotowy!\n")


# ==================== INTERFEJS ====================

def ask_question(query: str) -> tuple[str, str]:
    if not query.strip():
        return "Proszę wpisać pytanie.", ""

    result = qa_chain.invoke({"query": query})
    answer = result.get("result", "Brak odpowiedzi")

    source_docs = result.get("source_documents", [])
    sources = ""
    seen = set()
    for doc in source_docs:
        src = doc.metadata.get("source", "nieznane")
        typ = doc.metadata.get("type", "")
        key = f"{src}|{typ}"
        if key not in seen:
            seen.add(key)
            sources += f"• [{typ}] {src}\n"

    return answer, sources.strip() or "Brak informacji o źródłach"


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="RAG — Orzeczenia Sądu Najwyższego"
) as interface:

    gr.Markdown(f"""
    # ⚖️ Lokalny Asystent Prawniczy RAG
    **Baza wiedzy:** Orzeczenia Sądu Najwyższego &nbsp;|&nbsp; **Profil:** {profil['opis']}
    """)

    with gr.Row():
        with gr.Column(scale=3):
            query_box = gr.Textbox(
                label="Twoje pytanie",
                placeholder="np. Kiedy Sąd Najwyższy odmawia przyjęcia skargi kasacyjnej?",
                lines=3
            )
            submit_btn = gr.Button("🔍 Szukaj", variant="primary")

        with gr.Column(scale=2):
            source_box = gr.Textbox(
                label="📂 Źródła",
                lines=6,
                interactive=False
            )

    answer_box = gr.Textbox(
        label="📋 Odpowiedź",
        lines=10,
        interactive=False
    )

    submit_btn.click(
        fn=ask_question,
        inputs=query_box,
        outputs=[answer_box, source_box]
    )

    gr.Examples(
        examples=[
            ["Kiedy Sąd Najwyższy odmawia przyjęcia skargi kasacyjnej do rozpoznania?"],
            ["Co to znaczy orzeczenie niezgodne z prawem w rozumieniu art. 4241 kpc?"],
            ["Jakie są przesłanki podziału majątku wspólnego małżonków?"],
        ],
        inputs=query_box
    )

interface.launch(server_name="0.0.0.0", server_port=7860)
