# offline_rag_app.py — Zaktualizowana wersja 2025/2026
# Zmiany: nowe importy LangChain, persystencja FAISS, ładowanie JSONL, invoke() zamiast run()

import os
import json
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

# ==================== USTAWIENIA ====================
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# === WYBÓR MODELU (zmień zależnie od dostępnego RAM) ===
# 8 GB  RAM → Gemma-4-E4B-IT-Q4_K_M.gguf          (~3 GB)   ← szybki
# 16 GB RAM → Gemma-4-26B-A4B-IT-Q4_K_M.gguf      (~14 GB)  ← zalecany ⭐
# 32 GB RAM → Gemma-4-26B-A4B-IT-Q8_0.gguf         (~28 GB)  ← najlepsza jakość
MODEL_PATH = "Gemma-4-E4B-IT-Q4_K_M.gguf"  # zmień na wariant pasujący do Twojego RAM
DOCS_FOLDER     = "docs"
JSONL_ORZECZENIA = "orzeczenia_SN.jsonl"
JSONL_QA         = "fine_tune_qa.jsonl"
FAISS_INDEX_PATH = "faiss_index"             # folder do zapisu/odczytu indeksu

# Limit rekordów JSONL (None = wszystkie; ustaw np. 5000 jeśli RAM jest ograniczony)
MAX_ORZECZENIA   = 10_000
MAX_QA           = 5_000
# ====================================================


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


# --- Loader: fine_tune_qa.jsonl (instruction + input jako kontekst) ---
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

                # Sklejamy pytanie + kontekst + odpowiedź jako jeden dokument
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


# ==================== INICJALIZACJA ====================

print("\n" + "="*55)
print("  Lokalny Asystent RAG — Orzeczenia Sądu Najwyższego")
print("="*55)

print("\n[1] Ładowanie dokumentów...")
all_docs = []
all_docs += load_folder_documents(DOCS_FOLDER)
all_docs += load_orzeczenia(JSONL_ORZECZENIA, limit=MAX_ORZECZENIA)
all_docs += load_qa_pairs(JSONL_QA, limit=MAX_QA)
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
    model_kwargs={"device": "cpu"}   # zmień na "cuda" jeśli masz GPU
)

if os.path.isdir(FAISS_INDEX_PATH):
    print(f"  Wczytywanie istniejącego indeksu z '{FAISS_INDEX_PATH}'...")
    vectordb = FAISS.load_local(
        FAISS_INDEX_PATH,
        embedding,
        allow_dangerous_deserialization=True
    )
else:
    print("  Budowanie nowego indeksu (może chwilę potrwać)...")
    vectordb = FAISS.from_documents(chunks, embedding)
    vectordb.save_local(FAISS_INDEX_PATH)
    print(f"  Indeks zapisany w '{FAISS_INDEX_PATH}'")

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

print("\n[4] Ładowanie lokalnego modelu LLM...")
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel '{MODEL_PATH}' nie znaleziony!\n"
        "Pobierz np. Llama-3.2-8B-Instruct.Q4_K_M.gguf z HuggingFace:\n"
        "  https://huggingface.co/bartowski/Llama-3.2-8B-Instruct-GGUF\n"
        "lub Zephyr:\n"
        "  https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF"
    )

llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.3,        # niżej = bardziej deterministyczny (lepiej dla prawa)
    max_tokens=1024,
    top_p=0.95,
    n_ctx=4096,             # zwiększony kontekst
    n_batch=512,
    verbose=False
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True  # zwracamy źródła
)

print("\n✅ System gotowy!\n")


# ==================== INTERFEJS ====================

def ask_question(query: str) -> tuple[str, str]:
    if not query.strip():
        return "Proszę wpisać pytanie.", ""

    result = qa_chain.invoke({"query": query})   # invoke() zamiast run()
    answer = result.get("result", "Brak odpowiedzi")

    # Formatujemy listę źródeł
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

    gr.Markdown("""
    # ⚖️ Lokalny Asystent Prawniczy RAG
    **Baza wiedzy:** Orzeczenia Sądu Najwyższego (dane offline, bez internetu)
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
