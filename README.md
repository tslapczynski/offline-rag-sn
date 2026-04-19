# ⚖️ Lokalny Asystent Prawniczy — Offline RAG

Aplikacja do zadawania pytań na podstawie orzeczeń Sądu Najwyższego.  
Działa **w pełni offline** — bez internetu, bez chmury, dane nie opuszczają komputera.

**Stack:** Python · LangChain · FAISS · LlamaCpp · Gradio · sentence-transformers

---

## 📋 Wymagania systemowe

| | Minimum | Zalecane |
|---|---|---|
| **RAM** | 8 GB | 16 GB |
| **Dysk** | 10 GB wolnego | 20 GB |
| **CPU** | 4 rdzenie | 8+ rdzeni |
| **GPU** | nie wymagane | NVIDIA (opcja) |
| **Python** | 3.10+ | 3.12 |

---

## 🚀 Instalacja krok po kroku

### Krok 1 — Pobierz repozytorium

```bash
git clone https://github.com/tslapczynski/offline-rag-sn.git
cd offline-rag-sn
```

### Krok 2 — Utwórz wirtualne środowisko Python

**Windows:**
```powershell
python -m venv myenv
.\myenv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv myenv
source myenv/bin/activate
```

### Krok 3 — Zainstaluj zależności

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Krok 4 — Zainstaluj PyTorch

> ⚠️ Torch instalujemy osobno — wersja zależy od Twojego sprzętu.

**Tylko CPU (każdy komputer):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU (szybciej):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Krok 5 — Zainstaluj llama-cpp-python

> ⚠️ Ta biblioteka obsługuje modele GGUF. Wersja zależy od sprzętu.

**Tylko CPU:**
```bash
pip install llama-cpp-python
```

**NVIDIA GPU (znacznie szybciej!):**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
```

**Windows + GPU (PowerShell):**
```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall
```

---

## 📦 Dane — pliki JSONL (za duże na GitHub)

Pliki danych są za duże na GitHub (>100 MB). Prześlij je osobno na docelowy komputer.

| Plik | Rozmiar | Opis |
|---|---|---|
| `orzeczenia_SN.jsonl` | ~123 MB | Treści orzeczeń Sądu Najwyższego |
| `fine_tune_qa.jsonl` | ~263 MB | Pary pytanie–odpowiedź |

**Skopiuj oba pliki do głównego folderu projektu** (obok `offline_rag_app.py`).

---

## 🤖 Model LLM — plik GGUF

> **Dlaczego Gemma 4?** Gemma 4 od Google (kwiecień 2026, Apache 2.0) ma natywne wsparcie 140+ języków — **polski jest priorytetowy**.  
> Llama 3.2 oficjalnie wspiera tylko 8 języków (polskiego nie ma na liście).

Pobierz jeden z modeli i umieść w głównym folderze projektu:

| Model | RAM | Jakość PL | Link |
|---|---|---|---|
| **Gemma-4-E4B** ⭐ *8 GB RAM* | ~3 GB | ⭐⭐⭐⭐ | [pobierz](https://huggingface.co/bartowski/gemma-4-e4b-it-GGUF/resolve/main/gemma-4-e4b-it-Q4_K_M.gguf) |
| **Gemma-4-26B MoE** 🏆 *16 GB RAM* | ~14 GB | ⭐⭐⭐⭐⭐ | [pobierz](https://huggingface.co/bartowski/gemma-4-26b-a4b-it-GGUF/resolve/main/gemma-4-26b-a4b-it-Q4_K_M.gguf) |
| Llama 3.2 8B *(backup)* | ~5 GB | ⭐⭐⭐ | [pobierz](https://huggingface.co/bartowski/Llama-3.2-8B-Instruct-GGUF/resolve/main/Llama-3.2-8B-Instruct-Q4_K_M.gguf) |

Następnie zaktualizuj `MODEL_PATH` w `offline_rag_app.py` na nazwę pobranego pliku:

```python
# 8 GB RAM:
MODEL_PATH = "gemma-4-e4b-it-Q4_K_M.gguf"

# 16 GB RAM (zalecany):
MODEL_PATH = "gemma-4-26b-a4b-it-Q4_K_M.gguf"
```

---

## ▶️ Uruchomienie

```bash
# Aktywuj środowisko (jeśli nie jest aktywne)
.\myenv\Scripts\activate        # Windows
source myenv/bin/activate       # Linux/macOS
```

### Wybór profilu sprzętowego

Przy starcie aplikacja pyta o profil — wybierz odpowiedni dla swojego komputera:

```
📋 Wybierz profil sprzętowy:

  [1] 💻 Słaby komputer (8 GB RAM)    — Llama 3.2 8B    (~5 GB model)
  [2] 🖥️ Średni komputer (12-16 GB)  — Gemma 4 E4B     (~3 GB model)
  [3] 🚀 Mocny komputer (16+ GB RAM)  — Gemma 4 26B MoE (~14 GB model) ⭐

Wpisz numer (1/2/3):
```

Możesz też podać profil jako argument — pomija menu:

```bash
python offline_rag_app.py --profil slaby    # 8 GB RAM  → Llama 3.2 8B
python offline_rag_app.py --profil sredni   # 12 GB RAM → Gemma 4 E4B
python offline_rag_app.py --profil mocny    # 16+ GB    → Gemma 4 26B MoE ⭐
```

Otwórz przeglądarkę: **http://localhost:7860**

### Co dzieje się przy pierwszym uruchomieniu?

```
[1] Ładowanie dokumentów...     ← wczytuje JSONL (kilka sekund)
[2] Tworzenie chunków...        ← dzieli tekst na fragmenty
[3] Embeddingi i FAISS...       ← BUDUJE INDEKS (może trwać 5-20 min!)
    Indeks zapisany w 'faiss_index_slaby'   ← osobny per profil!
[4] Ładowanie modelu LLM...     ← ładuje model GGUF
✅ System gotowy!
```

> 💡 **Każdy profil ma własny indeks FAISS** — możesz przełączać się między profilami bez przebudowy.  
> Każde kolejne uruchomienie tego samego profilu startuje błyskawicznie!

---

## 📁 Struktura projektu

```
offline-rag-sn/
├── offline_rag_app.py       # główna aplikacja
├── requirements.txt         # zależności Python
├── .gitignore               # co nie wchodzi na GitHub
├── README.md                # ta instrukcja
│
├── orzeczenia_SN.jsonl      # ⚠️ NIE na GitHub — prześlij osobno
├── fine_tune_qa.jsonl       # ⚠️ NIE na GitHub — prześlij osobno
├── *.gguf                   # ⚠️ NIE na GitHub — pobierz osobno
│
├── faiss_index/             # generowany automatycznie (nie commituj)
├── docs/                    # opcjonalnie: Twoje własne PDF/DOCX/TXT
└── myenv/                   # środowisko wirtualne (nie commituj)
```

---

## ⚙️ Konfiguracja (offline_rag_app.py)

```python
# Ile rekordów załadować z JSONL (None = wszystkie)
MAX_ORZECZENIA = 10_000   # ← zmniejsz jeśli mało RAM
MAX_QA         = 5_000

# CPU vs GPU dla embeddingów
model_kwargs={"device": "cpu"}   # ← zmień na "cuda" jeśli masz GPU NVIDIA
```

---

## 🔧 Rozwiązywanie problemów

**`FileNotFoundError: model *.gguf`**  
→ Pobierz plik modelu i umieść go w folderze projektu, zaktualizuj `MODEL_PATH`

**`MemoryError` lub zawiesza się**  
→ Zmniejsz `MAX_ORZECZENIA = 3000` i `MAX_QA = 1000` w pliku `offline_rag_app.py`

**Pierwsze uruchomienie trwa bardzo długo**  
→ To normalne — buduje indeks FAISS z tysięcy dokumentów. Kolejne starty będą szybkie.

**Port 7860 zajęty**  
→ Zmień `server_port=7860` na inny np. `7861`

---

## 📄 Licencja

Projekt prywatny / do celów badawczych.
