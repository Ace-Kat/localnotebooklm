# Local NotebookLM

A fully local, privacy-preserving document Q&A app. Upload PDFs, text files, or Markdown files, then ask questions and get cited answers — everything runs on-device with no cloud API calls or API keys required.

## Prerequisites

- **Python 3.11+**
- **Ollama** — local LLM runtime (handles both CUDA on Windows/Linux and Metal on Apple Silicon automatically)

## Step 1: Install Ollama

### Windows (RTX 3060 dev machine)
```
winget install Ollama.Ollama
```
Or download the installer from https://ollama.com/download/windows

### macOS (M1 target machine)
```
brew install ollama
```
Or download from https://ollama.com/download/mac

After installing, start the Ollama service:
```
ollama serve
```
(On macOS, Ollama starts automatically after install; on Windows you may need to run this or start it from the tray icon.)

## Step 2: Pull models

Run these once (Ollama caches them; re-run on a new machine):
```
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull nomic-embed-text
```

Downloads: ~4.5 GB (Qwen) + ~280 MB (nomic-embed-text). Pull them before setup so they're ready when you start the backend.

## Step 3: Set up the Python environment

```
cd notebooklm-local/backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Step 4: Run the backend

From inside `notebooklm-local/backend/` with the venv active:
```
uvicorn main:app --reload
```

The API runs at http://localhost:8000. Leave this terminal open.

## Step 5: Open the frontend

Open `notebooklm-local/frontend/index.html` directly in your browser (file:// works — no dev server needed):
```
# Windows
start notebooklm-local\frontend\index.html

# macOS
open notebooklm-local/frontend/index.html
```

## Usage

1. Click **Choose File**, pick a `.pdf`, `.txt`, or `.md` file, then click **Upload**.
2. Wait for the "N chunks indexed" confirmation (embedding takes a few seconds).
3. Type a question in the chat box and press **Enter** or **Send**.
4. The answer appears with **Sources** showing which file and chunk the answer came from.

---

## Hardware portability note

The same code runs unchanged on both the Windows RTX 3060 dev machine and the target 16 GB M1 MacBook Air. Ollama automatically selects CUDA on Windows/Linux and Metal (MPS) on Apple Silicon — no code changes, no environment variables, no platform-specific paths. Just re-run `ollama pull` on the new machine on first use.

### Memory budget for M1 16 GB

| Component | VRAM / RAM |
|-----------|-----------|
| Qwen2.5 7B Q4_K_M | ~4.5 GB |
| nomic-embed-text | ~0.3 GB |
| OS + browser overhead | ~3–4 GB |
| **Available for context** | **~7–8 GB** |

This leaves comfortable headroom for an 8–16 K token context window, which covers typical document Q&A workloads well.

---

## Project layout

```
notebooklm-local/
  backend/
    main.py          # FastAPI app (upload, documents, chat endpoints)
    rag.py           # chunking, embedding, ChromaDB retrieval, Ollama calls
    requirements.txt
  frontend/
    index.html       # single-page UI — upload + chat, no build step
  data/
    uploads/         # (not currently used for storage, files embedded directly)
    chroma/          # ChromaDB persistent vector store (auto-created on first run)
  README.md
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Upload a file; returns `{filename, chunks}` |
| GET | `/documents` | List indexed documents with chunk counts |
| POST | `/chat` | `{question, filenames?}` → `{answer, sources}` |
