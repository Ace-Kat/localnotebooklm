# LocalNotebookLM

A fully local, offline-first document Q&A desktop app. Create notebooks, upload documents (PDF, TXT, MD), and chat with them — all running on your machine. No cloud accounts, no API keys, no internet required.

**Architecture:** Tauri desktop shell + Python FastAPI sidecar + ChromaDB (vector store) + SQLite (metadata/chat) + Ollama (LLM + embeddings).

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Rust + Cargo | stable | https://rustup.rs |
| Node.js | 18+ | https://nodejs.org |
| Python | 3.10+ | https://python.org |
| Ollama | latest | https://ollama.ai |

**Windows only:** also install the [Microsoft C++ Build Tools](https://aka.ms/buildtools) (required by Rust).

**macOS only:** install Xcode Command Line Tools — `xcode-select --install`.

---

## Ollama models

The app requires two models (~5 GB total download, one-time per machine):

```
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull nomic-embed-text
```

The first-run screen will guide you through this if models are missing.

---

## Development setup

### 1. Python backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\pip install -r requirements.txt

# macOS / Linux
.venv/bin/pip install -r requirements.txt
```

### 2. Tauri CLI

```bash
npm install
```

### 3. Icons (one-time)

```bash
python scripts/create_icons.py
```

For production-quality icons, provide a 1024×1024 PNG and run:

```bash
cargo tauri icon assets/your-icon.png
```

### 4. Run in development

Start Ollama (if not already running as a system service):

```bash
ollama serve
```

Launch the app (Tauri automatically starts the Python backend):

```bash
npx tauri dev
```

The Tauri window will open. On first run, the app checks for Ollama and will prompt you to pull any missing models.

---

## Building installers

### Windows (.msi / .exe)

Run on the Windows machine:

```bash
npx tauri build
```

Installer output: `src-tauri/target/release/bundle/`

### macOS (.dmg)

**Must be run on macOS hardware** — cross-compilation from Windows is not supported by Tauri/Apple tooling.

```bash
npx tauri build
```

Installer output: `src-tauri/target/release/bundle/dmg/`

> **Note:** The macOS `.dmg` installer must be built on macOS. If you need both Windows and macOS installers, build on each platform separately (or use macOS CI, e.g. GitHub Actions with a `macos-latest` runner).

---

## Production: bundling the Python backend

For distribution, the Python backend should be compiled into a standalone executable using PyInstaller so end-users don't need Python installed.

### Install PyInstaller

```bash
cd backend
.venv\Scripts\pip install pyinstaller   # Windows
# or
.venv/bin/pip install pyinstaller        # macOS/Linux
```

### Create the executable

```bash
# Windows
.venv\Scripts\pyinstaller --onefile --name backend main.py

# macOS/Linux
.venv/bin/pyinstaller --onefile --name backend main.py
```

The frozen binary lands in `backend/dist/backend` (or `backend/dist/backend.exe` on Windows).

### Wire it into Tauri

In `src-tauri/tauri.conf.json`, add to `bundle`:

```json
"externalBin": ["../backend/dist/backend"]
```

Then `npx tauri build` will bundle it. The Rust `lib.rs` code already looks for the bundled binary at `resources/backend[.exe]` before falling back to Python.

---

## Data storage

All notebooks, documents, embeddings, and chat history are stored in the platform app-data directory:

- **Windows:** `%APPDATA%\com.localnotebooklm.app\`
- **macOS:** `~/Library/Application Support/com.localnotebooklm.app/`

During development (without Tauri), data lives in `data/` at the project root.

---

## Capacity

Each notebook can hold at least **5 GB** of source documents. Uploads are processed incrementally (chunk-and-embed per document, not all-in-memory), so memory usage stays flat regardless of notebook size. A progress indicator shows embedding status during large uploads.

---

## Stretch goals / out of scope for v1

- `.docx` (Word) document parsing
- Multi-user accounts or login
- Cloud sync or backup
- Auto-update mechanism
- Mobile support
