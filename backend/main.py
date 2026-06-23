import os
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")

import asyncio
import json
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import database
import rag
import training

database.init_db()

app = FastAPI(title="LocalNotebookLM", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE = "http://localhost:11434"
REQUIRED_MODELS = ["qwen2.5:7b-instruct-q4_K_M", "nomic-embed-text"]


# ── Health / status ──────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
            tags = resp.json()
    except Exception:
        return {"ollama_running": False, "models": {m: False for m in REQUIRED_MODELS}}

    available_names = {m["name"] for m in tags.get("models", [])}
    models = {}
    for required in REQUIRED_MODELS:
        base = required.split(":")[0]
        models[required] = any(
            a == required or a.startswith(base + ":") for a in available_names
        )
    return {"ollama_running": True, "models": models}


@app.post("/api/models/pull/{model:path}")
async def pull_model(model: str):
    async def _stream() -> AsyncGenerator[str, None]:
        try:
            async with httpx.AsyncClient(timeout=7200.0) as client:
                async with client.stream(
                    "POST", f"{OLLAMA_BASE}/api/pull", json={"name": model}
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line.strip():
                            yield f"data: {line}\n\n"
            yield 'data: {"status":"done"}\n\n'
        except Exception as e:
            yield f'data: {{"error": "{str(e)}"}}\n\n'

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ── Notebooks ─────────────────────────────────────────────────────────────────

class _NotebookBody:
    pass

from pydantic import BaseModel


class NotebookCreate(BaseModel):
    name: str


class NotebookRename(BaseModel):
    name: str


@app.get("/api/notebooks")
def list_notebooks():
    return database.list_notebooks()


@app.post("/api/notebooks", status_code=201)
def create_notebook(body: NotebookCreate):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Notebook name cannot be empty.")
    return database.create_notebook(name)


@app.get("/api/notebooks/{nb_id}")
def get_notebook(nb_id: str):
    nb = database.get_notebook(nb_id)
    if not nb:
        raise HTTPException(404, "Notebook not found.")
    database.touch_notebook(nb_id)
    return nb


@app.put("/api/notebooks/{nb_id}")
def rename_notebook(nb_id: str, body: NotebookRename):
    if not database.get_notebook(nb_id):
        raise HTTPException(404, "Notebook not found.")
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Notebook name cannot be empty.")
    database.update_notebook_name(nb_id, name)
    return database.get_notebook(nb_id)


@app.delete("/api/notebooks/{nb_id}", status_code=204)
def delete_notebook(nb_id: str):
    if not database.get_notebook(nb_id):
        raise HTTPException(404, "Notebook not found.")
    rag.delete_collection(nb_id)
    database.delete_notebook(nb_id)


# ── Documents ─────────────────────────────────────────────────────────────────

@app.get("/api/notebooks/{nb_id}/documents")
def list_documents(nb_id: str):
    if not database.get_notebook(nb_id):
        raise HTTPException(404, "Notebook not found.")
    return rag.list_documents(nb_id)


@app.post("/api/notebooks/{nb_id}/upload")
async def upload_document(nb_id: str, file: UploadFile = File(...)):
    if not database.get_notebook(nb_id):
        raise HTTPException(404, "Notebook not found.")

    allowed = {".pdf", ".txt", ".md"}
    ext = ("." + file.filename.rsplit(".", 1)[-1].lower()) if "." in file.filename else ""
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: pdf, txt, md.")

    content = await file.read()
    filename = file.filename

    async def _stream() -> AsyncGenerator[str, None]:
        progress_q: asyncio.Queue[str] = asyncio.Queue()

        async def on_progress(current: int, total: int):
            await progress_q.put(
                json.dumps({"type": "progress", "current": current, "total": total, "filename": filename})
            )

        async def run_ingest():
            try:
                chunks = await rag.ingest_file(nb_id, content, filename, on_progress)
                await progress_q.put(
                    json.dumps({"type": "done", "chunks": chunks, "filename": filename})
                )
            except Exception as exc:
                await progress_q.put(json.dumps({"type": "error", "message": str(exc)}))

        ingest_task = asyncio.create_task(run_ingest())
        try:
            while True:
                item = await progress_q.get()
                yield f"data: {item}\n\n"
                parsed = json.loads(item)
                if parsed.get("type") in ("done", "error"):
                    break
        finally:
            await ingest_task

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.get("/api/notebooks/{nb_id}/training-status")
def get_training_status(nb_id: str):
    if not database.get_notebook(nb_id):
        raise HTTPException(404, "Notebook not found.")
    return {"trained": training.adapter_exists(nb_id)}


@app.post("/api/notebooks/{nb_id}/train")
async def train_notebook_endpoint(nb_id: str):
    if not database.get_notebook(nb_id):
        raise HTTPException(404, "Notebook not found.")
    collection = rag.get_collection(nb_id)
    all_data = collection.get(include=["documents", "metadatas"])
    if not all_data["ids"]:
        raise HTTPException(400, "No documents in notebook. Upload documents first.")
    chunks = all_data["documents"]
    sources = [m["source"] for m in all_data["metadatas"]]

    async def _stream() -> AsyncGenerator[str, None]:
        try:
            async for event in training.train_notebook(nb_id, chunks, sources):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f'data: {json.dumps({"type": "error", "message": str(e)})}\n\n'

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.delete("/api/notebooks/{nb_id}/adapter", status_code=204)
def delete_adapter(nb_id: str):
    if not database.get_notebook(nb_id):
        raise HTTPException(404, "Notebook not found.")
    training.delete_adapter(nb_id)


@app.delete("/api/notebooks/{nb_id}/documents/{filename:path}", status_code=204)
def delete_document(nb_id: str, filename: str):
    if not database.get_notebook(nb_id):
        raise HTTPException(404, "Notebook not found.")
    rag.delete_document(nb_id, filename)


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str


@app.get("/api/notebooks/{nb_id}/chat")
def get_chat_history(nb_id: str):
    if not database.get_notebook(nb_id):
        raise HTTPException(404, "Notebook not found.")
    return database.get_chat_history(nb_id)


@app.post("/api/notebooks/{nb_id}/chat")
async def send_message(nb_id: str, body: ChatRequest):
    if not database.get_notebook(nb_id):
        raise HTTPException(404, "Notebook not found.")
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty.")

    database.add_chat_message(nb_id, "user", question)
    result = await rag.query_notebook(nb_id, question)
    return database.add_chat_message(nb_id, "assistant", result["answer"], result["sources"])


@app.delete("/api/notebooks/{nb_id}/chat", status_code=204)
def clear_chat(nb_id: str):
    if not database.get_notebook(nb_id):
        raise HTTPException(404, "Notebook not found.")
    database.clear_chat_history(nb_id)
