import os
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import rag

app = FastAPI(title="Local NotebookLM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    filenames: list[str] | None = None


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".md"}
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Allowed: pdf, txt, md.")
    content = await file.read()
    try:
        chunks = await rag.ingest(content, file.filename)
    except Exception as e:
        msg = str(e)
        if "connect" in msg.lower() or "connection" in msg.lower():
            raise HTTPException(status_code=503, detail="Cannot reach Ollama. Please make sure Ollama is running (`ollama serve`).")
        raise HTTPException(status_code=500, detail=msg)
    return {"filename": file.filename, "chunks": chunks}


@app.get("/documents")
def documents():
    return rag.list_documents()


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = await rag.query(req.question, req.filenames)
    return result
