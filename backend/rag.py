import os
import platform
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, Awaitable

import httpx
import vectorstore as _vs

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b-instruct-q4_K_M"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_BATCH = 32


def _vstore_path() -> Path:
    d = Path(os.environ.get("APP_DATA_DIR", str(Path(__file__).parent.parent / "data"))) / "vstore"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_client() -> _vs.Client:
    return _vs.Client(_vstore_path())


def _collection_name(notebook_id: str) -> str:
    return "nb_" + notebook_id.replace("-", "")


def get_collection(notebook_id: str) -> _vs.Collection:
    return _get_client().get_or_create_collection(_collection_name(notebook_id))


def delete_collection(notebook_id: str):
    try:
        _get_client().delete_collection(_collection_name(notebook_id))
    except Exception:
        pass


def _word_chunks(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += size - overlap
    return chunks


def parse_text(file_bytes: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file_bytes.decode("utf-8", errors="replace")


async def embed_texts(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/api/embed",
            json={"model": EMBED_MODEL, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


ProgressCallback = Callable[[int, int], Awaitable[None]]


async def ingest_file(
    notebook_id: str,
    file_bytes: bytes,
    filename: str,
    on_progress: ProgressCallback | None = None,
) -> int:
    text = parse_text(file_bytes, filename)
    chunks = _word_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        return 0

    collection = get_collection(notebook_id)
    upload_time = datetime.now(timezone.utc).isoformat()

    existing = collection.get(where={"source": filename})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    total = len(chunks)
    ids = [f"{notebook_id}::{filename}::chunk{i}" for i in range(total)]
    metadatas = [
        {"source": filename, "chunk_index": i, "upload_time": upload_time}
        for i in range(total)
    ]

    for batch_start in range(0, total, EMBED_BATCH):
        batch_end = min(batch_start + EMBED_BATCH, total)
        embeddings = await embed_texts(chunks[batch_start:batch_end])
        collection.add(
            ids=ids[batch_start:batch_end],
            embeddings=embeddings,
            documents=chunks[batch_start:batch_end],
            metadatas=metadatas[batch_start:batch_end],
        )
        if on_progress:
            await on_progress(batch_end, total)

    return total


def list_documents(notebook_id: str) -> list[dict]:
    collection = get_collection(notebook_id)
    all_items = collection.get(include=["metadatas"])
    if not all_items["ids"]:
        return []

    docs: dict[str, dict] = {}
    for meta in all_items["metadatas"]:
        src = meta["source"]
        if src not in docs:
            docs[src] = {
                "filename": src,
                "chunk_count": 0,
                "upload_time": meta.get("upload_time", ""),
            }
        docs[src]["chunk_count"] += 1
    return sorted(docs.values(), key=lambda d: d["upload_time"], reverse=True)


def delete_document(notebook_id: str, filename: str):
    collection = get_collection(notebook_id)
    existing = collection.get(where={"source": filename})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])


def notebook_chunk_count(notebook_id: str) -> int:
    return get_collection(notebook_id).count()


async def query_notebook(notebook_id: str, question: str, top_k: int = 4) -> dict:
    collection = get_collection(notebook_id)
    total = collection.count()
    if total == 0:
        return {
            "answer": "This notebook has no documents yet. Upload some documents first.",
            "sources": [],
        }

    q_embedding = (await embed_texts([question]))[0]
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=min(top_k, total),
        include=["documents", "metadatas"],
    )

    chunks = results["documents"][0]
    metas = results["metadatas"][0]

    context_parts = []
    for i, (chunk, meta) in enumerate(zip(chunks, metas)):
        context_parts.append(
            f"[Source {i+1}: {meta['source']}, chunk {meta['chunk_index']}]\n{chunk}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "You are a helpful assistant answering questions about documents the user has uploaded "
        "to this notebook. Answer directly and naturally, as if you simply know the material — "
        'never reference "the context," "the provided information," "the document," or similar '
        "meta-phrasing. If the answer isn't in the material below, say so plainly and directly, "
        "without hedging.\n\n"
        "You do not have personal opinions on political, ideological, or contested social topics. "
        "If a question asks for your opinion on such a topic, or if answering would require taking "
        "a side on a contested political or ideological issue, give a neutral, balanced summary of "
        "relevant perspectives instead of an opinion, and note explicitly that you don't take a "
        "position. This applies regardless of what the uploaded documents contain — summarize what "
        "documents say without adopting their stance as your own.\n\n"
        f"Reference material:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    answer = None
    used_adapter = False

    # Use the fine-tuned adapter when available (macOS only)
    if platform.system() == "Darwin":
        try:
            import training
            if training.adapter_exists(notebook_id):
                answer = await training.generate_with_adapter(notebook_id, prompt)
                used_adapter = True
        except Exception:
            pass  # fall through to Ollama

    if answer is None:
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE}/api/chat",
                    json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                answer = resp.json()["message"]["content"]
        except httpx.ConnectError:
            return {
                "answer": "Cannot reach Ollama. Make sure Ollama is running (`ollama serve`).",
                "sources": [],
                "used_adapter": False,
            }

    return {
        "answer": answer,
        "sources": [
            {"filename": m["source"], "chunk_index": m["chunk_index"]} for m in metas
        ],
        "used_adapter": used_adapter,
    }
