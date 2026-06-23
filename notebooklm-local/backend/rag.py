import re
import httpx
import chromadb
from pathlib import Path
from datetime import datetime, timezone

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma"
CHUNK_SIZE = 500   # approximate tokens (words * 1.3)
CHUNK_OVERLAP = 50

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = _client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _word_chunks(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
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


async def ingest(file_bytes: bytes, filename: str) -> int:
    text = parse_text(file_bytes, filename)
    raw_chunks = _word_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not raw_chunks:
        return 0

    collection = _get_collection()
    upload_time = datetime.now(timezone.utc).isoformat()

    # delete any existing chunks for this document so re-upload is clean
    existing = collection.get(where={"source": filename})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    embeddings = await embed_texts(raw_chunks)

    ids = [f"{filename}::chunk{i}" for i in range(len(raw_chunks))]
    metadatas = [
        {"source": filename, "chunk_index": i, "upload_time": upload_time}
        for i in range(len(raw_chunks))
    ]
    collection.add(ids=ids, embeddings=embeddings, documents=raw_chunks, metadatas=metadatas)
    return len(raw_chunks)


def list_documents() -> list[dict]:
    collection = _get_collection()
    all_items = collection.get(include=["metadatas"])
    if not all_items["ids"]:
        return []

    docs: dict[str, dict] = {}
    for meta in all_items["metadatas"]:
        src = meta["source"]
        if src not in docs:
            docs[src] = {"filename": src, "chunk_count": 0, "upload_time": meta.get("upload_time", "")}
        docs[src]["chunk_count"] += 1
    return sorted(docs.values(), key=lambda d: d["upload_time"], reverse=True)


async def query(question: str, filenames: list[str] | None = None, top_k: int = 4) -> dict:
    collection = _get_collection()
    total = collection.count()
    if total == 0:
        return {"answer": "No documents have been uploaded yet. Please upload a document first.", "sources": []}

    q_embedding = (await embed_texts([question]))[0]

    query_kwargs: dict = {
        "query_embeddings": [q_embedding],
        "n_results": min(top_k, total),
        "include": ["documents", "metadatas"],
    }
    if filenames:
        query_kwargs["where"] = {"source": {"$in": filenames}}
    results = collection.query(**query_kwargs)

    chunks = results["documents"][0]
    metas = results["metadatas"][0]

    context_parts = []
    for i, (chunk, meta) in enumerate(zip(chunks, metas)):
        context_parts.append(f"[Source {i+1}: {meta['source']}, chunk {meta['chunk_index']}]\n{chunk}")
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant that answers questions strictly based on the provided context. "
        "Do not use any prior knowledge. If the context does not contain the answer, say so. "
        "At the end of your answer, cite which sources you used by referencing their [Source N] labels."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": "qwen2.5:7b-instruct-q4_K_M",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
            )
            resp.raise_for_status()
            answer = resp.json()["message"]["content"]
    except httpx.ConnectError:
        return {"answer": "Cannot reach Ollama. Please make sure Ollama is running (`ollama serve`).", "sources": []}

    sources = [
        {"filename": m["source"], "chunk_index": m["chunk_index"]}
        for m in metas
    ]
    return {"answer": answer, "sources": sources}
