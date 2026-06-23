"""
Lightweight vector store — drop-in replacement for chromadb.
Uses numpy for cosine similarity; falls back to pure Python if numpy is absent.
Persists metadata as JSON and embeddings as .npy binary files.
"""

import json
import math
import shutil
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False


def _cosine_py(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-10) if na > 0 and nb > 0 else 0.0


class Collection:
    def __init__(self, store_dir: Path):
        self._dir = store_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = store_dir / "meta.json"
        self._emb_path = store_dir / "embeddings.npy"
        self._load()

    def _load(self):
        if self._meta_path.exists():
            d = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._ids: list[str] = d.get("ids", [])
            self._docs: list[str] = d.get("documents", [])
            self._metas: list[dict] = d.get("metadatas", [])
        else:
            self._ids, self._docs, self._metas = [], [], []

        if _NP:
            if self._emb_path.exists() and self._ids:
                self._emb = np.load(str(self._emb_path))
            else:
                self._emb = np.zeros((0, 768), dtype=np.float32)
        else:
            self._emb_list: list[list[float]] = []

    def _save(self):
        self._meta_path.write_text(
            json.dumps({"ids": self._ids, "documents": self._docs, "metadatas": self._metas}),
            encoding="utf-8",
        )
        if _NP:
            np.save(str(self._emb_path), self._emb)

    def count(self) -> int:
        return len(self._ids)

    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]):
        if _NP:
            new = np.array(embeddings, dtype=np.float32)
            self._emb = new if self._emb.shape[0] == 0 else np.vstack([self._emb, new])
        else:
            self._emb_list.extend(embeddings)
        self._ids.extend(ids)
        self._docs.extend(documents)
        self._metas.extend(metadatas)
        self._save()

    def get(self, where: Optional[dict] = None, include: Optional[list] = None) -> dict:
        if where:
            idx = [i for i, m in enumerate(self._metas) if all(m.get(k) == v for k, v in where.items())]
        else:
            idx = list(range(len(self._ids)))
        result: dict = {"ids": [self._ids[i] for i in idx]}
        if not include or "documents" in include:
            result["documents"] = [self._docs[i] for i in idx]
        if not include or "metadatas" in include:
            result["metadatas"] = [self._metas[i] for i in idx]
        return result

    def delete(self, ids: Optional[list] = None, where: Optional[dict] = None):
        if where:
            remove = {i for i, m in enumerate(self._metas) if all(m.get(k) == v for k, v in where.items())}
        elif ids:
            id_set = set(ids)
            remove = {i for i, id_ in enumerate(self._ids) if id_ in id_set}
        else:
            return
        if not remove:
            return
        keep = [i for i in range(len(self._ids)) if i not in remove]
        self._ids   = [self._ids[i]  for i in keep]
        self._docs  = [self._docs[i] for i in keep]
        self._metas = [self._metas[i] for i in keep]
        if _NP:
            dim = self._emb.shape[1] if self._emb.shape[0] > 0 else 768
            self._emb = self._emb[keep] if keep else np.zeros((0, dim), dtype=np.float32)
        else:
            self._emb_list = [self._emb_list[i] for i in keep]
        self._save()

    def query(self, query_embeddings: list, n_results: int, include: Optional[list] = None) -> dict:
        n = len(self._ids)
        if n == 0:
            return {"documents": [[]], "metadatas": [[]]}
        q = query_embeddings[0]
        k = min(n_results, n)
        if _NP:
            q_arr = np.array(q, dtype=np.float32)
            norms = np.linalg.norm(self._emb, axis=1)
            q_norm = np.linalg.norm(q_arr)
            sims = (self._emb @ q_arr) / (norms * q_norm + 1e-10)
            top = np.argsort(sims)[::-1][:k].tolist()
        else:
            sims = [_cosine_py(e, q) for e in self._emb_list]
            top = sorted(range(n), key=lambda i: sims[i], reverse=True)[:k]
        result: dict = {"documents": [[self._docs[i] for i in top]]}
        if not include or "metadatas" in include:
            result["metadatas"] = [[self._metas[i] for i in top]]
        return result


class Client:
    def __init__(self, base: Path):
        self._base = base

    def get_or_create_collection(self, name: str, **_) -> Collection:
        return Collection(self._base / name)

    def delete_collection(self, name: str):
        d = self._base / name
        if d.exists():
            shutil.rmtree(d)
