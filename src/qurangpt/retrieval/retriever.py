import json
import re

import faiss
from sentence_transformers import SentenceTransformer

from qurangpt.config import EMBED_MODEL, INDEX, TOP_K
from qurangpt.preprocessing.arabic import normalize_ar

_REF = re.compile(r"\b(\d{1,3})\s*[:/]\s*(\d{1,3})\b")


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.read_index(str(INDEX / "passages.faiss"))
        with open(INDEX / "passages_meta.jsonl", encoding="utf-8") as f:
            self.meta = [json.loads(l) for l in f if l.strip()]
        self.by_ref = {}
        for i, m in enumerate(self.meta):
            for ref in m["ayah_refs"]:
                self.by_ref.setdefault(ref, []).append(i)

    def lookup(self, ref: str):
        """Structured lookup, e.g. '2:255' -> verse + tafsir chunks for that ayah."""
        return [{**self.meta[i], "score": 1.0} for i in self.by_ref.get(ref, [])]

    def search(self, query: str, k: int = TOP_K):
        m = _REF.search(query)
        exact = self.lookup(f"{int(m.group(1))}:{int(m.group(2))}") if m else []
        q = self.model.encode([f"query: {normalize_ar(query)}"], normalize_embeddings=True)
        scores, ids = self.index.search(q.astype("float32"), k)
        hits = [{**self.meta[i], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i >= 0]
        seen, out = set(), []
        for h in exact + hits:
            if h["chunk_id"] not in seen:
                seen.add(h["chunk_id"])
                out.append(h)
        return out[: max(k, len(exact))]
