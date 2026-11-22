import json
import re

from qurangpt.config import EMBED_MODEL, INDEX, MAX_PER_BOOK, TOP_K
from qurangpt.preprocessing.arabic import normalize_ar

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_REF = re.compile(r"\b(\d{1,3})\s*[:/]\s*(\d{1,3})\b")


class Retriever:
    def __init__(self, model=None):
        import faiss
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(EMBED_MODEL)
        self.model = model
        self.index = faiss.read_index(str(INDEX / "passages.faiss"))
        with open(INDEX / "passages_meta.jsonl", encoding="utf-8") as f:
            self.meta = [json.loads(l) for l in f if l.strip()]
        assert self.index.ntotal == len(self.meta), "index and meta are out of sync - rebuild the index"
        self.by_ref = {}
        for i, m in enumerate(self.meta):
            for ref in m["ayah_refs"]:
                self.by_ref.setdefault(ref, []).append(i)

    def lookup(self, ref: str):
        """Structured lookup, e.g. '2:255' -> the verse + tafsir chunks that cover that ayah."""
        return [{**self.meta[i], "score": 1.0} for i in self.by_ref.get(ref, [])]

    def search(self, query: str, k: int = TOP_K, max_per_book: int = MAX_PER_BOOK):
        m = _REF.search(query.translate(_AR_DIGITS))
        exact = self.lookup(f"{int(m.group(1))}:{int(m.group(2))}") if m else []
        exact = exact[:k]

        q = self.model.encode([f"query: {normalize_ar(query)}"], normalize_embeddings=True)
        scores, ids = self.index.search(q.astype("float32"), k * 8)   # over-fetch, then diversify

        out, seen, per_book = [], set(), {}
        for h in exact + [{**self.meta[i], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i >= 0]:
            if h["chunk_id"] in seen or per_book.get(h["book_id"], 0) >= max_per_book:
                continue
            seen.add(h["chunk_id"])
            per_book[h["book_id"]] = per_book.get(h["book_id"], 0) + 1
            out.append(h)
            if len(out) >= k:
                break
        return out
