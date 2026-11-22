"""Embed verses + tafsir chunks and build a FAISS index.
Usage: python -m qurangpt.indexing.build_index
"""
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from qurangpt.config import CHUNKS, EMBED_MODEL, INDEX, PROCESSED


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main(batch=64):
    rows = []
    q = PROCESSED / "quran_verses.jsonl"
    if q.exists():
        rows += read_jsonl(q)
    rows += read_jsonl(CHUNKS / "chunks_all.jsonl")
    print(f"{len(rows)} passages")

    model = SentenceTransformer(EMBED_MODEL)
    # e5 models need the "passage: " prefix (queries use "query: ")
    texts = [f"passage: {r['text_norm']}" for r in rows]
    emb = model.encode(texts, batch_size=batch, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])   # exact search; fine up to ~1M vectors
    index.add(emb)
    INDEX.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX / "passages.faiss"))
    with open(INDEX / "passages_meta.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("index saved to", INDEX)


if __name__ == "__main__":
    main()
