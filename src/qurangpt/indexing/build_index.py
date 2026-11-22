"""Build the search index. Self-contained: needs only numpy + sentence-transformers (+ faiss for --faiss).

Two steps, so the slow one (embedding) can run on a free Colab GPU:
  --embed   passages -> embeddings.npy   (float16, slow on CPU, fast on GPU)
  --faiss   embeddings.npy -> passages.faiss + passages_meta.jsonl   (seconds, run locally)

Local, everything:   python -m qurangpt.indexing.build_index --embed --faiss
Colab (embed only):  python build_index.py --embed --chunks chunks_all.jsonl.gz --quran quran_verses.jsonl --out out
Local (after Colab): put embeddings.npy in data/index/, then  python -m qurangpt.indexing.build_index --faiss

Row order is fixed: Quran verses first, then tafsir chunks in file order. The meta file written by
--faiss uses the same order, so row i of the index == line i of passages_meta.jsonl.
"""
import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")


def read_jsonl(p):
    p = Path(p)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_rows(chunks, quran):
    verses = read_jsonl(quran) if quran and Path(quran).exists() else []
    rows = verses + read_jsonl(chunks)
    return rows, {r["ayah_refs"][0]: r["text_norm"] for r in verses}


def embed_text(r, verse_by_ref):
    """What gets embedded. Tafsir chunks get their verse in front: a chunk from the middle of a long
    commentary otherwise has no hint of which ayah it explains."""
    if r["book_id"] != "quran":
        v = verse_by_ref.get(r["ayah_refs"][0], "")[:300]
        if v:
            return f"passage: الآية: {v}\nالتفسير: {r['text_norm']}"
    return f"passage: {r['text_norm']}"


def do_embed(rows, verse_by_ref, out_dir, model_name, batch):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    model.max_seq_length = 512
    texts = [embed_text(r, verse_by_ref) for r in rows]
    # sort by length: much faster batches; restore the original order afterwards
    order = np.argsort([len(t) for t in texts])
    emb = model.encode([texts[i] for i in order], batch_size=batch, show_progress_bar=True,
                       normalize_embeddings=True)
    out = np.zeros_like(emb)
    out[order] = emb
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", out.astype("float16"))
    (out_dir / "embeddings_info.json").write_text(
        json.dumps({"rows": len(rows), "model": model_name, "dim": int(out.shape[1])}), encoding="utf-8")
    print("saved", out_dir / "embeddings.npy", out.shape)


def do_faiss(rows, out_dir):
    import faiss
    emb = np.load(out_dir / "embeddings.npy").astype("float32")
    assert len(emb) == len(rows), f"embeddings ({len(emb)}) != passages ({len(rows)}): rebuild with the same chunk files"
    index = faiss.IndexFlatIP(emb.shape[1])     # exact cosine search (vectors are normalized)
    index.add(emb)
    faiss.write_index(index, str(out_dir / "passages.faiss"))
    with open(out_dir / "passages_meta.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            r = {k: v for k, v in r.items() if k != "text_norm"}   # keep the meta file small
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("index:", index.ntotal, "passages ->", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--faiss", action="store_true")
    ap.add_argument("--chunks", default=str(ROOT / "data/chunks/chunks_all.jsonl"))
    ap.add_argument("--quran", default=str(ROOT / "data/processed/quran_verses.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/index"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    if not (a.embed or a.faiss):
        raise SystemExit("choose --embed and/or --faiss")
    rows, verse_by_ref = load_rows(a.chunks, a.quran)
    print(len(rows), "passages")
    out = Path(a.out)
    if a.embed:
        do_embed(rows, verse_by_ref, out, a.model, a.batch)
    if a.faiss:
        do_faiss(rows, out)


if __name__ == "__main__":
    main()
