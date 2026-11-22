"""
Split Quranpedia tafsir dumps (tafsir-book-*.json.gz) into RAG-ready chunks.

Usage (from project root):
    python -m qurangpt.ingestion.chunk_tafsir                 # all books in data/raw
    python -m qurangpt.ingestion.chunk_tafsir --books 1 7 12  # only some book ids
    python -m qurangpt.ingestion.chunk_tafsir --max-chars 1200 --overlap 150

Output (data/chunks): chunks_<book_id>.jsonl, chunks_all.jsonl, stats.json

Changes vs. the first version:
  * only reads tafsir-book-*.json[.gz]  (the bundle also holds quran/ayah/metadata files)
  * duplicate text shared by neighbouring ayahs is kept ONCE but remembers every ayah it
    covers ("ayah_refs"), so lookup by any ayah still finds it (nothing is lost)
  * dedupe is per-surah (identical short boilerplate in different surahs is not merged)
  * display text ("text") and search text ("text_norm") stored separately, always
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import re

from bs4 import BeautifulSoup
from tqdm import tqdm

from qurangpt.config import CHUNKS, RAW
from qurangpt.preprocessing.arabic import normalize_ar

# ---------------------------------------------------------------- HTML cleaning


def html_to_text(html: str, keep_footnotes: bool = False) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if not keep_footnotes:
        for tag in soup.select("div.foot-notes"):
            tag.decompose()
    for tag in soup.select("a.foot-note"):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(["p", "div", "tr", "hr"]):
        tag.append("\n")
    text = soup.get_text()
    text = text.replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


# ---------------------------------------------------------------- chunking

_SENT_SPLIT = re.compile(r"(?<=[.!؟?؛])\s+")


def _split_long(paragraph: str, max_chars: int):
    out, cur = [], ""
    for sent in _SENT_SPLIT.split(paragraph):
        while len(sent) > max_chars:
            cut = sent.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars * 0.5 else max_chars
            if cur:
                out.append(cur)
                cur = ""
            out.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if len(cur) + len(sent) + 1 <= max_chars:
            cur = f"{cur} {sent}".strip()
        else:
            if cur:
                out.append(cur)
            cur = sent
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150):
    pieces = []
    for p in (p.strip() for p in text.split("\n")):
        if not p:
            continue
        pieces.extend(_split_long(p, max_chars) if len(p) > max_chars else [p])

    chunks, cur = [], ""
    for p in pieces:
        if not cur:
            cur = p
        elif len(cur) + len(p) + 1 <= max_chars:
            cur += "\n" + p
        else:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            if tail and " " in tail:
                tail = tail[tail.find(" ") + 1:]
            cur = f"{tail}\n{p}".strip() if tail else p
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------- IO


def load_book(path: str) -> dict:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def process_book(path, out_dir, max_chars, overlap, keep_footnotes, min_chars):
    data = load_book(path)
    book = data["book"]
    book_id = str(book["id"])
    author = book.get("author") or {}
    author_name = author.get("ar_name") if isinstance(author, dict) else None
    book_name = book.get("short_name") or book.get("name")

    seen = {}          # (surah, hash) -> row
    rows = []
    n_shared = n_empty = 0

    for entry in data["ayahs"]:
        items = entry.get("content") or []
        html = "\n".join((it.get("text") or "") for it in items)
        text = html_to_text(html, keep_footnotes)
        if len(text) < min_chars:
            n_empty += 1
            continue

        surah, ayah = entry["surah"], entry["ayah"]
        first = items[0] if items else {}
        for i, ch in enumerate(chunk_text(text, max_chars, overlap)):
            norm = normalize_ar(ch)
            key = (surah, hashlib.md5(norm.encode("utf-8")).hexdigest())
            if key in seen:                       # same text already stored for a nearby ayah
                ref = f"{surah}:{ayah}"
                if ref not in seen[key]["ayah_refs"]:
                    seen[key]["ayah_refs"].append(ref)
                n_shared += 1
                continue
            row = {
                "chunk_id": f"{book_id}:{surah}:{ayah}:{i}",
                "book_id": book_id,
                "book_name": book_name,
                "author": author_name,
                "surah": surah,
                "ayah": ayah,
                "ayah_refs": [f"{surah}:{ayah}"],
                "part": first.get("part"),
                "page": first.get("page"),
                "chunk_index": i,
                "text": ch,            # original, for display / citation
                "text_norm": norm,     # normalized, for embedding / lexical search
            }
            seen[key] = row
            rows.append(row)

    with open(os.path.join(out_dir, f"chunks_{book_id}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = [len(r["text"]) for r in rows] or [0]
    return rows, {
        "book_id": book_id, "book_name": book_name,
        "ayah_entries": len(data["ayahs"]),
        "empty_entries_skipped": n_empty,
        "shared_text_merged": n_shared,
        "chunks": len(rows),
        "chars_min_avg_max": [min(lens), sum(lens) // len(lens), max(lens)],
    }


def find_tafsir_files(folder, books=None):
    files = sorted(glob.glob(os.path.join(folder, "**", "tafsir-book-*.json*"), recursive=True))
    if books:
        wanted = {str(b) for b in books}
        files = [f for f in files if re.search(r"tafsir-book-(\d+)", os.path.basename(f)).group(1) in wanted]
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(RAW))
    ap.add_argument("--output", default=str(CHUNKS))
    ap.add_argument("--books", nargs="*", help="only these book ids")
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--keep-footnotes", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.output, exist_ok=True)
    files = find_tafsir_files(a.input, a.books)
    if not files:
        raise SystemExit(f"No tafsir-book-*.json.gz files found under {a.input}")

    stats = []
    with open(os.path.join(a.output, "chunks_all.jsonl"), "w", encoding="utf-8") as allf:
        for path in tqdm(files, desc="books"):
            try:
                rows, st = process_book(path, a.output, a.max_chars, a.overlap, a.keep_footnotes, a.min_chars)
            except Exception as e:                       # one bad file must not kill the run
                print(f"!! skipped {os.path.basename(path)}: {e}")
                continue
            for r in rows:
                allf.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats.append(st)

    with open(os.path.join(a.output, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Done: {sum(s['chunks'] for s in stats)} chunks from {len(stats)} books -> {a.output}")


if __name__ == "__main__":
    main()
