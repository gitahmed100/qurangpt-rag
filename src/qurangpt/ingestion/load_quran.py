"""Turn the Quran text file from the bundle into data/processed/quran_verses.jsonl
(one record per ayah). The exact schema is not known to the author of this scaffold,
so this loader is heuristic: run `inspect_data` first, then adjust KEYS below if needed.

Usage: python -m qurangpt.ingestion.load_quran [path/to/quran-file.json.gz]
"""
import gzip
import json
import sys
from pathlib import Path

from qurangpt.config import PROCESSED, RAW
from qurangpt.preprocessing.arabic import normalize_ar

KEYS = {
    "surah": ("surah", "sura", "chapter", "surah_number", "sura_no"),
    "ayah": ("ayah", "aya", "verse", "ayah_number", "aya_no"),
    "text": ("text", "ayah_text", "aya_text", "text_uthmani", "uthmani", "content"),
}


def _pick(d, names):
    for n in names:
        if n in d and d[n] not in (None, ""):
            return d[n]
    return None


def _walk(o):
    """Yield every dict that looks like a verse record."""
    if isinstance(o, dict):
        s, a, t = (_pick(o, KEYS[k]) for k in ("surah", "ayah", "text"))
        if isinstance(t, str) and isinstance(s, (int, str)) and isinstance(a, (int, str)):
            yield {"surah": int(s), "ayah": int(a), "text": t}
            return
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def main(path=None):
    if path is None:
        cands = [p for p in RAW.rglob("*") if "quran" in p.name.lower() and "tafsir" not in p.name.lower()
                 and p.name.endswith((".json", ".gz"))]
        if not cands:
            raise SystemExit("No quran file found in data/raw. Pass the path explicitly.")
        path = cands[0]
    path = Path(path)
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    verses = {(v["surah"], v["ayah"]): v for v in _walk(data)}
    if not verses:
        raise SystemExit("Could not find verse records. Run inspect_data and edit KEYS in load_quran.py")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "quran_verses.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for (s, a), v in sorted(verses.items()):
            f.write(json.dumps({
                "chunk_id": f"quran:{s}:{a}", "book_id": "quran", "book_name": "القرآن الكريم",
                "surah": s, "ayah": a, "ayah_refs": [f"{s}:{a}"],
                "text": v["text"], "text_norm": normalize_ar(v["text"]),
            }, ensure_ascii=False) + "\n")
    print(f"{len(verses)} verses -> {out}  (expected 6236)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
