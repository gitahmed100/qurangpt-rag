"""Turn a Quranpedia mushaf file into data/processed/quran_verses.jsonl (one record per ayah).

Usage:  python -m qurangpt.ingestion.load_quran data/raw/mushafs-2.json.gz

The exact schema of the mushaf files was not visible when this was written, so the parser is
tolerant: it walks the whole JSON, finds anything that looks like a verse (a number + Arabic text)
and takes the surah number from the surrounding object. It then CHECKS the result against the
known ayah counts (6236 total). If it cannot find the verses it writes
data/processed/quran_structure.txt - paste that file's content to the assistant.
"""
import gzip
import json
import sys
from pathlib import Path

from qurangpt.config import PROCESSED, RAW
from qurangpt.ingestion.ayah_counts import AYAH_COUNTS
from qurangpt.preprocessing.arabic import normalize_ar

SURAH_KEYS = ("surah", "sura", "chapter", "surah_number", "surah_id", "sura_no", "surah_no", "chapter_id")
AYAH_KEYS = ("ayah", "aya", "verse", "ayah_number", "aya_no", "verse_number", "ayah_no", "verse_id", "number", "index", "n", "id")
TEXT_KEYS = ("text", "ayah_text", "aya_text", "text_uthmani", "uthmani", "verse_text", "content", "ayah_content", "value")
CHILD_KEYS = ("ayahs", "verses", "ayat", "ayas", "items", "data", "content")


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _text_of(d):
    """Verse text from a dict: a direct string field, or the first option/variant that has text."""
    for k in TEXT_KEYS:
        v = d.get(k)
        if isinstance(v, str) and len(v) > 1:
            return v
    for k in ("options", "variants", "texts", "readings"):
        v = d.get(k)
        if isinstance(v, list) and v:
            first = v[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                t = _text_of(first)
                if t:
                    return t
    return None


def _first(d, keys):
    for k in keys:
        if k in d and _int(d[k]) is not None:
            return _int(d[k])
    return None


def walk(o, surah=None, out=None):
    out = {} if out is None else out
    if isinstance(o, dict):
        s_local = _first(o, SURAH_KEYS)
        has_children = any(isinstance(o.get(k), list) for k in CHILD_KEYS)
        # a surah container: has a number-like id and a list of children
        if s_local is None and has_children:
            n = _first(o, ("number", "id", "index", "surah_id"))
            if n and 1 <= n <= 114 and _text_of(o) is None:
                s_local = n
        s = s_local or surah
        text = _text_of(o)
        a = _first(o, AYAH_KEYS)
        if text and a and s and 1 <= s <= 114 and 1 <= a <= AYAH_COUNTS[s - 1]:
            out.setdefault((s, a), text)
            return out
        for v in o.values():
            if isinstance(v, (dict, list)):
                walk(v, s, out)
    elif isinstance(o, list):
        if surah and o and all(isinstance(x, str) for x in o) and len(o) == AYAH_COUNTS[surah - 1]:
            for i, t in enumerate(o, 1):          # a plain list of verse strings
                out.setdefault((surah, i), t)
            return out
        for v in o:
            walk(v, surah, out)
    return out


def structure(o, depth=0, maxd=5):
    pad, lines = "  " * depth, []
    if isinstance(o, dict):
        for k, v in list(o.items())[:12]:
            sample = "" if isinstance(v, (dict, list)) else " = " + repr(v)[:70]
            n = f" (len {len(v)})" if isinstance(v, (dict, list)) else ""
            lines.append(f"{pad}{k}: {type(v).__name__}{n}{sample}")
            if depth < maxd and isinstance(v, (dict, list)):
                lines += structure(v, depth + 1, maxd)
    elif isinstance(o, list) and o:
        lines.append(f"{pad}[0] -> {type(o[0]).__name__}")
        if depth < maxd:
            lines += structure(o[0], depth + 1, maxd)
    return lines


def main(path=None):
    if path is None:
        path = RAW / "mushafs-2.json.gz"
    path = Path(path)
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    verses = walk(data.get("data", data) if isinstance(data, dict) else data)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    if len(verses) < 6000:
        diag = PROCESSED / "quran_structure.txt"
        d = data.get("data", data) if isinstance(data, dict) else data
        diag.write_text("\n".join(structure(d)), encoding="utf-8")
        raise SystemExit(f"Found only {len(verses)} verses (expected 6236). "
                         f"Structure written to {diag} - paste its content to the assistant.")

    missing = [(s, a) for s in range(1, 115) for a in range(1, AYAH_COUNTS[s - 1] + 1) if (s, a) not in verses]
    out = PROCESSED / "quran_verses.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for (s, a), t in sorted(verses.items()):
            f.write(json.dumps({
                "chunk_id": f"quran:{s}:{a}", "book_id": "quran", "book_name": "القرآن الكريم",
                "surah": s, "ayah": a, "ayah_refs": [f"{s}:{a}"],
                "text": t, "text_norm": normalize_ar(t),
            }, ensure_ascii=False) + "\n")
    print(f"{len(verses)} verses -> {out}")
    print("integrity:", "OK, all 6236 present" if not missing else f"MISSING {len(missing)} e.g. {missing[:5]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
