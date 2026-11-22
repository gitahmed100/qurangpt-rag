"""Integrity checks + coverage + human-readable samples for the tafsir chunks.

Usage:  python -m qurangpt.ingestion.validate_chunks
Writes: data/chunks/validation_report.json   (numbers, for your report)
        data/chunks/samples.txt              (random chunks per book, read them in VS Code)
"""
import json
import random
import re
from collections import Counter, defaultdict

from qurangpt.config import CHUNKS
from qurangpt.ingestion.ayah_counts import valid_ref

REQUIRED = ["chunk_id", "book_id", "book_name", "surah", "ayah", "ayah_refs", "text", "text_norm"]
HTML = re.compile(r"<[a-zA-Z/][^>]*>|&[a-zA-Z]+;|&#\d+;")
MOJIBAKE = re.compile(r"[ØÙÃ][\u0080-\u00BF]")
FOOT = re.compile(r"\(\s*\d{1,3}\s*\)")


def main(samples_per_book=6, seed=7):
    rows = [json.loads(l) for l in open(CHUNKS / "chunks_all.jsonl", encoding="utf-8") if l.strip()]
    report = {"total_chunks": len(rows), "books": {}}
    ids = Counter(r["chunk_id"] for r in rows)
    report["duplicate_chunk_ids"] = [k for k, v in ids.items() if v > 1][:20]

    by_book = defaultdict(list)
    for r in rows:
        by_book[r["book_id"]].append(r)

    for bid, rs in by_book.items():
        issues = Counter()
        covered = set()
        for r in rs:
            for k in REQUIRED:
                if r.get(k) in (None, "", []):
                    issues[f"missing_{k}"] += 1
            if not valid_ref(r.get("surah"), r.get("ayah")):
                issues["invalid_surah_ayah"] += 1
            for ref in r.get("ayah_refs", []):
                s, a = (int(x) for x in ref.split(":"))
                if valid_ref(s, a):
                    covered.add((s, a))
                else:
                    issues["invalid_ayah_ref"] += 1
            if HTML.search(r["text"]):
                issues["leftover_html"] += 1
            if MOJIBAKE.search(r["text"]):
                issues["mojibake_suspect"] += 1
            if FOOT.search(r["text"]):
                issues["footnote_like_(n)"] += 1
            if len(r["text"]) < 40:
                issues["very_short(<40)"] += 1
        lens = sorted(len(r["text"]) for r in rs)
        report["books"][bid] = {
            "book_name": rs[0]["book_name"],
            "chunks": len(rs),
            "ayah_coverage": f"{len(covered)}/6236 ({100 * len(covered) / 6236:.1f}%)",
            "len_p5_p50_p95": [lens[int(len(lens) * p)] for p in (0.05, 0.5, 0.95)],
            "issues": dict(issues),
        }

    with open(CHUNKS / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    rnd = random.Random(seed)
    with open(CHUNKS / "samples.txt", "w", encoding="utf-8") as f:
        for bid, rs in by_book.items():
            f.write(f"\n{'=' * 70}\n{rs[0]['book_name']}  (book {bid})\n{'=' * 70}\n")
            for r in rnd.sample(rs, min(samples_per_book, len(rs))):
                f.write(f"\n--- {r['chunk_id']}  refs={r['ayah_refs'][:6]}  len={len(r['text'])}\n{r['text']}\n")

    print(f"{len(rows)} chunks")
    for bid, b in report["books"].items():
        print(f"\n[{bid}] book\n  chunks={b['chunks']}  coverage={b['ayah_coverage']}"
              f"  len(p5/p50/p95)={b['len_p5_p50_p95']}\n  issues={b['issues'] or 'none'}")
    if report["duplicate_chunk_ids"]:
        print("\nDUPLICATE chunk_ids:", report["duplicate_chunk_ids"])
    print("\nWrote data/chunks/validation_report.json and data/chunks/samples.txt")


if __name__ == "__main__":
    main()
    