"""Print the structure of every file in data/raw so you can see the real schema.
Usage: python -m qurangpt.ingestion.inspect_data
"""
import gzip
import json
from pathlib import Path

from qurangpt.config import RAW


def shape(o, depth=0, maxd=3):
    pad = "  " * depth
    if isinstance(o, dict):
        for k, v in list(o.items())[:12]:
            n = f" (len {len(v)})" if isinstance(v, (dict, list)) else ""
            print(f"{pad}{k}: {type(v).__name__}{n}")
            if depth < maxd and isinstance(v, (dict, list)):
                shape(v, depth + 1, maxd)
    elif isinstance(o, list) and o:
        print(f"{pad}[0] -> {type(o[0]).__name__}")
        if depth < maxd:
            shape(o[0], depth + 1, maxd)


def load(p: Path):
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    for p in sorted(RAW.rglob("*")):
        if p.name.endswith((".gz", ".json")):
            print("\n=====", p.name)
            try:
                shape(load(p))
            except Exception as e:
                print("  could not read:", e)
