"""Recall@5 and MRR on data/eval/questions.jsonl
Each line: {"question": "...", "gold_refs": ["2:255", "2:256"]}
A retrieved chunk is a hit if any of its ayah_refs is in gold_refs.
Usage: python -m qurangpt.evaluation.eval_retrieval
"""
import json

from qurangpt.config import ROOT
from qurangpt.retrieval.retriever import Retriever


def main(k=5):
    qs = [json.loads(l) for l in open(ROOT / "data" / "eval" / "questions.jsonl", encoding="utf-8") if l.strip()]
    r = Retriever()
    rec = mrr = 0.0
    for q in qs:
        gold = set(q["gold_refs"])
        hits = r.search(q["question"], k)
        rank = next((i for i, h in enumerate(hits, 1) if gold & set(h["ayah_refs"])), None)
        rec += rank is not None
        mrr += 1 / rank if rank else 0
    n = len(qs)
    print(f"N={n}  Recall@{k}={rec / n:.3f}  MRR={mrr / n:.3f}")


if __name__ == "__main__":
    main()
