# QuranGPT — RAG for Quran & Tafsir (Quranpedia)

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env                                   # add your API key
```

## Pipeline (run in order)
| # | Command | Result |
|---|---------|--------|
| 1 | `python -m qurangpt.ingestion.extract "C:/path/bundle.zip"` | files in `data/raw/` |
| 2 | `python -m qurangpt.ingestion.inspect_data` | prints real JSON schemas — READ IT |
| 3 | `python -m qurangpt.ingestion.chunk_tafsir --books <ids>` | `data/chunks/chunks_all.jsonl` |
| 4 | `python -m qurangpt.ingestion.load_quran` | `data/processed/quran_verses.jsonl` (6236 ayahs) |
| 5 | `python -m qurangpt.indexing.build_index` | FAISS index in `data/index/` |
| 6 | `uvicorn qurangpt.api.main:app --reload` | UI at http://127.0.0.1:8000 |
| 7 | `python -m qurangpt.evaluation.eval_retrieval` | Recall@5, MRR |

Tests: `pytest`. Build the 100+ question eval set in `data/eval/questions.jsonl`.

## Rules
- `text` = original (display/citation). `text_norm` = search only. Never mix.
- Citations are rendered from metadata, never from LLM-written text.
- Check Quranpedia per-file license before redistributing any data.
