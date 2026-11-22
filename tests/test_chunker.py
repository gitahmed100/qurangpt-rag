import gzip, json
from qurangpt.ingestion import chunk_tafsir as ct


def test_chunk_sizes_and_overlap():
    text = "\n".join(f"هذه جملة رقم {i} في التفسير. " * 5 for i in range(40))
    chunks = ct.chunk_text(text, 400, 60)
    assert len(chunks) > 3 and max(map(len, chunks)) <= 400 + 80


def test_shared_text_keeps_all_ayahs(tmp_path):
    html = "<p>" + "تفسير مشترك بين الآيات. " * 10 + "</p>"
    ayahs = [{"surah": 1, "ayah": a, "content": [{"text": html, "part": 1, "page": 1}]} for a in (1, 2, 3)]
    p = tmp_path / "tafsir-book-9.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump({"book": {"id": 9, "short_name": "T"}, "ayahs": ayahs}, f)
    rows, st = ct.process_book(str(p), str(tmp_path), 1200, 150, False, 20)
    assert len(rows) == 1 and rows[0]["ayah_refs"] == ["1:1", "1:2", "1:3"]
