import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CHUNKS = ROOT / "data" / "chunks"
INDEX = ROOT / "data" / "index"

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
TOP_K = 5
MIN_SCORE = 0.78   # below this top score -> abstain (tune on your eval set)
