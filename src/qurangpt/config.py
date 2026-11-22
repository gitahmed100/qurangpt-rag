import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CHUNKS = ROOT / "data" / "chunks"
INDEX = ROOT / "data" / "index"

# The SAME model must be used when building the index (Colab) and when answering questions.
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")

TOP_K = 5
MAX_PER_BOOK = 2      # diversity: at most this many passages from one book in the final top-k
MIN_SCORE = 0.0       # abstain threshold; 0 = disabled. Calibrate on your eval set before raising it.
