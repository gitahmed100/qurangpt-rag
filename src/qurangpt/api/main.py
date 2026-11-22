from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from qurangpt.config import ROOT
from qurangpt.rag.pipeline import RAGPipeline

app = FastAPI(title="QuranGPT")
_pipe = None


class Q(BaseModel):
    question: str


@app.post("/ask")
def ask(q: Q):
    global _pipe
    _pipe = _pipe or RAGPipeline()
    return _pipe.ask(q.question)


@app.get("/")
def home():
    return FileResponse(ROOT / "frontend" / "index.html")
