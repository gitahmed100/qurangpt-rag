import time

from qurangpt.config import MIN_SCORE, TOP_K
from qurangpt.generation.llm import generate
from qurangpt.generation.prompt import SYSTEM, build_user_message
from qurangpt.retrieval.retriever import Retriever

ABSTAIN = "لا تتوفر لدي أدلة كافية في المصادر المتاحة للإجابة عن هذا السؤال."


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()

    def ask(self, question: str, k: int = TOP_K):
        t0 = time.time()
        hits = self.retriever.search(question, k)
        if not hits or max(h["score"] for h in hits) < MIN_SCORE:
            return {"answer": ABSTAIN, "sources": [], "latency_s": time.time() - t0}
        answer = generate(SYSTEM, build_user_message(question, hits))
        # sources are rendered from retrieved metadata, never from model-written text
        sources = [{"n": n, "book": h["book_name"], "surah": h["surah"], "ayah": h["ayah"],
                    "chunk_id": h["chunk_id"], "text": h["text"], "score": round(h["score"], 3)}
                   for n, h in enumerate(hits, 1)]
        return {"answer": answer, "sources": sources, "latency_s": round(time.time() - t0, 2)}
