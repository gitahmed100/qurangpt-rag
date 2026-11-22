import os

import anthropic

from qurangpt.config import LLM_MODEL

_client = None


def generate(system: str, user: str, max_tokens: int = 1000) -> str:
    """Swap this function to change LLM provider; the rest of the pipeline is unaffected."""
    global _client
    _client = _client or anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    r = _client.messages.create(model=LLM_MODEL, max_tokens=max_tokens, temperature=0,
                                system=system, messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in r.content if b.type == "text")
