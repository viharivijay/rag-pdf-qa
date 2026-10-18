"""
llm.py
Thin wrapper around the LLM API call, so the rest of the app doesn't care
which provider is behind it. Defaults to OpenAI; swap the internals to use
Groq, Anthropic, or a local model without touching main.py.
"""

import os
from openai import OpenAI

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def call_llm(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 500) -> str:
    """
    Send the grounded prompt to the LLM and return the text response.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
