"""
prompt.py
Builds the grounded prompt sent to the LLM, using retrieved chunks as context.
"""

from typing import List, Dict, Tuple


SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context from a document. If the answer cannot be found in "
    "the context, say 'I couldn't find that in the document' instead of "
    "guessing. Keep answers concise and cite the page number(s) you used."
)


def build_prompt(query: str, retrieved: List[Tuple[str, Dict, float]]) -> str:
    """
    Combine retrieved chunks with the user's question into a single
    grounded prompt for the LLM.
    """
    if not retrieved:
        context_block = "No relevant context was found in the document."
    else:
        parts = []
        for chunk, meta, _ in retrieved:
            page = meta.get("page", "?")
            parts.append(f"[Page {page}]\n{chunk}")
        context_block = "\n\n---\n\n".join(parts)

    prompt = f"""{SYSTEM_INSTRUCTION}

Context:
{context_block}

Question: {query}

Answer:"""
    return prompt
