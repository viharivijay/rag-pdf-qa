"""
ingest.py
Handles PDF text extraction, chunking, and embedding.
"""

from typing import List, Tuple
import pdfplumber
from sentence_transformers import SentenceTransformer

# Loaded once and reused across requests
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_by_page(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extract text from a PDF, page by page.
    Returns a list of (page_number, text) tuples.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i + 1, text))
    return pages


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping word-based chunks.
    Overlap preserves context across chunk boundaries.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def chunk_pdf(pdf_path: str, chunk_size: int = 400, overlap: int = 50):
    """
    Extract and chunk a PDF, keeping track of which page each chunk came from.
    Returns:
        chunks: List[str]
        metadata: List[dict] with page number for each chunk
    """
    pages = extract_text_by_page(pdf_path)
    chunks = []
    metadata = []

    for page_num, text in pages:
        page_chunks = chunk_text(text, chunk_size, overlap)
        for chunk in page_chunks:
            chunks.append(chunk)
            metadata.append({"page": page_num})

    return chunks, metadata


def embed_chunks(chunks: List[str]):
    """
    Embed a list of text chunks into vectors.
    """
    if not chunks:
        return []
    return _embedding_model.encode(chunks, show_progress_bar=False)


def embed_query(query: str):
    """
    Embed a single query string using the same model as the chunks,
    so they live in the same vector space.
    """
    return _embedding_model.encode([query])[0]
