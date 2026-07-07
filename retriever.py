"""
retriever.py
In-memory FAISS vector store and similarity search.
"""

from typing import List, Dict, Tuple
import numpy as np
import faiss

from app.ingest import embed_chunks, embed_query


class VectorStore:
    """
    A minimal in-memory vector store backed by FAISS.
    One instance holds the index for a single uploaded document.
    """

    def __init__(self):
        self.index = None
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []

    def build(self, chunks: List[str], metadata: List[Dict]):
        """Embed chunks and build a fresh FAISS index."""
        self.chunks = chunks
        self.metadata = metadata

        if not chunks:
            self.index = None
            return

        vectors = np.array(embed_chunks(chunks)).astype("float32")
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(vectors)

    def search(self, query: str, k: int = 3) -> List[Tuple[str, Dict, float]]:
        """
        Return the top-k most relevant chunks for a query.
        Each result is (chunk_text, metadata, distance).
        """
        if self.index is None or not self.chunks:
            return []

        query_vector = np.array([embed_query(query)]).astype("float32")
        k = min(k, len(self.chunks))
        distances, indices = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], self.metadata[idx], float(dist)))
        return results

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0
