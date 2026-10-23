
"""
Embedding Module
Uses a local Sentence Transformers model for embeddings.
"""

from typing import List
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Downloads the model the first time; runs locally afterward.
model = SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Generate one embedding vector for each document chunk."""
    if not chunks:
        return []

    vectors = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return vectors.tolist()


def embed_user_query(query: str) -> List[float]:
    """Generate an embedding vector for a user's question."""
    vector = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return vector.tolist()
