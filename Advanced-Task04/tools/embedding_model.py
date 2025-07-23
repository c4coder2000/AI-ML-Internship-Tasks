# tools/embedding_model.py

from sentence_transformers import SentenceTransformer
from typing import List

# 🔹 Load model once during import (efficient!)
_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str, verbose: bool = False) -> List[float]:
    """
    Generates embedding vector for the input text using Hugging Face's SentenceTransformer.

    Args:
        text (str): Input string to embed.
        verbose (bool): If True, logs embedding stats.

    Returns:
        List[float]: Embedding vector.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    if not text.strip():
        raise ValueError("Cannot generate embedding for empty text.")

    embedding = _MODEL.encode(text).tolist()

    if verbose:
        print(f"✅ Embedded {len(text.split())} words → {len(embedding)}-dim vector")

    return embedding
