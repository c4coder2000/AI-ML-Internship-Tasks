# tools/pinecone_tool.py

import os

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer



# ✅ Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("prompt-refiner")

# ✅ Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def input_vector(text: str, top_k: int = 5) -> list:
    vector = embedder.encode(text).tolist()
    response = index.query(vector=vector, top_k=top_k, include_metadata=True, namespace="corpus_chunks")

    return [
        {
            "content": match["metadata"].get("content", ""),
            "score": match["score"],
            "metadata": match["metadata"]
        }
        for match in response["matches"]
    ]
