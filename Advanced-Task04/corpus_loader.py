# pinecone_loader.py

import os
from pinecone import Pinecone
from tools.embedding_model import get_embedding

api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
index = pc.Index("prompt-refiner")

documents = [
    {
        "id": "doc_001",
        "content": "Use benefit-focused language to make prompts persuasive.",
        "metadata": {"content": "Use benefit-focused language to make prompts persuasive.", "source": "prompt_tips.txt"}
    },
    {
        "id": "doc_002",
        "content": "Remove unnecessary adjectives and fluff for better clarity.",
        "metadata": {"content": "Remove unnecessary adjectives and fluff for better clarity.", "source": "editing_guide.txt"}
    },
    {
        "id": "doc_003",
        "content": "Clarify the user's intent by restructuring their message using goal-oriented framing.",
        "metadata": {"content": "Clarify the user's intent by restructuring their message using goal-oriented framing.", "source": "intent_refiner.txt"}
    }
]

vectors = [
    (doc["id"], get_embedding(doc["content"]), doc["metadata"])
    for doc in documents
]

index.upsert(vectors=vectors)
print(f"✅ Upserted {len(vectors)} documents into Pinecone.")
