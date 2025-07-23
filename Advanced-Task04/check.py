# pinecone_check.py

import os
from pinecone import Pinecone
from tools.embedding_model import get_embedding

api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
index = pc.Index("prompt-refiner")

query_text = "who is hassnain and what does he do"
vector = get_embedding(query_text)

response = index.query(vector=vector, top_k=3, include_metadata=True, namespace="corpus_chunks")

print(f"\n🔍 Results for query: \"{query_text}\"")
if response["matches"]:
    for i, match in enumerate(response["matches"]):
        print(f"\n📄 Result {i+1}")
        print(f"Content   : {match['metadata'].get('content', '[no content]')}")
        print(f"Source    : {match['metadata'].get('source', 'N/A')}")
        print(f"Score     : {match['score']:.4f}")
else:
    print("🚫 No matches found. Try a closer query or confirm the corpus is loaded.")
