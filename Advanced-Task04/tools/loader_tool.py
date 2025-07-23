# tools/loader.py

import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ✅ Pinecone setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("prompt-refiner")

# ✅ Embedder setup (ensure dimension matches Pinecone index)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_and_store(path: str, namespace="corpus_chunks"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"📁 Found {len(lines)} lines in file.")
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        return

    # ✅ Join lines into full text
    text = "\n".join([line.strip() for line in lines if line.strip()])
    if not text:
        print("⚠️ All lines are blank or unprintable.")
        return

    print("📃 Text Preview:\n", text[:300])
    
    # 🔩 Proceed with chunking + embedding as before
    chunks = [text[i:i+500].strip() for i in range(0, len(text), 500)]
    entries = []
    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        entries.append((f"{namespace}_{i}", embedding, {
            "type": "rag_chunk",
            "source": path,
            "chunk_id": str(i),
            "content": chunk
        }))

    if not entries:
        print("❌ No valid chunks to upsert.")
        return

    index.upsert(vectors=entries, namespace=namespace)
    print(f"✅ Upserted {len(entries)} chunks.")



    

