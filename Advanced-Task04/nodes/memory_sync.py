import os
from datetime import datetime
from typing import Dict

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ✅ Pinecone & embedder setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("prompt-refiner")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def sync_memory(state: Dict) -> Dict:
    """
    Upserts a semantic memory snapshot to Pinecone within 'corpus_chunks' namespace.
    Uses metadata structure aligned with rag_chunk format but labeled as session_memory.
    """
    user_msg = state.get("user_message", "").strip()
    refined_prompt = state.get("refined_prompt", "").strip()
    llm_output = state.get("llm_response", "").strip()

    if not llm_output:
        raise ValueError("LLM response missing — nothing to sync.")

    # 🧠 Embed the LLM response
    embedding = embedder.encode(llm_output).tolist()

    # 🔐 Unique ID with timestamp
    timestamp = datetime.now().isoformat()
    memory_id = f"session_memory_{timestamp}_{abs(hash(user_msg))}"

    # 📦 Metadata aligned with corpus chunks, but typed for memory
    metadata = {
        "type": "session_memory",              # Distinct from rag_chunk
        "source": "chatbot",
        "chunk_id": f"memory_{timestamp}",
        "content": llm_output,
        "user_message": user_msg,
        "refined_prompt": refined_prompt,
        "timestamp": timestamp
    }

    # 💾 Store in shared namespace
    index.upsert([(memory_id, embedding, metadata)], namespace="corpus_chunks")

    state["memory_synced"] = True
    return state
