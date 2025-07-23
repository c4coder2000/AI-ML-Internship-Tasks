# nodes/retriever.py

from typing import Dict
from tools.pinecone_tool import input_vector

def retrieve_documents(state: Dict) -> Dict:
    """
    LangGraph-compatible node that retrieves context chunks from Pinecone.

    Args:
        state (Dict): LangGraph state with 'chunk_embeddings' list,
                      where each item is a dict: { "text": str, "vector": List[float] }

    Returns:
        Dict: Updated state with retrieval trace and filtered context.
    """
    chunk_texts = [chunk["text"] for chunk in state.get("chunk_embeddings", [])]

    retrieval_results = []
    for chunk in chunk_texts:
        docs = input_vector(chunk)
        retrieval_results.append({
            "query_chunk": chunk,
            "retrieved_docs": docs
        })

    # ✅ Filter based on score (e.g. only include results above 0.6)
    flattened_context = [
        doc for result in retrieval_results
        for doc in result["retrieved_docs"]
        if doc["score"] >= 0.3
    ]

    state.update({
        "retrieval_results": retrieval_results,
        "retrieved_context": flattened_context
    })

    return state


