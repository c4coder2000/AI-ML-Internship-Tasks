from typing import Dict, List
from tools.embedding_model import get_embedding

def embed_input(state: Dict) -> Dict:
    """
    Embed preprocessed user input and its chunks using HF model.

    Args:
        state (dict): Contains 'preprocessed_input', 'chunks'

    Returns:
        dict: Updated state with embeddings
    """
    print("🧪 Received preprocessed_input:", state.get("preprocessed_input"))

    input_text = state.get("preprocessed_input", "")
    if not input_text:
        raise ValueError("Missing preprocessed input for embedding.")

    # ✅ Get embedding directly (not a model object)
    full_embedding = get_embedding(input_text, verbose=True)

    # ✅ Embed chunks individually
    chunk_embeddings = []
    for chunk in state.get("chunks", []):
        vec = get_embedding(chunk)
        chunk_embeddings.append({"text": chunk, "vector": vec})

    # 🔧 Update state
    state.update({
        "full_embedding": full_embedding,
        "chunk_embeddings": chunk_embeddings
    })

    return state
