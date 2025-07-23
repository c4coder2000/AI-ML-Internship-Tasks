# nodes/preprocessor.py
# so this node is pre processing the input for embedding and retrieval
# It cleans up the input, removes fluff, and prepares it for further processing 
import re
from typing import Dict

def preprocess_input(state: Dict) -> Dict:
    """
    Preprocesses the user's input for embedding + retrieval.

    Args:
        state (dict): Contains 'user_input' and 'metadata'

    Returns:
        dict: Updated state with preprocessed input
    """
    raw_input = state.get("user_message", "")

    # Basic cleanup
    cleaned = raw_input.strip().lower()

    # Optional: remove special characters or repeated whitespace
    cleaned = re.sub(r"[^\w\s.,!?-]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Optional: chunking (for multi-sentence inputs)
    chunks = [chunk.strip() for chunk in re.split(r"[.?!]", cleaned) if chunk.strip()]

    state.update({
        "preprocessed_input": cleaned,
        "chunks": chunks  # can be used for chunk-wise embedding
    })
    print("🧪 Cleaned:", cleaned)
    print("🧪 Chunks:", chunks)

    return state

