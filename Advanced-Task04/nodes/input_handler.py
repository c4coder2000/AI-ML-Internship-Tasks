# nodes/input_handler.py

import uuid
from datetime import datetime
from langdetect import detect


def handle_user_input(input: dict) -> dict:
    """
    Extract and enrich user's prompt input with metadata.

    Args:
        input (dict): Incoming state dictionary (e.g. {"user_input": "Refine my prompt..."})

    Returns:
        dict: Updated state with input, enriched metadata
    """
    user_query = input.get("user_message", "").strip()
    print(f"Received user input: {user_query}")
    session_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Language detection (basic)
    try:
        language = detect(user_query) if user_query else "unknown"
    except Exception:
        language = "error"

    state = {
        "user_message": user_query,
        "metadata": {
            "session_id": session_id,
            "timestamp": timestamp,
            "source": input.get("source", "cli"),  # default to CLI
            "language": language,
        }
    }
    return state


