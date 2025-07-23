from typing import Dict

def build_prompt(state: Dict) -> Dict:
    """
    LangGraph node that constructs a refined prompt using the user's input and retrieved context.

    Args:
        state (Dict): Contains 'user_message' and 'retrieved_context'

    Returns:
        Dict: Updated state with 'refined_prompt' string
    """
    user_message = state.get("user_message", "").strip()
    retrieved_context = state.get("retrieved_context", [])

    # Format context lines
    context_lines = [
        f"{i+1}. {chunk['content']}"
        for i, chunk in enumerate(retrieved_context)
        if chunk.get("content")
    ]
    formatted_context = "\n".join(context_lines) if context_lines else "None"

    # ✅ Construct the refined prompt with explicit original message
    refined_prompt = f"""### Original Message:
{user_message}

### Task:
Make this message clearer and more persuasive

### Supporting Context:
{formatted_context}
"""

    state["refined_prompt"] = refined_prompt
    print("🧠 Final Refined Prompt:\n", refined_prompt)
    print("🧠 Type:", type(refined_prompt))

    return state


