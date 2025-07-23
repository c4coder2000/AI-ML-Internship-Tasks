from typing import Dict
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os

# Initialize the OpenRouter-backed model
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model="openai/gpt-4o",
    temperature=0.0
)

def generate_refined_response(state: Dict) -> Dict:
    """
    LangGraph node that sends the refined prompt to LLM and stores the response.

    Args:
        state (dict): Contains 'refined_prompt'

    Returns:
        dict: Updated state with 'llm_response'
    """
    prompt = state.get("refined_prompt", "").strip()
    if not prompt:
        raise ValueError("Missing 'refined_prompt' in state.")

    messages = [
        SystemMessage(
            content=(
                "You are a professional prompt engineer. Refine user messages by applying clarity, goal-orientation, "
                "benefit-driven language, and emotional appeal. Focus on improving tone, specificity, and effectiveness."
            )
        ),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    state["llm_response"] = response.content
    return state
