# langgraph.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Optional

# 🧩 Importing node functions
from nodes.input_handler import handle_user_input
from nodes.preprocessor import preprocess_input
from nodes.embedder import embed_input
from nodes.retriever import retrieve_documents
from nodes.prompt_builder import build_prompt
from nodes.llm_caller import generate_refined_response
from nodes.hitl_editor import hitl_editor
from nodes.evaluator import evaluate_response
from nodes.memory_sync import sync_memory

# 🧠 Define shared state schema
from typing import TypedDict, List, Dict, Optional

class WorkflowState(TypedDict, total=False):
    # 📝 User Input + Metadata
    user_message: str
    metadata: Dict[str, str]  # session_id, timestamp, language, source

    # 🧼 Preprocessing
    preprocessed_input: str
    chunks: List[str]  # sentence-level splits

    # 🧠 Embedding
    full_embedding: List[float]
    chunk_embeddings: List[Dict[str, List[float]]]  # [{text, vector}]

    # 🔍 Retrieval
    retrieved_context: List[Dict[str, str]]         # from Pinecone
    retrieval_results: List[Dict[str, List[Dict]]]  # trace per chunk

    # ✏️ Prompt Refinement
    refined_prompt: str

    # 🤖 LLM Response
    llm_response: str

    # 📊 Evaluation
    eval_score: int
    eval_passed: bool

    # 👤 Human-in-the-loop
    hitl_required: bool
    hitl_status: str

    # ✅ Memory sync
    memory_synced: bool


# 🚀 Build the LangGraph
graph = StateGraph(WorkflowState)

# 🧩 Register all core nodes
graph.add_node("input_handler", handle_user_input)
graph.add_node("preprocessor", preprocess_input)
graph.add_node("embedder", embed_input)
graph.add_node("retriever", retrieve_documents)
graph.add_node("prompt_builder", build_prompt)
graph.add_node("llm_caller", generate_refined_response)
graph.add_node("hitl_editor", hitl_editor)
graph.add_node("evaluator", evaluate_response)
graph.add_node("memory_sync", sync_memory)

# 🧠 Entry point
graph.set_entry_point("input_handler")

# 🔁 Sequential edges up to HITL
graph.add_edge("input_handler", "preprocessor")
graph.add_edge("preprocessor", "embedder")
graph.add_edge("embedder", "retriever")
graph.add_edge("retriever", "prompt_builder")
graph.add_edge("prompt_builder", "llm_caller")
graph.add_edge("llm_caller", "hitl_editor")

# 🔀 Conditional edge after HITL
def hitl_routing(state: WorkflowState) -> str:
    status = state.get("hitl_status", "approved")
    if status in ["approved", "edited"]:
        return "evaluator"
    elif status == "regenerate":
        return "llm_caller"
    elif status == "excluded":
        return "end"  
    return "evaluator"


graph.add_conditional_edges("hitl_editor", hitl_routing, {
    "evaluator": "evaluator",
    "llm_caller": "llm_caller",
    "end": END  
})


def eval_routing(state: WorkflowState) -> str:
    return "memory_sync" if state.get("eval_passed", False) else "end"


graph.add_conditional_edges("evaluator", eval_routing, {
    "memory_sync": "memory_sync",
    "end": END
})

# ✅ Final edge
graph.add_edge("memory_sync", END)

# 🧠 Compile the app
app = graph.compile()

# 🧪 Optional test block
if __name__ == "__main__":

    input_text = input("📝 Enter your prompt to refine: ")
    result = app.invoke({"user_message": input_text})


    print("\n🧠 Final Output:")
    print(result.get("llm_response", "No response generated."))


def run_chat_turn(user_input: str, history: Optional[List[Dict]] = None) -> str:
    """Wraps LangGraph execution for Streamlit or other interfaces."""
    
    # Construct the initial state payload
    state: WorkflowState = {
        "user_message": user_input,
        "metadata": {
            "source": "streamlit",
            "session_id": "default",  # can be made dynamic per user
        }
    }

    # Optionally append recent history if using memory chaining
    # You can modify this logic if you embed past turns yourself
    if history:
        state["metadata"]["history"] = str(history)

    # Invoke LangGraph and return the LLM response
    result = app.invoke(state)
    return result.get("llm_response", "No response generated.")
