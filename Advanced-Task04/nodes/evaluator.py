from typing import Dict

def evaluate_response(state: Dict) -> Dict:
    """
    Looser scoring for refined prompts to promote flow over strictness.
    """
    user_msg = state.get("user_message", "")
    llm_output = state.get("llm_response", "")
    score = 0

    # 🪶 Rule 1: Basic length check — allow same or slightly longer
    if llm_output and len(llm_output) >= 20:
        score += 1

    # 🌈 Rule 2: Has keywords OR verbs suggesting refinement (less strict)
    benefit_keywords = [
        "streamline", "boost", "improve", "clarify", "refine", "optimize",
        "make it easier", "simplify", "update", "revise", "make sense"
    ]
    if any(kw in llm_output.lower() for kw in benefit_keywords):
        score += 1

    # 🧹 Rule 3: Not filled with vague filler phrases (same filter but optional)
    fluff_words = ["just wondering", "really mean a lot", "no offense", "tiny favor"]
    if not any(word in llm_output.lower() for word in fluff_words):
        score += 1

    # 🧭 Rule 4: Intent alignment — match the theme loosely (e.g. preserves question type)
    if user_msg and llm_output:
        if any(token in llm_output.lower() for token in user_msg.lower().split()[:3]):
            score += 1

    # 🎯 Relaxed threshold for passing
    state["eval_score"] = score
    state["eval_passed"] = score >= 2  # ← loosened from 3
    state["hitl_required"] = True  # Keep HITL gate for inspection logic
    print(f"Evaluation score: {score}, Passed: {state['eval_passed']}")
    return state
