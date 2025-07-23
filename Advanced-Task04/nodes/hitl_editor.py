# nodes/hitl_editor.py

from typing import Dict

def hitl_editor(state: Dict) -> Dict:
    """
    Human-in-the-loop validator and editor for LLM output.
    Offers approval, manual edit, regeneration flag, or exclusion.
    """
    user_msg = state.get("user_message", "")
    refined_prompt = state.get("refined_prompt", "")
    llm_output = state.get("llm_response", "")

    print("\n🧠 Original User Message:\n")
    print(user_msg)

    print("\n🔧 Refined Prompt:\n")
    print(refined_prompt)

    print("\n📤 LLM Output:\n")
    print(llm_output)

    print("\n✅ HITL Options:")
    print("1 → Approve")
    print("2 → Edit Manually")
    print("3 → Regenerate (loop to llm_caller)")
    print("4 → Exclude from memory")

    choice = input("👉 Your choice: ").strip()

    if choice == "1":
        state["hitl_status"] = "approved"

    elif choice == "2":
        print("\n✏️ Enter your revised version below:")
        revised = input("📝 Revised Output: ").strip()
        if revised:
            state["llm_response"] = revised
            state["hitl_status"] = "edited"
        else:
            print("⚠️ No revision provided. Keeping original output.")
            state["hitl_status"] = "approved"

    elif choice == "3":
        state["hitl_status"] = "regenerate"

    elif choice == "4":
        state["hitl_status"] = "excluded"

    else:
        print("⚠️ Invalid choice. Defaulting to 'approved'.")
        state["hitl_status"] = "approved"

    return state


