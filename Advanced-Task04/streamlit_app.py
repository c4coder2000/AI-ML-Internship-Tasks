import streamlit as st
from langgraph_app import run_chat_turn  # your LangGraph chain

st.title("🧠 Context-Aware Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me anything...")
if user_input:
    st.session_state.chat_history.append({"user": user_input})
    response = run_chat_turn(user_input, st.session_state.chat_history)
    st.session_state.chat_history.append({"bot": response})

for msg in st.session_state.chat_history:
    role = list(msg.keys())[0]
    st.chat_message(role).markdown(msg[role])
