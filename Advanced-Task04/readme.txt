# 🧠 Prompt Refiner — Modular LangGraph Workflow

Built with precision and curiosity, **Prompt Refiner** is a smart LangGraph-powered orchestration that refines user prompts using LLMs via OpenRouter API, augments responses through RAG, and stores semantic memory in Pinecone. It includes evaluation logic, human-in-the-loop edits, and a custom text-based knowledge loader.

---

## 📦 Project Overview

This repo showcases:

- 🔄 A full LangGraph pipeline with custom nodes.
- ✨ Prompt refinement using LLM completions.
- 🔍 Context retrieval via Pinecone embeddings.
- 🧑‍💻 Human-in-the-loop editing and response vetting.
- 📊 Evaluation logic with pass/fail routing.
- 🧠 Memory sync for refined prompts.
- 📁 RAG-enabled augmentation using `hassnain_info.txt`.

---

## 🗺️ Workflow Summary

```text
User Input → Preprocess → Embed → Retrieve → Build Prompt → LLM Refinement
→ Human Editor → Evaluation → Memory Sync / End
