# Financial AI Assistant

A production-grade RAG (Retrieval-Augmented Generation) system for querying financial regulatory documents and GL accounting data using natural language.

Built entirely with local, open-source models — no paid APIs required.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Mistral](https://img.shields.io/badge/LLM-Mistral%207B-orange)
![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red)

---

## What it does

**Document Q&A** — Ask questions about financial regulatory documents (Basel III, BaFin circulars) and get precise answers with page citations.

**GL Data Q&A** — Ask natural language questions about General Ledger data and get SQL-backed answers with visualisations.

**Automatic routing** — The system detects whether your question is about documents or data and routes it to the right pipeline automatically.

---

## Architecture