# 📊 Financial AI Assistant

A hybrid AI system that answers questions over:

- 📄 Financial regulatory documents (Basel III)
- 🗄️ Structured financial data (GL database)

## 🚀 Features

- Hybrid routing (Document RAG vs SQL)
- Vector search with Qdrant
- BM25 + embedding retrieval
- Text-to-SQL pipeline
- Automatic evaluation framework
- Streamlit UI with charts and insights

## 🧠 Architecture

User Query  
→ Router (DATA vs DOCUMENT)  
→  
- DOCUMENT → Retriever → Generator (RAG)  
- DATA → SQL Generator → SQLite → Answer  

## 📊 Example Questions

### Document
- What is the minimum CET1 capital ratio?
- What is the Liquidity Coverage Ratio?

### Data
- What is the total amount by entity?
- How many transactions are there for Solaris SE?
- Show monthly balances for gl_number 1300

## 📈 Evaluation

- Pipeline accuracy: 100% (9/9 basic tests)
- Extended evaluation: ~91% (11-test suite)
- Includes:
  - routing accuracy
  - SQL validation
  - exact numeric checks
  - retrieval validation
  - edge cases

## 🛠 Tech Stack

- Python
- Streamlit
- Qdrant
- SentenceTransformers
- SQLite
- Ollama (Mistral)

## ▶️ Run locally

```bash
pip install -r requirements.txt
streamlit run app.py