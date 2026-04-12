# src/generator.py
import requests
from typing import List

class Generator:
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def _build_prompt(self, query: str, chunks: List[dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk["metadata"]["source"]
            page = chunk["metadata"]["page"]
            context_parts.append(
                f"[Source {i+1} | {source} | Page {page}]\n{chunk['content']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a careful financial regulation assistant.

Answer the question using ONLY the context below.

Rules:
- Use only the provided context
- If the answer is not in the context, say: "I cannot find this in the provided documents."
- For definition questions, answer in 1-3 precise sentences
- Include the regulatory concept clearly and directly
- End with the source file and page numbers

Context:
{context}

Question:
{query}

Answer:
"""
        return prompt

    def generate(self, query: str, retrieved_chunks: List[dict]) -> dict:
        """Generate an answer given a query and retrieved context chunks."""

        prompt = self._build_prompt(query, retrieved_chunks)

        print("🤖 Generating answer with Mistral...")

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 512
                }
            }
        )

        answer = response.json()["response"].strip()

        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "source": c["metadata"]["source"],
                    "page": c["metadata"]["page"],
                    "score": round(c["score"], 3)
                }
                for c in retrieved_chunks
            ]
        }