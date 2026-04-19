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

        prompt = f"""You are a financial regulation assistant.

Answer the question using ONLY the context below.

You MUST follow this exact format:

Direct answer: <short answer, max 10 words>
Explanation: <one clear sentence explaining the concept>
Insight: <one meaningful sentence explaining why it matters>
Source: <file name and page>

Important rules:
- Direct answer MUST contain the final answer and must not be empty
- Explanation MUST explain the concept, not where it appears
- Insight MUST explain practical importance in banking or regulation
- Do NOT mention pages, sources, documents, annexes, sections, or "context" in the Direct answer, Explanation, or Insight
- Do NOT repeat the same sentence across sections
- Keep answers clear, concise, and professional
- Use your own words rather than copying long sentences
- If possible, use the format "<concept>: <value>" in the Direct answer

If the answer is not in the context, output exactly:

Direct answer: I cannot find this in the provided documents.
Explanation: The retrieved text does not contain enough information.
Insight: A broader or different source may be needed.
Source: None

Context:
{context}

Question:
{query}

Answer:
"""
        return prompt

    def generate(self, query: str, retrieved_chunks: List[dict]) -> dict:
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
            },
            timeout=120
        )
        response.raise_for_status()

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