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

        prompt = f"""You are a financial regulatory expert assistant.

You MUST follow this exact format. Do NOT skip sections.

📊 Direct Answer
<one sentence answer with the key number or definition>

📝 Explanation
<2-3 sentences explaining the concept clearly>

🧠 Insight
<1 meaningful insight about implications or real-world usage>

STRICT RULES:
- Use ONLY the information in the context below
- If the answer is not in the context, say: "I cannot find this in the provided documents."
- Be precise and concise
- Always mention the source and page number

Context:
{context}

Question:
{query}

Answer:
"""
        return prompt

    def _clean_answer(self, text: str) -> str:
        if not text:
            return ""
        bad_phrases = [
            "Unable to properly structure the explanation.",
            "This information is relevant for financial understanding.",
        ]
        for phrase in bad_phrases:
            text = text.replace(phrase, "")
        return text.strip()

    def _ensure_structure(self, text: str) -> str:
        if not text or len(text.strip()) < 20:
            return (
                "📊 Direct Answer\n"
                "I cannot find this in the provided documents.\n\n"
                "📝 Explanation\n"
                "The system could not extract a reliable explanation from the retrieved context.\n\n"
                "🧠 Insight\n"
                "This may indicate insufficient or weakly relevant retrieved information."
            )

        if "📊 Direct Answer" not in text:
            text = "📊 Direct Answer\n" + text
        if "📝 Explanation" not in text:
            text += "\n\n📝 Explanation\nExplanation not clearly generated."
        if "🧠 Insight" not in text:
            text += "\n\n🧠 Insight\nNo additional insight provided."

        return text.strip()

    def generate(self, query: str, retrieved_chunks: List[dict]) -> dict:
        prompt = self._build_prompt(query, retrieved_chunks)
        print("🤖 Generating answer with Mistral...")

        try:
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
            raw_answer = response.json().get("response", "").strip()
        except Exception as e:
            print(f"❌ Generation error: {e}")
            raw_answer = ""

        cleaned = self._clean_answer(raw_answer)
        final_answer = self._ensure_structure(cleaned)

        return {
            "query": query,
            "answer": final_answer,
            "sources": [
                {
                    "source": c["metadata"]["source"],
                    "page": c["metadata"]["page"],
                    "score": round(c["score"], 3)
                }
                for c in retrieved_chunks
            ]
        }