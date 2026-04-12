# src/router.py
import requests


class QueryRouter:
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def _build_prompt(self, query: str) -> str:
        return f"""You are a classifier.

Decide whether the user's question should be answered using:
- DATA = structured database queries, aggregations, counts, averages, trends, filtering over internal data
- DOCUMENT = explanations, definitions, regulatory concepts, policy text, Basel rules, capital ratios, liquidity ratios, leverage ratio definitions

Important:
- Questions about Basel III, CET1, leverage ratio, liquidity coverage ratio, capital requirements, or definitions of regulatory terms should be DOCUMENT
- Even if the question mentions a number, ratio, threshold, minimum, or requirement, choose DOCUMENT if it is asking about a regulatory rule or concept
- Choose DATA only if the answer must come from the structured database

Examples:
Question: What is CET1?
Answer: DOCUMENT

Question: What is the minimum Common Equity Tier 1 capital ratio?
Answer: DOCUMENT

Question: Explain the Liquidity Coverage Ratio
Answer: DOCUMENT

Question: How many transactions are there for Solaris SE?
Answer: DATA

Question: What is the average amount for gl_number 4100?
Answer: DATA

Question: Which entity has the higher total balance amount?
Answer: DATA

Answer ONLY one word:
DATA or DOCUMENT

Question:
{query}

Answer:
"""

    def route(self, query: str) -> str:
        prompt = self._build_prompt(query)

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 5
                }
            },
            timeout=30
        )
        response.raise_for_status()

        decision = response.json()["response"].strip().upper()

        if "DATA" in decision:
            return "DATA"
        return "DOCUMENT"