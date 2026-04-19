# src/sql_generator.py
import requests
from typing import Dict, List


class SQLGenerator:
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def _build_prompt(self, question: str, schema: Dict[str, List[dict]]) -> str:
        schema_text = []
        for table, cols in schema.items():
            col_text = ", ".join([f"{c['name']} ({c['type']})" for c in cols])
            schema_text.append(f"{table}: {col_text}")

        schema_block = "\n".join(schema_text)

        examples = """
Example 1
Question: What is the total amount by entity in gl_balances?
SQL:
SELECT entity, SUM(amount) AS total_amount
FROM gl_balances
GROUP BY entity
ORDER BY total_amount DESC;

Example 2
Question: Show monthly balances for gl_number 1300
SQL:
SELECT period, entity, amount
FROM gl_balances
WHERE gl_number = '1300'
ORDER BY period, entity;

Example 3
Question: How many transactions are there for Solaris SE?
SQL:
SELECT COUNT(*) AS transaction_count
FROM gl_transactions
WHERE entity = 'Solaris SE';

Example 4
Question: List all income accounts
SQL:
SELECT gl_number, account_name, account_type
FROM gl_accounts
WHERE account_type = 'Income'
ORDER BY gl_number;

Example 5
Question: Show the top 5 GL accounts by average amount in gl_balances
SQL:
SELECT gl_number, AVG(amount) AS avg_amount
FROM gl_balances
GROUP BY gl_number
ORDER BY avg_amount DESC
LIMIT 5;

Example 6
Question: Which entity has the higher total balance amount?
SQL:
SELECT entity, SUM(amount) AS total_amount
FROM gl_balances
GROUP BY entity
ORDER BY total_amount DESC
LIMIT 1;
"""

        prompt = f"""You are an expert SQLite SQL assistant.

Write one valid SQLite SELECT query for the user's question.

Rules:
- Use ONLY the tables and columns listed below
- Return ONLY SQL
- Do not add explanations
- Use SQLite syntax
- Only generate SELECT statements
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE
- Prefer the simplest correct query
- Do not use nested queries unless necessary
- If only one table is needed, use only one table
- Fully qualify column names when joining tables if a column exists in both tables
- If the question asks for totals, use SUM(...)
- If the question asks for averages, use AVG(...)
- If the question asks for counts, use COUNT(*)
- If useful, join tables using gl_number
- Prefer clear aliases

Database schema:
{schema_block}

Examples:
{examples}

User question:
{question}

SQL:
"""
        return prompt

    def generate_sql(self, question: str, schema: Dict[str, List[dict]]) -> str:
        prompt = self._build_prompt(question, schema)

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 256
                }
            },
            timeout=60
        )
        response.raise_for_status()

        sql = response.json()["response"].strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()

        if "SELECT" in sql.upper():
            select_pos = sql.upper().find("SELECT")
            sql = sql[select_pos:]

        return sql

    def repair_sql(
        self,
        question: str,
        schema: Dict[str, List[dict]],
        bad_sql: str,
        error_message: str
    ) -> str:
        schema_text = []
        for table, cols in schema.items():
            col_text = ", ".join([f"{c['name']} ({c['type']})" for c in cols])
            schema_text.append(f"{table}: {col_text}")

        schema_block = "\n".join(schema_text)

        prompt = f"""You are an expert SQLite SQL assistant.

The previous SQL failed. Fix it.

Rules:
- Return ONLY corrected SQL
- Use SQLite syntax
- Only generate a SELECT query
- Use ONLY the schema below
- Keep the query as simple as possible
- If columns are ambiguous, fully qualify them
- If the previous query used unnecessary nesting, simplify it

Database schema:
{schema_block}

User question:
{question}

Bad SQL:
{bad_sql}

SQLite error:
{error_message}

Corrected SQL:
"""

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 256
                }
            },
            timeout=60
        )
        response.raise_for_status()

        sql = response.json()["response"].strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()

        if "SELECT" in sql.upper():
            select_pos = sql.upper().find("SELECT")
            sql = sql[select_pos:]

        return sql