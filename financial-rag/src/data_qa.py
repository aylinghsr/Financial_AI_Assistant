# src/data_qa.py
import requests
import pandas as pd
from src.db import Database
from src.sql_generator import SQLGenerator


class DataQA:
    def __init__(
        self,
        db_path: str | None = None,
        model: str = "mistral",
        base_url: str = "http://localhost:11434"
    ):
        self.db = Database(db_path)
        self.sql_generator = SQLGenerator(model=model, base_url=base_url)
        self.model = model
        self.base_url = base_url

    def is_safe_sql(self, sql: str) -> bool:
        sql_clean = sql.strip().lower()
        if not sql_clean.startswith("select"):
            return False

        forbidden = [
            "insert", "update", "delete", "drop",
            "alter", "truncate", "create", "replace",
            "attach", "detach", "pragma"
        ]
        return not any(word in sql_clean for word in forbidden)

    def format_value(self, value):
        if pd.isna(value):
            return value

        if isinstance(value, (int, float)):
            abs_val = abs(value)

            if abs_val >= 1_000_000_000:
                return f"{value / 1_000_000_000:,.2f} billion"
            elif abs_val >= 1_000_000:
                return f"{value / 1_000_000:,.2f} million"
            elif abs_val >= 1_000:
                return f"{value:,.2f}"
            else:
                return f"{value:.2f}"

        return value

    def format_dataframe_for_prompt(self, df: pd.DataFrame) -> str:
        df_copy = df.copy()

        for col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(self.format_value)

        return df_copy.head(20).to_string(index=False)

    def explain_result(self, question: str, sql: str, df: pd.DataFrame) -> str:
        if df.empty:
            return """Summary: No matching rows were found.
Insight: The query executed successfully, but the dataset does not contain records matching the requested filter.
"""

        preview = self.format_dataframe_for_prompt(df)

        prompt = f"""You are a financial data assistant.

The user asked:
{question}

The SQL used was:
{sql}

The query result is:
{preview}

Write the answer in this format:

Summary: <1-2 short sentences with the direct answer>
Insight: <1 short sentence with the most useful takeaway>

Rules:
- Use ONLY the query result
- Do not invent any numbers or facts
- Respect the numeric formatting exactly as shown
- Be concise, structured, and professional
- If the result is a single value, state it directly
- If the result is a table, summarize the key pattern or comparison

Answer:
"""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 256
                }
            },
            timeout=60
        )
        response.raise_for_status()

        return response.json()["response"].strip()

    def execute_sql_with_retry(self, question: str, sql: str) -> tuple[str, pd.DataFrame | None, str | None]:
        schema = self.db.get_schema()

        try:
            df = self.db.run_query(sql)
            return sql, df, None
        except Exception as e:
            first_error = str(e)

        repaired_sql = self.sql_generator.repair_sql(
            question=question,
            schema=schema,
            bad_sql=sql,
            error_message=first_error
        )

        if not self.is_safe_sql(repaired_sql):
            return repaired_sql, None, "Generated repaired SQL was not safe to execute."

        try:
            df = self.db.run_query(repaired_sql)
            return repaired_sql, df, None
        except Exception as e:
            return repaired_sql, None, str(e)

    def ask(self, question: str) -> dict:
        schema = self.db.get_schema()
        sql = self.sql_generator.generate_sql(question, schema)

        if not self.is_safe_sql(sql):
            return {
                "question": question,
                "sql": sql,
                "rows": None,
                "answer": "Summary: Generated SQL was not safe to execute.\nInsight: The request was blocked for safety."
            }

        final_sql, df, error = self.execute_sql_with_retry(question, sql)

        if error is not None:
            return {
                "question": question,
                "sql": final_sql,
                "rows": None,
                "answer": f"Summary: SQL execution failed.\nInsight: {error}"
            }

        answer = self.explain_result(question, final_sql, df)

        return {
            "question": question,
            "sql": final_sql,
            "rows": df,
            "answer": answer
        }