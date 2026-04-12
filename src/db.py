# src/db.py
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


class Database:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent
            self.db_path = project_root / "data" / "gl_database.db"
        else:
            self.db_path = Path(db_path)

    def run_query(self, sql: str) -> pd.DataFrame:
        conn = sqlite3.connect(str(self.db_path))
        try:
            df = pd.read_sql_query(sql, conn)
            return df
        finally:
            conn.close()

    def get_schema(self) -> Dict[str, List[dict]]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            schema = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                schema[table] = [
                    {
                        "name": col[1],
                        "type": col[2]
                    }
                    for col in columns
                ]

            return schema
        finally:
            conn.close()