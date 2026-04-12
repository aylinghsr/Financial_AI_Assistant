# evaluate.py
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from src.loader import load_all_documents
from src.chunker import split_into_chunks
from src.embedder import Embedder
from src.indexer import Indexer
from src.retriever import HybridRetriever
from src.generator import Generator
from src.data_qa import DataQA
from src.router import QueryRouter


def load_eval_questions(filepath: str = "eval/eval_questions.json") -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def keyword_match_score(answer: str, expected_keywords: List[str]) -> Dict[str, Any]:
    answer_lower = answer.lower()
    found = []
    missing = []

    for kw in expected_keywords:
        if kw.lower() in answer_lower:
            found.append(kw)
        else:
            missing.append(kw)

    score = len(found) / len(expected_keywords) if expected_keywords else 0.0

    return {
        "score": score,
        "found": found,
        "missing": missing,
        "passed": score >= 0.6
    }


def check_expected_columns(df: pd.DataFrame | None, expected_columns: List[str] | None) -> Dict[str, Any]:
    if expected_columns is None:
        return {"passed": True, "missing": []}
    if df is None:
        return {"passed": False, "missing": expected_columns}

    actual = set(df.columns)
    missing = [c for c in expected_columns if c not in actual]
    return {
        "passed": len(missing) == 0,
        "missing": missing
    }


def check_expected_rows(df: pd.DataFrame | None, expected_rows_min: int | None) -> Dict[str, Any]:
    if expected_rows_min is None:
        return {"passed": True, "actual_rows": None}
    if df is None:
        return {"passed": False, "actual_rows": 0}

    return {
        "passed": len(df) >= expected_rows_min,
        "actual_rows": len(df)
    }


def check_expected_empty(df: pd.DataFrame | None, expected_empty: bool | None) -> Dict[str, Any]:
    if expected_empty is None:
        return {"passed": True}
    if df is None:
        return {"passed": True}
    return {"passed": df.empty}


def check_expected_exact(df: pd.DataFrame | None, expected_exact: Dict[str, Any] | None) -> Dict[str, Any]:
    if expected_exact is None:
        return {"passed": True, "details": {}}
    if df is None or df.empty:
        return {"passed": False, "details": {"reason": "No dataframe returned"}}

    row = df.iloc[0]
    details = {}
    passed = True

    for col, expected_value in expected_exact.items():
        actual_value = row.get(col, None)
        ok = actual_value == expected_value
        details[col] = {
            "expected": expected_value,
            "actual": actual_value,
            "passed": ok
        }
        if not ok:
            passed = False

    return {"passed": passed, "details": details}


def check_expected_approx(df: pd.DataFrame | None, expected_approx: Dict[str, Any] | None) -> Dict[str, Any]:
    if expected_approx is None:
        return {"passed": True, "details": {}}
    if df is None or df.empty:
        return {"passed": False, "details": {"reason": "No dataframe returned"}}

    tolerance_pct = expected_approx.get("tolerance_pct", 0.05)
    details = {}
    passed = True

    for col, expected_value in expected_approx.items():
        if col == "tolerance_pct":
            continue

        actual_value = float(df.iloc[0][col])
        lower = expected_value * (1 - tolerance_pct)
        upper = expected_value * (1 + tolerance_pct)
        ok = lower <= actual_value <= upper

        details[col] = {
            "expected": expected_value,
            "actual": actual_value,
            "lower": lower,
            "upper": upper,
            "passed": ok
        }

        if not ok:
            passed = False

    return {"passed": passed, "details": details}


def check_expected_source(sources: List[Dict[str, Any]], expected_source: str | None) -> Dict[str, Any]:
    if expected_source is None:
        return {"passed": True}

    source_names = [s["source"] for s in sources]
    passed = expected_source in source_names
    return {
        "passed": passed,
        "retrieved_sources": source_names
    }


def print_divider():
    print("\n" + "=" * 100)


def evaluate_router(router: QueryRouter, question: str, expected_type: str) -> Dict[str, Any]:
    predicted = router.route(question)
    passed = predicted == expected_type
    return {
        "expected": expected_type,
        "predicted": predicted,
        "passed": passed
    }


def evaluate_data_question(data_qa: DataQA, item: Dict[str, Any]) -> Dict[str, Any]:
    result = data_qa.ask(item["question"])
    answer = result["answer"]
    sql = result.get("sql", "")
    rows = result.get("rows", None)

    keyword_eval = keyword_match_score(answer, item.get("expected_keywords", []))
    sql_ok = sql.strip().lower().startswith("select") if sql else False
    rows_ok = rows is not None

    column_eval = check_expected_columns(rows, item.get("expected_columns"))
    row_eval = check_expected_rows(rows, item.get("expected_rows_min"))
    empty_eval = check_expected_empty(rows, item.get("expected_empty"))
    exact_eval = check_expected_exact(rows, item.get("expected_exact"))
    approx_eval = check_expected_approx(rows, item.get("expected_approx"))

    passed = all([
        sql_ok,
        keyword_eval["passed"],
        column_eval["passed"],
        row_eval["passed"],
        empty_eval["passed"],
        exact_eval["passed"],
        approx_eval["passed"]
    ])

    return {
        "answer": answer,
        "sql": sql,
        "rows_ok": rows_ok,
        "sql_ok": sql_ok,
        "keyword_eval": keyword_eval,
        "column_eval": column_eval,
        "row_eval": row_eval,
        "empty_eval": empty_eval,
        "exact_eval": exact_eval,
        "approx_eval": approx_eval,
        "passed": passed,
        "rows": rows
    }


def evaluate_document_question(
    retriever: HybridRetriever,
    generator: Generator,
    item: Dict[str, Any]
) -> Dict[str, Any]:
    results = retriever.retrieve(item["question"], top_k=5)
    response = generator.generate(item["question"], results)
    answer = response["answer"]
    sources = response.get("sources", [])

    keyword_eval = keyword_match_score(answer, item.get("expected_keywords", []))
    source_ok = len(sources) > 0
    source_eval = check_expected_source(sources, item.get("expected_source"))

    passed = all([
        source_ok,
        keyword_eval["passed"],
        source_eval["passed"]
    ])

    return {
        "answer": answer,
        "sources": sources,
        "source_ok": source_ok,
        "source_eval": source_eval,
        "keyword_eval": keyword_eval,
        "passed": passed
    }


def save_results(results_summary: List[Dict[str, Any]], filepath: str = "eval/results.json"):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)


def main():
    print("Loading evaluation questions...")
    eval_items = load_eval_questions()

    print("Loading document pipeline...")
    docs = load_all_documents("documents")
    chunks = split_into_chunks(docs)
    embedder = Embedder()
    indexer = Indexer()
    retriever = HybridRetriever(chunks, embedder, indexer)
    generator = Generator()

    print("Loading data pipeline...")
    data_qa = DataQA()

    print("Loading router...")
    router = QueryRouter()

    total = len(eval_items)
    total_passed = 0
    router_passed = 0
    results_summary = []

    start_time = time.time()

    for i, item in enumerate(eval_items, start=1):
        print_divider()
        print(f"TEST {i}/{total}")
        print(f"ID          : {item['id']}")
        print(f"TYPE        : {item['type']}")
        print(f"QUESTION    : {item['question']}")
        print(f"DESCRIPTION : {item['description']}")

        router_eval = evaluate_router(router, item["question"], item["type"])
        if router_eval["passed"]:
            router_passed += 1

        print(f"\n[Router]")
        print(f"Expected route : {router_eval['expected']}")
        print(f"Predicted route: {router_eval['predicted']}")
        print(f"Router passed  : {router_eval['passed']}")

        if item["type"] == "DATA":
            eval_result = evaluate_data_question(data_qa, item)

            print(f"\n[Data Answer]")
            print(eval_result["answer"])

            print(f"\n[Generated SQL]")
            print(eval_result["sql"])

            print(f"\n[Checks]")
            print(f"SQL ok         : {eval_result['sql_ok']}")
            print(f"Rows returned  : {eval_result['rows_ok']}")
            print(f"Keyword score  : {eval_result['keyword_eval']['score']:.2f}")
            print(f"Columns passed : {eval_result['column_eval']['passed']} | missing={eval_result['column_eval']['missing']}")
            print(f"Rows passed    : {eval_result['row_eval']['passed']} | actual_rows={eval_result['row_eval']['actual_rows']}")
            print(f"Empty passed   : {eval_result['empty_eval']['passed']}")
            print(f"Exact passed   : {eval_result['exact_eval']['passed']} | details={eval_result['exact_eval']['details']}")
            print(f"Approx passed  : {eval_result['approx_eval']['passed']} | details={eval_result['approx_eval']['details']}")
            print(f"Test passed    : {eval_result['passed']}")

        else:
            eval_result = evaluate_document_question(retriever, generator, item)

            print(f"\n[Document Answer]")
            print(eval_result["answer"])

            print(f"\n[Sources]")
            for s in eval_result["sources"]:
                print(f"- {s['source']} page {s['page']} (score: {s['score']})")

            print(f"\n[Checks]")
            print(f"Source ok      : {eval_result['source_ok']}")
            print(f"Source passed  : {eval_result['source_eval']['passed']}")
            print(f"Keyword score  : {eval_result['keyword_eval']['score']:.2f}")
            print(f"Found          : {eval_result['keyword_eval']['found']}")
            print(f"Missing        : {eval_result['keyword_eval']['missing']}")
            print(f"Test passed    : {eval_result['passed']}")

        if eval_result["passed"]:
            total_passed += 1

        results_summary.append({
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "router_passed": router_eval["passed"],
            "pipeline_passed": eval_result["passed"]
        })

    elapsed = time.time() - start_time

    save_results(results_summary)

    print_divider()
    print("FINAL SUMMARY")
    print_divider()
    print(f"Total tests           : {total}")
    print(f"Pipeline tests passed : {total_passed}/{total} ({total_passed / total:.1%})")
    print(f"Router tests passed   : {router_passed}/{total} ({router_passed / total:.1%})")
    print(f"Elapsed time          : {elapsed:.2f} seconds")
    print("Results saved to      : eval/results.json")

    print("\nPer-test summary:")
    for row in results_summary:
        print(
            f"- {row['id']} | {row['type']} | "
            f"router={'PASS' if row['router_passed'] else 'FAIL'} | "
            f"pipeline={'PASS' if row['pipeline_passed'] else 'FAIL'}"
        )


if __name__ == "__main__":
    main()