# evaluation/evaluate.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.loader import load_all_documents
from src.chunker import split_into_chunks
from src.embedder import Embedder
from src.indexer import Indexer
from src.retriever import HybridRetriever
from src.generator import Generator
from evaluation.test_questions import TEST_QUESTIONS


def run_evaluation():
    print("🔧 Setting up pipeline...")
    docs     = load_all_documents("documents")
    chunks   = split_into_chunks(docs)
    embedder = Embedder()
    indexer  = Indexer()
    retriever = HybridRetriever(chunks, embedder, indexer)
    generator = Generator()

    # use only first 4 questions to keep runtime manageable
    test_set = TEST_QUESTIONS[:4]
    print(f"📝 Running {len(test_set)} test questions...\n")

    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []

    for i, item in enumerate(test_set):
        question     = item["question"]
        ground_truth = item["ground_truth"]
        print(f"  [{i+1}/{len(test_set)}] {question[:60]}...")

        results  = retriever.retrieve(question, top_k=3)
        response = generator.generate(question, results)

        questions.append(question)
        answers.append(response["answer"])
        contexts.append([r["content"] for r in results])
        ground_truths.append(ground_truth)

    print("\n✅ All questions answered. Running RAGAs evaluation...\n")

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths
    })

    llm = LangchainLLMWrapper(
        OllamaLLM(model="mistral", timeout=120)
    )
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=llm,
        embeddings=emb,
        raise_exceptions=False,
        batch_size=1        # ← run one at a time, no parallel overwhelm
    )

    df = result.to_pandas()

    os.makedirs("evaluation/results", exist_ok=True)
    df.to_csv("evaluation/results/ragas_results.csv", index=False)
    print("💾 Results saved to evaluation/results/ragas_results.csv")

    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)

    metrics = {
        "Faithfulness":      "faithfulness",
        "Answer Relevancy":  "answer_relevancy",
        "Context Precision": "context_precision",
        "Context Recall":    "context_recall",
    }

    scores = {}
    for label, col in metrics.items():
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 0:
                score = valid.mean()
                scores[label] = score
                bar = "█" * int(score * 20)
                print(f"{label:<22} {score:.3f}  {bar}")
            else:
                print(f"{label:<22} N/A   (all timeouts)")
        else:
            print(f"{label:<22} N/A   (not computed)")

    if scores:
        avg = sum(scores.values()) / len(scores)
        print("=" * 70)
        print(f"Overall Average: {avg:.3f}")

    # also print per-question breakdown
    print("\n📋 Per-question scores:")
    print(df.to_string(index=False))

    return scores


if __name__ == "__main__":
    run_evaluation()