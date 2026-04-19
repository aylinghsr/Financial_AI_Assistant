# evaluation/generate_training_data.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import requests
import time
from src.loader import load_all_documents
from src.chunker import split_into_chunks


def generate_question_for_chunk(chunk_text: str, model: str = "mistral") -> str:
    """Ask Mistral to generate one question answered by this chunk."""

    prompt = f"""You are creating a financial Q&A dataset.

Read the following text from a financial regulatory document and write ONE clear question that this text directly answers.

Rules:
- Write only the question, nothing else
- No preamble like "Here is a question:"
- The question must be answerable from the text
- Use financial/regulatory language naturally

Text:
{chunk_text[:600]}

Question:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 80
            }
        },
        timeout=60
    )

    return response.json()["response"].strip()


def generate_training_data(max_chunks: int = 60):
    """Generate synthetic (question, chunk) training pairs."""

    print("📄 Loading documents and chunks...")
    docs   = load_all_documents("documents")
    chunks = split_into_chunks(docs)

    # use a subset — enough to fine-tune without taking hours
    selected = chunks[:max_chunks]
    print(f"✅ Using {len(selected)} chunks for training data generation\n")

    training_pairs = []
    failed         = 0

    for i, chunk in enumerate(selected):
        print(f"  [{i+1}/{len(selected)}] Generating question...")

        try:
            question = generate_question_for_chunk(chunk.content)

            # basic quality filter
            if len(question) > 15 and "?" in question:
                training_pairs.append({
                    "question": question,
                    "positive": chunk.content,
                    "source":   chunk.metadata["source"],
                    "page":     chunk.metadata["page"]
                })
            else:
                print(f"    ⚠️  Low quality question, skipping")
                failed += 1

        except Exception as e:
            print(f"    ❌ Error: {e}")
            failed += 1

        # small delay to avoid overwhelming Ollama
        time.sleep(0.5)

    # save
    os.makedirs("evaluation/results", exist_ok=True)
    output_path = "evaluation/results/training_data.json"

    with open(output_path, "w") as f:
        json.dump(training_pairs, f, indent=2)

    print(f"\n✅ Generated {len(training_pairs)} training pairs")
    print(f"❌ Failed/skipped: {failed}")
    print(f"💾 Saved to {output_path}")

    # preview
    print("\n--- Sample training pair ---")
    if training_pairs:
        sample = training_pairs[0]
        print(f"Question: {sample['question']}")
        print(f"Positive: {sample['positive'][:200]}...")

    return training_pairs


if __name__ == "__main__":
    generate_training_data(max_chunks=60)