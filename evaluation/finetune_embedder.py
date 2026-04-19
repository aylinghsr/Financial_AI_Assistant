# evaluation/finetune_embedder.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import torch
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader


def load_training_data(path: str = "evaluation/results/training_data.json"):
    with open(path, "r") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} training pairs")
    return data


def build_examples(training_data: list) -> list:
    """Convert training pairs into InputExample objects."""
    examples = []
    for item in training_data:
        examples.append(InputExample(
            texts=[item["question"], item["positive"]]
        ))
    random.shuffle(examples)
    print(f"✅ Built {len(examples)} training examples")
    return examples


def finetune(
    base_model:  str = "all-MiniLM-L6-v2",
    output_path: str = "models/finetuned-financial-embedder",
    epochs:      int = 3,
    batch_size:  int = 8,
    warmup_steps: int = 10
):
    print(f"\n🔧 Loading base model: {base_model}")
    model = SentenceTransformer(base_model)

    # load data
    training_data = load_training_data()
    examples      = build_examples(training_data)

    # dataloader
    dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=batch_size
    )

    # loss function
    # MultipleNegativesRankingLoss:
    # for each (question, positive_chunk) pair in a batch,
    # all other chunks become automatic negatives
    loss = MultipleNegativesRankingLoss(model)

    total_steps = len(dataloader) * epochs
    print(f"\n📊 Training config:")
    print(f"   Base model:   {base_model}")
    print(f"   Examples:     {len(examples)}")
    print(f"   Epochs:       {epochs}")
    print(f"   Batch size:   {batch_size}")
    print(f"   Total steps:  {total_steps}")
    print(f"   Output:       {output_path}")
    print(f"   Device:       {'mps' if torch.backends.mps.is_available() else 'cpu'}")

    print(f"\n🚀 Starting fine-tuning...\n")

    os.makedirs(output_path, exist_ok=True)

    # fine-tune
    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True
    )

    print(f"\n✅ Fine-tuning complete!")
    print(f"💾 Model saved to: {output_path}")
    return output_path


def compare_models(query: str, base_model: str, finetuned_path: str):
    """Compare retrieval scores before and after fine-tuning."""
    from src.loader import load_all_documents
    from src.chunker import split_into_chunks
    import numpy as np

    print(f"\n🔍 Comparing models on query: '{query}'")

    docs   = load_all_documents("documents")
    chunks = split_into_chunks(docs)
    texts  = [c.content for c in chunks]

    # base model
    base  = SentenceTransformer(base_model)
    ft    = SentenceTransformer(finetuned_path)

    q_base = base.encode(query, normalize_embeddings=True)
    q_ft   = ft.encode(query, normalize_embeddings=True)

    corpus_base = base.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    corpus_ft   = ft.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    scores_base = corpus_base @ q_base
    scores_ft   = corpus_ft   @ q_ft

    top3_base = np.argsort(scores_base)[::-1][:3]
    top3_ft   = np.argsort(scores_ft)[::-1][:3]

    print(f"\n--- Base model top 3 ---")
    for idx in top3_base:
        print(f"  Score: {scores_base[idx]:.3f} | Page {chunks[idx].metadata['page']} | {chunks[idx].content[:100]}...")

    print(f"\n--- Fine-tuned model top 3 ---")
    for idx in top3_ft:
        print(f"  Score: {scores_ft[idx]:.3f} | Page {chunks[idx].metadata['page']} | {chunks[idx].content[:100]}...")


if __name__ == "__main__":
    # step 1 — fine-tune
    output_path = finetune(
        base_model="all-MiniLM-L6-v2",
        output_path="models/finetuned-financial-embedder",
        epochs=3,
        batch_size=8
    )

    # step 2 — compare before vs after
    compare_models(
        query="What is the minimum CET1 capital ratio?",
        base_model="all-MiniLM-L6-v2",
        finetuned_path=output_path
    )