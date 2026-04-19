# src/retriever.py
from rank_bm25 import BM25Okapi
from src.chunker import Chunk
from src.embedder import Embedder
from src.indexer import Indexer
from typing import List
import numpy as np

class HybridRetriever:
    def __init__(self, chunks: List[Chunk], embedder: Embedder, indexer: Indexer):
        self.chunks = chunks
        self.embedder = embedder
        self.indexer = indexer

        # build BM25 index over all chunk texts
        print("Building BM25 index...")
        tokenized = [chunk.content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"✅ BM25 index built over {len(chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[dict]:
        """
        Hybrid retrieval combining vector search and BM25.
        alpha=0.5 means equal weight to both.
        alpha=1.0 means pure vector, alpha=0.0 means pure BM25.
        """
        # --- Vector search scores ---
        query_vector = self.embedder.embed_query(query)
        vector_results = self.indexer.search(query_vector, top_k=top_k * 2)

        # map chunk id → vector score
        vector_scores = {r.id: r.score for r in vector_results}

        # --- BM25 scores ---
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # normalize BM25 scores to 0-1 range
        bm25_max = bm25_scores.max()
        if bm25_max > 0:
            bm25_scores = bm25_scores / bm25_max

        # --- Combine scores ---
        combined = {}
        all_ids = set(vector_scores.keys()) | set(range(len(self.chunks)))

        for idx in vector_scores.keys():
            vec_score = vector_scores.get(idx, 0.0)
            bm25_score = float(bm25_scores[idx]) if idx < len(bm25_scores) else 0.0
            combined[idx] = alpha * vec_score + (1 - alpha) * bm25_score

        # sort by combined score
        top_ids = sorted(combined, key=combined.get, reverse=True)[:top_k]

        # build results
        results = []
        for idx in top_ids:
            chunk = self.chunks[idx]
            results.append({
                "content": chunk.content,
                "metadata": chunk.metadata,
                "score": combined[idx],
                "vector_score": vector_scores.get(idx, 0.0),
                "bm25_score": float(bm25_scores[idx]) if idx < len(bm25_scores) else 0.0
            })

        return results