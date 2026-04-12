# src/embedder.py
from sentence_transformers import SentenceTransformer
from typing import List
from src.chunker import Chunk
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # all-MiniLM-L6-v2 output size

    def embed_chunks(self, chunks: List[Chunk]) -> List[np.ndarray]:
        """Embed a list of chunks, returns list of vectors."""
        texts = [chunk.content for chunk in chunks]
        
        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True  # important for cosine similarity
        )
        print(f"✅ Embeddings done, shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.model.encode(query, normalize_embeddings=True)