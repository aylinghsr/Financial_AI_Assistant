# src/indexer.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)
from typing import List
import numpy as np
from src.chunker import Chunk

COLLECTION_NAME = "financial_docs"

class Indexer:
    def __init__(self, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=url)
        self.collection_name = COLLECTION_NAME

    def create_collection(self, vector_size: int = 384):
        """Create Qdrant collection, drop existing one if present."""
        existing = [c.name for c in self.client.get_collections().collections]
        
        if self.collection_name in existing:
            print(f"Dropping existing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Collection '{self.collection_name}' created")

    def index_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Store chunks and their embeddings in Qdrant."""
        points = []

        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={
                    "content": chunk.content,
                    **chunk.metadata
                }
            ))

        # upload in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"  Uploaded batch {i//batch_size + 1} ({len(batch)} points)")

        print(f"✅ Indexed {len(points)} chunks into Qdrant")

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        """Search for most similar chunks."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True
        ).points
        return results