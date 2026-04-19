# test_search.py
from src.embedder import Embedder
from src.indexer import Indexer

embedder = Embedder()
indexer = Indexer()

query = "What are the capital requirements for banks?"

print(f"\n🔍 Query: {query}\n")
query_vector = embedder.embed_query(query)
results = indexer.search(query_vector, top_k=3)

for i, result in enumerate(results):
    print(f"--- Result {i+1} (score: {result.score:.3f}) ---")
    print(f"Source: {result.payload['source']}, Page: {result.payload['page']}")
    print(f"{result.payload['content'][:300]}")
    print()