# test_retriever.py
from src.loader import load_all_documents
from src.chunker import split_into_chunks
from src.embedder import Embedder
from src.indexer import Indexer
from src.retriever import HybridRetriever

# reload everything
docs = load_all_documents("documents")
chunks = split_into_chunks(docs)
embedder = Embedder()
indexer = Indexer()
retriever = HybridRetriever(chunks, embedder, indexer)

# test with a query that has exact financial terms
query = "What is the Tier 1 capital ratio requirement?"
print(f"\n🔍 Query: {query}\n")

results = retriever.retrieve(query, top_k=3)

for i, r in enumerate(results):
    print(f"--- Result {i+1} (combined: {r['score']:.3f} | vector: {r['vector_score']:.3f} | bm25: {r['bm25_score']:.3f}) ---")
    print(f"Page: {r['metadata']['page']}")
    print(r['content'][:300])
    print()