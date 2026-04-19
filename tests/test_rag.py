# test_rag.py
from src.loader import load_all_documents
from src.chunker import split_into_chunks
from src.embedder import Embedder
from src.indexer import Indexer
from src.retriever import HybridRetriever
from src.generator import Generator

print("Setting up pipeline...")
docs = load_all_documents("documents")
chunks = split_into_chunks(docs)
embedder = Embedder()
indexer = Indexer()
retriever = HybridRetriever(chunks, embedder, indexer)
generator = Generator()

# three test questions
questions = [
    "What is the minimum Common Equity Tier 1 capital ratio?",
    "How does Basel III define the leverage ratio?",
    "What are the liquidity coverage ratio requirements?"
]

for query in questions:
    print(f"\n{'='*60}")
    print(f"❓  {query}")
    print('='*60)

    # step 1 — retrieve relevant chunks
    results = retriever.retrieve(query, top_k=3)

    # step 2 — generate answer
    response = generator.generate(query, results)

    # print answer
    print(f"\n💬  Answer:\n{response['answer']}")

    # print sources
    print(f"\n📚  Sources used:")
    for s in response['sources']:
        print(f"    - {s['source']}, page {s['page']} (score: {s['score']})")