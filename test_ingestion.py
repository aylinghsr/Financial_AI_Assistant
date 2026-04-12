# test_ingestion.py
from src.loader import load_all_documents
from src.chunker import split_into_chunks

# Load
docs = load_all_documents("documents")

# Chunk
chunks = split_into_chunks(docs)

# Inspect a few
print("\n--- Sample Chunk ---")
print(chunks[0].content[:300])
print("\n--- Metadata ---")
print(chunks[0].metadata)
print(f"\n--- Total chunks ready for embedding: {len(chunks)} ---")