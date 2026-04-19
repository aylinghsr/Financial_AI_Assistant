# ingest.py
from src.loader import load_all_documents
from src.chunker import split_into_chunks
from src.embedder import Embedder
from src.indexer import Indexer

def main():
    # 1. Load
    docs = load_all_documents("documents")

    # 2. Chunk
    chunks = split_into_chunks(docs)

    # 3. Embed
    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks)

    # 4. Index
    indexer = Indexer()
    indexer.create_collection(vector_size=embedder.dimension)
    indexer.index_chunks(chunks, embeddings)

    print("\n🎉 Ingestion complete! All chunks are in Qdrant.")

if __name__ == "__main__":
    main()