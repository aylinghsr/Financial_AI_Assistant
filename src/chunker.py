# src/chunker.py
from typing import List
from src.loader import Document
from dataclasses import dataclass

@dataclass
class Chunk:
    content: str
    metadata: dict

def split_into_chunks(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Chunk]:
    """
    Split documents into overlapping chunks.
    Overlap ensures context isn't lost at boundaries.
    """
    chunks = []

    for doc in documents:
        words = doc.content.split()
        
        if not words:
            continue

        start = 0
        chunk_index = 0

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            # only keep chunks with meaningful content
            if len(chunk_text.strip()) > 100:
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": chunk_index,
                        "chunk_size": len(chunk_words)
                    }
                ))
                chunk_index += 1

            # move forward by (chunk_size - overlap)
            start += chunk_size - chunk_overlap

    print(f"✅ Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks