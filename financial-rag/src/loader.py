# src/loader.py
import fitz  # pymupdf
import os
from dataclasses import dataclass
from typing import List

@dataclass
class Document:
    content: str
    metadata: dict

def load_pdf(filepath: str) -> List[Document]:
    """Load a PDF and extract text page by page, preserving structure."""
    docs = []
    pdf = fitz.open(filepath)
    filename = os.path.basename(filepath)

    for page_num, page in enumerate(pdf):
        text = page.get_text("text")  # plain text extraction
        
        # skip nearly empty pages
        if len(text.strip()) < 50:
            continue

        docs.append(Document(
            content=text.strip(),
            metadata={
                "source": filename,
                "page": page_num + 1,
                "total_pages": len(pdf)
            }
        ))

    pdf.close()
    print(f"✅ Loaded {len(docs)} pages from {filename}")
    return docs

def load_all_documents(folder: str = "documents") -> List[Document]:
    """Load all PDFs from the documents folder."""
    all_docs = []
    files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    
    if not files:
        print("⚠️  No PDF files found in documents/ folder")
        return []

    for filename in files:
        filepath = os.path.join(folder, filename)
        all_docs.extend(load_pdf(filepath))

    print(f"\n📄 Total pages loaded: {len(all_docs)}")
    return all_docs