from pathlib import Path
from typing import List
import fitz  # PyMuPDF

from app.chunker import chunk_text
from app.retriever import HybridRetriever

DATA_DIR = Path("data/sample_pdfs")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from all pages of a PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def load_pdfs_from_dir(directory: Path = DATA_DIR) -> List[str]:
    """Load and chunk all PDFs from a directory."""
    all_chunks = []
    pdf_files = list(directory.glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError(f"No PDF files found in {directory}")

    for pdf in pdf_files:
        print(f"📄 Processing: {pdf.name}")
        text = extract_text_from_pdf(pdf)
        chunks = chunk_text(text)
        print(f"   → {len(chunks)} chunks extracted")
        all_chunks.extend(chunks)

    print(f"\n✅ Total chunks from all PDFs: {len(all_chunks)}")
    return all_chunks


def load_store(directory: Path = DATA_DIR) -> HybridRetriever:
    """
    Build and return a HybridRetriever loaded with all PDFs from directory.
    """
    retriever = HybridRetriever()
    chunks = load_pdfs_from_dir(directory)
    retriever.add_texts(chunks)
    return retriever


def load_store_from_bytes(pdf_bytes: bytes, filename: str = "upload.pdf") -> HybridRetriever:
    """
    Build a HybridRetriever from a single PDF uploaded as bytes.
    Used for the /upload endpoint.
    """
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        text = extract_text_from_pdf(tmp_path)
        chunks = chunk_text(text)
        print(f"📄 Uploaded '{filename}': {len(chunks)} chunks")

        retriever = HybridRetriever()
        retriever.add_texts(chunks)
        return retriever
    finally:
        os.unlink(tmp_path)
