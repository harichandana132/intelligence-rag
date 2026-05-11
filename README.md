# Document Intelligence — RAG-based Q&A System

An AI-powered Question & Answering system that lets users upload technical PDF documents and query them using a **Hybrid Retrieval pipeline** combining FAISS dense search, BM25 keyword search, Reciprocal Rank Fusion, and cross-encoder reranking — powered by Groq API (Llama 3.3 70B).

---

## Overview

Most RAG systems rely on simple vector similarity search, which often misses exact keyword matches or returns semantically similar but irrelevant chunks. This project solves that by combining **FAISS dense retrieval + BM25 keyword search** using Reciprocal Rank Fusion (RRF), followed by a **cross-encoder reranker** for precise relevance scoring.

Multi-turn conversation is supported — the system maintains chat history so follow-up questions are answered in context.

---

## Architecture
```
User Query
    │
    ▼
┌─────────────────────────────────┐
│          HybridRetriever         │
│  ┌──────────────┐ ┌──────────┐  │
│  │     FAISS    │ │   BM25   │  │
│  │ Dense Search │ │ Keyword  │  │
│  │  (cosine)    │ │  Search  │  │
│  └──────┬───────┘ └────┬─────┘  │
│         └──────┬────────┘        │
│   Reciprocal Rank Fusion (RRF)   │
│                │                 │
│   Cross-Encoder Reranker         │
│  (ms-marco-MiniLM-L-6-v2)        │
└────────────────┬─────────────────┘
                 │
                 ▼
         Top-K Relevant Chunks
                 │
                 ▼
    Groq API — Llama 3.3 70B
    (with multi-turn chat history)
                 │
                 ▼
            Final Answer
```
           

---

## Features

- **Hybrid Retrieval** — FAISS (dense cosine) + BM25 (sparse keyword) combined via Reciprocal Rank Fusion
- **Cross-encoder Reranking** — ms-marco-MiniLM-L-6-v2 is used for precise relevance scoring
- **Recursive Boundary-aware Chunking** — respects paragraph → sentence → word boundaries with overlap
- **Multi-turn Chat** — last 3 conversation exchanges maintained for context-aware follow-up answers
- **PDF Upload** — upload any technical PDF via /upload endpoint or Streamlit UI
- **FastAPI Backend** — /upload, /query, and health check endpoints
- **Streamlit Frontend** — clean chat interface

---

## Tech Stack

|       Component      |                Technology                   |
|----------------------|---------------------------------------------|
|          LLM         | Groq API (Llama 3.3 70B Versatile)          |
|        Embeddings    | Sentence Transformers (all-MiniLM-L6-v2)    |
|     Dense Retrieval  | FAISS (IndexFlatIP + L2 normalization)      |
|    Sparse Retrieval  | BM25 (rank-bm25, BM25Okapi)                 |
|        Fusion        | Reciprocal Rank Fusion (RRF)                |
|       Reranking      | CrossEncoder (ms-marco-MiniLM-L-6-v2)       |
|      PDF Parsing     | PyMuPDF                                     |
|       Backend        | FastAPI + Uvicorn                           |
|       Frontend       | Streamlit                                   |
|       Language       | Python 3.10+                                |

---

## Project Structure

UI/
└── app.py              # Streamlit frontend — chat interface + PDF upload
app/
├── chunker.py          # Recursive boundary-aware chunker (paragraph→sentence→word)
├── retriever.py        # HybridRetriever: FAISS + BM25 + RRF + cross-encoder reranking
├── llm.py              # Groq API client (Llama 3.3 70B) with multi-turn chat history
├── pdf_loader.py       # PDF text extraction using PyMuPDF
├── main.py             # FastAPI backend — /upload, /query endpoints
data/
└── sample_pdfs/        # Sample technical PDFs for testing
requirements.txt

---

## Setup & Installation

git clone https://github.com/harichandana132/intelligence-rag.git
cd intelligence-rag
pip install -r requirements.txt
echo "GROQ_API_KEY=your_api_key_here" > .env
uvicorn app.main:app --reload
streamlit run UI/app.py

---

## How It Works

1. **Upload** a PDF document via the Streamlit UI or /upload endpoint
2. The PDF is parsed using PyMuPDF and split into chunks using recursive boundary-aware splitting
3. Chunks are indexed in both **FAISS** (dense cosine embeddings) and **BM25** (keyword index)
4. On each query, both retrievers return top candidates independently
5. Results are merged using **Reciprocal Rank Fusion (RRF)** with configurable alpha weighting
6. A **cross-encoder reranker** scores and re-orders the merged candidates for precision
7. Top chunks are passed to **Groq API (Llama 3.3 70B)** with conversation history
8. The LLM generates a grounded answer strictly based on retrieved context

---

## API Endpoints

| Method | Endpoint |                Description                   |
|--------|----------|----------------------------------------------|
|  POST  |  /upload | Upload a PDF and rebuild the retriever index |
|  POST  |  /query  | Query the document with optional chat history|
|  GET   |  /       | Health check                                 |

---

## Sample Use Case

Upload a research paper and ask:
- "What dataset was used for training?"
- "What was the main contribution of this paper?"
- "How does this compare to previous approaches?" — follow-up, uses chat history

---

## Future Improvements

- Support for multiple PDFs simultaneously
- Table and figure extraction from PDFs
- Persistent FAISS index storage across sessions
- Evaluation metrics (RAGAS, MRR, NDCG)



Just select all → paste → commit
