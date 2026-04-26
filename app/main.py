from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

from app.pdf_loader import load_store, load_store_from_bytes
from app.retriever import HybridRetriever
from app.llm import LLMClient

# ---- FastAPI app ----
app = FastAPI(
    title="Document Intelligence RAG API",
    description="Hybrid RAG with FAISS + BM25 + Cross-encoder reranking + Multi-turn chat",
    version="2.0.0"
)

# ---- Request / Response Models ----
class ChatTurn(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    chat_history: Optional[List[ChatTurn]] = []

class Chunk(BaseModel):
    text: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Chunk]

# ---- Load components at startup ----
print("🔄 Loading RAG components...")
retriever: HybridRetriever = load_store()
llm = LLMClient()
print("✅ RAG system ready")

# ---- Query Endpoint ----
@app.post("/query", response_model=QueryResponse)
def query_docs(req: QueryRequest):
    # 1. Hybrid retrieval + reranking
    results = retriever.search(req.query, top_k=req.top_k)

    if not results:
        return QueryResponse(
            answer="No relevant information found in the document.",
            sources=[]
        )

    # 2. Build context from top reranked chunks
    context = "\n\n---\n\n".join([r["text"] for r in results])

    # 3. Convert chat history for LLM
    history = [{"role": t.role, "content": t.content} for t in req.chat_history]

    # 4. Generate answer with multi-turn support
    answer = llm.generate_answer(req.query, context, chat_history=history)

    # 5. Return answer + sources
    sources = [Chunk(text=r["text"], score=r["score"]) for r in results]
    return QueryResponse(answer=answer, sources=sources)


# ---- Upload Endpoint (replaces hardcoded PDF directory) ----
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a new PDF and rebuild the retriever index on it."""
    global retriever

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    retriever = load_store_from_bytes(pdf_bytes, filename=file.filename)
    return {"status": "success", "message": f"'{file.filename}' indexed successfully."}


# ---- Health Check ----
@app.get("/")
def health():
    return {
        "status": "RAG API running",
        "version": "2.0.0",
        "features": ["hybrid-search", "reranking", "multi-turn-chat", "pdf-upload"]
    }
