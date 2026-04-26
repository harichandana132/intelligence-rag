"""
Hybrid Retriever: combines dense vector search (FAISS) + sparse keyword search (BM25),
then re-ranks results using a cross-encoder for maximum relevance.
"""

from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        alpha: float = 0.6,          # weight for dense vs sparse (0=all BM25, 1=all FAISS)
    ):
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(embedding_model)

        print("Loading cross-encoder reranker...")
        self.reranker = CrossEncoder(reranker_model)

        self.alpha = alpha
        self.texts: List[str] = []
        self.faiss_index = None
        self.bm25 = None

    def add_texts(self, texts: List[str]):
        """Build both FAISS index and BM25 index from text chunks."""
        self.texts = texts

        # --- Dense index (FAISS) ---
        print(f"Encoding {len(texts)} chunks...")
        embeddings = self.embedder.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings, dtype="float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalization
        self.faiss_index.add(embeddings)

        # --- Sparse index (BM25) ---
        print("Building BM25 index...")
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

        print(f"✅ Indexed {len(texts)} chunks (FAISS + BM25)")

    def search(self, query: str, top_k: int = 5, fetch_k: int = 20) -> List[Dict]:
        """
        1. Retrieve top fetch_k candidates from both FAISS and BM25
        2. Merge with reciprocal rank fusion
        3. Re-rank top candidates with cross-encoder
        4. Return top_k results
        """
        if not self.texts:
            return []

        fetch_k = min(fetch_k, len(self.texts))

        # --- Dense retrieval ---
        query_emb = self.embedder.encode([query], normalize_embeddings=True)
        query_emb = np.array(query_emb, dtype="float32")
        dense_scores, dense_indices = self.faiss_index.search(query_emb, fetch_k)
        dense_indices = dense_indices[0]
        dense_scores = dense_scores[0]

        # --- Sparse retrieval (BM25) ---
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:fetch_k]

        # --- Reciprocal Rank Fusion (RRF) ---
        rrf_scores: Dict[int, float] = {}
        k_rrf = 60  # standard RRF constant

        for rank, idx in enumerate(dense_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + self.alpha / (k_rrf + rank)

        for rank, idx in enumerate(bm25_top_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - self.alpha) / (k_rrf + rank)

        # Get top candidates by RRF score
        candidate_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:fetch_k]
        candidate_texts = [self.texts[i] for i in candidate_indices]

        # --- Cross-encoder Re-ranking ---
        pairs = [(query, text) for text in candidate_texts]
        rerank_scores = self.reranker.predict(pairs)

        # Sort by reranker score
        ranked = sorted(
            zip(candidate_indices, candidate_texts, rerank_scores),
            key=lambda x: x[2],
            reverse=True
        )[:top_k]

        return [
            {"text": text, "score": float(score), "index": idx}
            for idx, text, score in ranked
        ]
