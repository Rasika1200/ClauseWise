"""
Module 3: Embedder + Vector Store (FINAL STABLE VERSION)
TF-IDF embeddings — fast, deterministic, no threading issues.
No Streamlit caching → avoids deadlocks.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import hashlib
import warnings
import numpy as np
import chromadb
# streamlit removed to avoid deadlocks in FastAPI
from typing import List, Dict, Optional
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=RuntimeWarning)

load_dotenv()

CHROMA_DIR = None
COLLECTION_NAME = "contract_chunks_semantic"

_bm25_cache = {}

def clear_bm25_cache():
    global _bm25_cache
    _bm25_cache.clear()


class FastTFIDFEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=dim, stop_words='english')
        self.is_fitted = False

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        cleaned = [t.strip() if t and t.strip() else "empty text" for t in texts]

        if not self.is_fitted:
            try:
                matrix = self.vectorizer.fit_transform(cleaned).toarray()
                self.is_fitted = True
            except Exception:
                matrix = np.zeros((len(texts), self.dim))
        else:
            try:
                matrix = self.vectorizer.transform(cleaned).toarray()
            except Exception:
                matrix = np.zeros((len(texts), self.dim))

        if matrix.shape[1] < self.dim:
            pad_width = self.dim - matrix.shape[1]
            matrix = np.pad(matrix, ((0, 0), (0, pad_width)), mode='constant')
        elif matrix.shape[1] > self.dim:
            matrix = matrix[:, :self.dim]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms

        return normalized.tolist()

_tfidf_embedder = FastTFIDFEmbedder(384)

def neural_models_enabled() -> bool:
    """
    Neural models (HuggingFace MiniLM embeddings + cross-encoder re-ranker) are ON by default
    on Linux/cloud. They default OFF on macOS, where PyTorch thread deadlocks were seen locally.
    Override either way with CLAUSEWISE_NEURAL=1 or CLAUSEWISE_NEURAL=0.
    """
    import sys
    flag = os.getenv("CLAUSEWISE_NEURAL")
    if flag is not None:
        return flag.strip().lower() in ("1", "true", "yes", "on")
    return sys.platform != "darwin"


_neural_embedder = None
_neural_failed = False

def _get_neural_embedder():
    """Load all-MiniLM-L6-v2 once; on any failure fall back to TF-IDF for the rest of the session."""
    global _neural_embedder, _neural_failed
    if _neural_failed or not neural_models_enabled():
        return None
    if _neural_embedder is None:
        try:
            import torch
            torch.set_num_threads(1)
            from sentence_transformers import SentenceTransformer
            _neural_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        except Exception as e:
            print(f"Neural embedder unavailable ({e}); using TF-IDF embeddings.")
            _neural_failed = True
            return None
    return _neural_embedder


def embedding_backend() -> str:
    return "MiniLM-L6-v2 (HuggingFace)" if _get_neural_embedder() is not None else "TF-IDF"


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Dense embeddings: HuggingFace all-MiniLM-L6-v2 (384-d) when neural models are enabled,
    otherwise a fast deterministic TF-IDF projection (also 384-d).
    """
    if not texts:
        return []
    model = _get_neural_embedder()
    if model is not None:
        cleaned = [t.strip() if t and t.strip() else "empty text" for t in texts]
        vecs = model.encode(cleaned, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()
    global _tfidf_embedder
    return _tfidf_embedder.embed(texts)



_chroma_client = None

def _get_chroma_client():
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client
        
    if CHROMA_DIR:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    else:
        _chroma_client = chromadb.Client()
    return _chroma_client


def get_collection() -> chromadb.Collection:
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _make_id(chunk_id: str) -> str:
    return hashlib.md5(chunk_id.encode()).hexdigest()


# ── Index ─────────────────────────────────────────────────────────────────────
def clear_collection():
    """Wipes the entire vector store collection to prevent data bleed between contracts."""
    client = _get_chroma_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    clear_bm25_cache()
    global _tfidf_embedder
    _tfidf_embedder = FastTFIDFEmbedder(384)

def index_chunks(chunks, topic_labels: Optional[List[Dict]] = None, clear_first: bool = True) -> int:
    """Embed and store contract chunks."""
    
    if clear_first:
        clear_collection()

    texts = [c.text for c in chunks]
    print(f"📦 Embedding {len(texts)} chunks...")

    embeddings = embed_texts(texts)

    collection = get_collection()

    ids, metadatas, documents = [], [], []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        meta = {
            "source_file": chunk.source_file or "",
            "page_num": int(chunk.page_num) if chunk.page_num is not None else 1,
            "chunk_id_original": chunk.chunk_id or "",
        }

        if topic_labels and i < len(topic_labels):
            label = topic_labels[i]
            # Accept both label shapes: topic_modeler ("clause_type"/"confidence") and
            # clause_matcher.match_chunks ("matched_clause"/"match_score"), which the app passes.
            clause_type = label.get("clause_type") or label.get("matched_clause") or "other"
            confidence = label.get("confidence")
            if confidence is None:
                confidence = label.get("match_score", 0.0)
            meta.update({
                "clause_type": clause_type,
                "risk_level": label.get("risk_level") or "low",
                "confidence": str(confidence),
                "summary": label.get("summary") or "",
            })

        ids.append(_make_id(chunk.chunk_id))
        metadatas.append(meta)
        documents.append(chunk.text)

    # batch insert
    for i in range(0, len(ids), 100):
        collection.upsert(
            ids=ids[i:i+100],
            embeddings=embeddings[i:i+100],
            metadatas=metadatas[i:i+100],
            documents=documents[i:i+100],
        )

    print(f"✅ Indexed {len(ids)} chunks into ChromaDB")
    return len(ids)


# ── Search ────────────────────────────────────────────────────────────────────
def semantic_search(
    query: str,
    top_k: int = 5,
    filter_clause_type: Optional[str] = None,
    filter_risk_level: Optional[str] = None,
    filter_source_file: Optional[str] = None,
) -> List[Dict]:

    if not query.strip():
        return []

    query_embedding = embed_texts([query])[0]
    collection = get_collection()

    where = {}
    if filter_clause_type:
        where["clause_type"] = filter_clause_type
    if filter_risk_level:
        where["risk_level"] = filter_risk_level
    if filter_source_file:
        where["source_file"] = filter_source_file

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=min(top_k, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )

    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    return [
        {
            "text": doc,
            "metadata": meta,
            "similarity": round(1 - dist, 4),
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


# ── Stats ─────────────────────────────────────────────────────────────────────
def get_collection_stats() -> Dict:
    return {
        "total_chunks": get_collection().count(),
        "collection_name": COLLECTION_NAME,
        "persist_dir": CHROMA_DIR,
    }


# ── Sparse & Hybrid Search ────────────────────────────────────────────────────
def _tokenize_text(text: str) -> List[str]:
    import re
    return re.findall(r'\b\w+\b', text.lower())

def get_bm25_model(source_file: str) -> Optional[tuple]:
    """
    Get or fit the BM25 model for a specific source_file.
    Returns (bm25_model, chunks) or None if no chunks are found.
    """
    global _bm25_cache
    if source_file in _bm25_cache:
        return _bm25_cache[source_file]

    from rank_bm25 import BM25Okapi
    collection = get_collection()
    results = collection.get(where={"source_file": source_file})
    
    if not results or not results.get("documents"):
        return None

    documents = results["documents"]
    metadatas = results["metadatas"]
    ids = results["ids"]

    chunks = [
        {
            "text": doc,
            "metadata": meta,
            "id": cid
        }
        for doc, meta, cid in zip(documents, metadatas, ids)
    ]

    tokenized_docs = [_tokenize_text(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_docs)

    _bm25_cache[source_file] = (bm25, chunks)
    return bm25, chunks

def bm25_search(
    query: str,
    top_k: int = 5,
    source_file: Optional[str] = None,
    filter_clause_type: Optional[str] = None,
    filter_risk_level: Optional[str] = None,
) -> List[Dict]:
    """
    Sparse keyword search using rank_bm25.
    """
    if not query.strip() or not source_file:
        return []

    model_info = get_bm25_model(source_file)
    if not model_info:
        return []

    bm25, chunks = model_info
    tokenized_query = _tokenize_text(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_results = []
    for chunk, score in zip(chunks, scores):
        meta = chunk.get("metadata", {})
        if filter_clause_type and meta.get("clause_type") != filter_clause_type:
            continue
        if filter_risk_level and meta.get("risk_level") != filter_risk_level:
            continue
            
        chunk_copy = dict(chunk)
        chunk_copy["sparse_score"] = float(score)
        chunk_copy["similarity"] = float(score)  # fallback compatibility
        ranked_results.append(chunk_copy)

    ranked_results.sort(key=lambda x: x["sparse_score"], reverse=True)
    return ranked_results[:top_k]

def hybrid_search(
    query: str,
    top_k: int = 5,
    source_file: Optional[str] = None,
    filter_clause_type: Optional[str] = None,
    filter_risk_level: Optional[str] = None,
) -> List[Dict]:
    """
    Hybrid Search combining Dense (ChromaDB semantic) and Sparse (BM25 keyword)
    fused using Reciprocal Rank Fusion (RRF).
    """
    if not query.strip():
        return []

    # Get dense candidate search
    dense_results = semantic_search(
        query=query,
        top_k=max(top_k * 3, 20),
        filter_clause_type=filter_clause_type,
        filter_risk_level=filter_risk_level,
        filter_source_file=source_file
    )

    if not source_file:
        return dense_results[:top_k]

    # Get sparse candidate search
    sparse_results = bm25_search(
        query=query,
        top_k=max(top_k * 3, 20),
        source_file=source_file,
        filter_clause_type=filter_clause_type,
        filter_risk_level=filter_risk_level
    )

    rrf_scores = {}
    chunk_by_id = {}

    def get_chunk_key(c):
        meta = c.get("metadata", {})
        return meta.get("chunk_id_original") or hashlib.md5(c["text"].encode()).hexdigest()

    for rank, chunk in enumerate(dense_results):
        key = get_chunk_key(chunk)
        chunk_by_id[key] = chunk
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60.0 + (rank + 1)))

    for rank, chunk in enumerate(sparse_results):
        key = get_chunk_key(chunk)
        if key not in chunk_by_id:
            chunk_by_id[key] = chunk
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60.0 + (rank + 1)))

    fused_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    final_results = []
    for key in fused_keys[:top_k]:
        chunk = dict(chunk_by_id[key])
        chunk["rrf_score"] = round(rrf_scores[key], 6)
        if "similarity" not in chunk:
            chunk["similarity"] = chunk.get("rrf_score", 0.0)
        final_results.append(chunk)

    return final_results


if __name__ == "__main__":
    print("Testing embedder.py with BM25 and Hybrid search...")
    # Create simple mock chunks
    from collections import namedtuple
    MockChunk = namedtuple("MockChunk", ["text", "chunk_id", "source_file", "page_num"])
    
    test_chunks = [
        MockChunk("The liability cap under this agreement shall be fifty thousand dollars ($50,000).", "id_1", "test_contract.pdf", 1),
        MockChunk("This agreement will automatically renew unless terminated with 30 days written notice.", "id_2", "test_contract.pdf", 2),
        MockChunk("Executive shall be paid a base salary of two hundred thousand dollars annually.", "id_3", "test_contract.pdf", 3)
    ]
    
    # Index them
    index_chunks(test_chunks, topic_labels=None, clear_first=True)
    
    # Run tests
    bm25_res = bm25_search("liability cap", top_k=2, source_file="test_contract.pdf")
    print(f"BM25 Search Results count: {len(bm25_res)}")
    assert len(bm25_res) > 0, "BM25 search returned empty results!"
    print(f"Top BM25 chunk: {bm25_res[0]['text']}")
    
    hybrid_res = hybrid_search("base salary", top_k=2, source_file="test_contract.pdf")
    print(f"Hybrid Search Results count: {len(hybrid_res)}")
    assert len(hybrid_res) > 0, "Hybrid search returned empty results!"
    print(f"Top Hybrid chunk: {hybrid_res[0]['text']}")
    print("All checks passed successfully!")