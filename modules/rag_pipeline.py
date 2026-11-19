"""
Module 5: RAG Pipeline (FREE — local embeddings + Groq)
Retrieves relevant chunks and assembles context for the LLM.
"""

import os
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


def generate_query_expansion(query: str) -> List[str]:
    """Use the LLM to generate 3 alternative phrasings of the question."""
    from modules.llm_engine import call_llm, _parse_json
    prompt = f"""
    You are an expert AI Legal Assistant. Generate 3 alternative search queries to retrieve relevant clauses for this contract question.
    Rewrite plain-English wording into the formal legal terms a contract would actually use
    (e.g. "mistake in the documents" -> "untrue statement or omission of a material fact; indemnification";
    "when does it end" -> "termination; expiration date; term of agreement").
    Question: {query}
    
    Return EXACTLY a valid JSON object in this format:
    {{"queries": ["alternative 1", "alternative 2", "alternative 3"]}}
    """
    try:
        result = call_llm(prompt, json_mode=True)
        data = _parse_json(result)
        queries = data.get("queries", [])
        valid_queries = [q for q in queries if isinstance(q, str) and q.strip()]
        if valid_queries:
            return valid_queries
    except Exception as e:
        print(f"Query expansion failed with error: {e}. Falling back to original query.")
    
    return [query]


def generate_hyde_document(query: str) -> str:
    """Ask the LLM to generate a hypothetical legal answer/clause for the question."""
    from modules.llm_engine import call_llm
    prompt = f"""
    You are an expert AI Legal Assistant. Write a hypothetical contract clause or direct answer that would answer the following question.
    Do not include any intro/outro. Write only the hypothetical text.
    Question: {query}
    """
    try:
        result = call_llm(prompt)
        if result and result.strip():
            return result.strip()
    except Exception as e:
        print(f"HyDE document generation failed with error: {e}. Falling back to original query.")
    
    return query


def retrieve(
    query: str,
    top_k: int = 6,
    filter_clause_type: Optional[str] = None,
    filter_source_file: Optional[str] = None,
    search_method: str = "Hybrid (RRF Fusion)",
    query_enhancement: str = "None",
) -> List[Dict]:
    """Retrieve top_k similar chunks using dense, sparse, or hybrid search."""
    from modules.embedder import semantic_search, bm25_search, hybrid_search

    search_queries = [query]
    dense_query = query
    sparse_query = query

    if query_enhancement == "Query Expansion":
        search_queries = [query] + generate_query_expansion(query)
    elif query_enhancement == "HyDE":
        dense_query = generate_hyde_document(query)
        sparse_query = query

    candidates = []
    
    if search_method == "Dense (Semantic)":
        if query_enhancement == "Query Expansion":
            seen = set()
            for q in search_queries:
                results = semantic_search(
                    query=q,
                    top_k=top_k,
                    filter_clause_type=filter_clause_type,
                    filter_source_file=filter_source_file
                )
                for r in results:
                    key = r.get("metadata", {}).get("chunk_id_original") or r["text"]
                    if key not in seen:
                        seen.add(key)
                        candidates.append(r)
        else:
            candidates = semantic_search(
                query=dense_query,
                top_k=top_k,
                filter_clause_type=filter_clause_type,
                filter_source_file=filter_source_file
            )

    elif search_method == "Sparse (BM25)":
        if query_enhancement == "Query Expansion":
            seen = set()
            for q in search_queries:
                results = bm25_search(
                    query=q,
                    top_k=top_k,
                    source_file=filter_source_file,
                    filter_clause_type=filter_clause_type
                )
                for r in results:
                    key = r.get("metadata", {}).get("chunk_id_original") or r["text"]
                    if key not in seen:
                        seen.add(key)
                        candidates.append(r)
        else:
            candidates = bm25_search(
                query=sparse_query,
                top_k=top_k,
                source_file=filter_source_file,
                filter_clause_type=filter_clause_type
            )

    else:  # Hybrid (RRF Fusion)
        if query_enhancement == "Query Expansion":
            # Reciprocal Rank Fusion across the original query and its rewrites, so a chunk
            # that ranks well for several phrasings rises to the top
            fused, by_key = {}, {}
            for q in search_queries:
                results = hybrid_search(
                    query=q,
                    top_k=top_k,
                    source_file=filter_source_file,
                    filter_clause_type=filter_clause_type
                )
                for rank, r in enumerate(results):
                    key = r.get("metadata", {}).get("chunk_id_original") or r["text"]
                    by_key.setdefault(key, r)
                    fused[key] = fused.get(key, 0.0) + 1.0 / (60.0 + rank + 1)
            for key in sorted(fused, key=fused.get, reverse=True):
                chunk = dict(by_key[key])
                chunk["rrf_score"] = round(fused[key], 6)
                chunk["fused_order"] = True
                candidates.append(chunk)
        elif query_enhancement == "HyDE":
            import hashlib
            dense_results = semantic_search(
                query=dense_query,
                top_k=max(top_k * 3, 20),
                filter_clause_type=filter_clause_type,
                filter_source_file=filter_source_file
            )
            sparse_results = bm25_search(
                query=sparse_query,
                top_k=max(top_k * 3, 20),
                source_file=filter_source_file,
                filter_clause_type=filter_clause_type
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
            candidates = []
            for key in fused_keys[:top_k]:
                chunk = dict(chunk_by_id[key])
                chunk["rrf_score"] = round(rrf_scores[key], 6)
                if "similarity" not in chunk:
                    chunk["similarity"] = chunk.get("rrf_score", 0.0)
                candidates.append(chunk)
        else:
            candidates = hybrid_search(
                query=query,
                top_k=top_k,
                source_file=filter_source_file,
                filter_clause_type=filter_clause_type
            )

    return candidates


_reranker_model = None
_reranker_failed = False

def get_reranker():
    """
    Cross-encoder re-ranker (ms-marco-MiniLM-L-6-v2). Enabled wherever neural models are
    enabled (Linux/cloud by default); off on macOS to avoid PyTorch thread deadlocks.
    """
    global _reranker_model, _reranker_failed
    from modules.embedder import neural_models_enabled
    if _reranker_failed or not neural_models_enabled():
        return None
    if _reranker_model is None:
        try:
            import torch
            torch.set_num_threads(1)
            from sentence_transformers import CrossEncoder
            _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        except Exception as e:
            print(f"Cross-encoder unavailable ({e}); ranking by retrieval score instead.")
            _reranker_failed = True
            return None
    return _reranker_model

def rerank(query: str, candidates: List[Dict]) -> List[Dict]:
    """
    Rerank candidates. Uses local cross-encoder if available,
    otherwise falls back to similarity score ordering.
    """
    model = get_reranker()
    if model is not None and candidates:
        pairs = [(query, c["text"]) for c in candidates]
        try:
            scores = model.predict(pairs)
            import math
            for c, score in zip(candidates, scores):
                c["rerank_score"] = float(score)
                # Cross-encoder outputs logits; squash to 0-1 so confidence displays sensibly
                c["similarity"] = 1.0 / (1.0 + math.exp(-float(score)))
            return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        except Exception as e:
            print(f"Error during reranking prediction: {e}")
            pass

    # Fallback: candidates already fused by rank across queries keep their fused order
    if candidates and all(c.get("fused_order") for c in candidates):
        for c in candidates:
            c.setdefault("rerank_score", float(c.get("rrf_score", 0.0)))
        return candidates

    # Fallback to similarity
    for c in candidates:
        if "rerank_score" not in c:
            c["rerank_score"] = float(c.get("similarity", 0.0))
    return sorted(candidates, key=lambda x: x.get("similarity", 0), reverse=True)


def assemble_context(chunks: List[Dict], max_tokens: int = 6000) -> str:
    """Assemble retrieved chunks into a context string with metadata headers."""
    char_budget = max_tokens * 4
    parts = []
    used = 0

    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        header = (f"[Chunk {i+1} | File: {meta.get('source_file','?')} | "
                  f"Page: {meta.get('page_num','?')} | "
                  f"Clause: {meta.get('clause_type','?')} | "
                  f"Risk: {meta.get('risk_level','?')}]")
        block = f"{header}\n{chunk['text']}\n"
        if used + len(block) > char_budget:
            break
        parts.append(block)
        used += len(block)

    return "\n---\n".join(parts)


def rag_retrieve_and_assemble(
    query: str,
    top_k: int = 8,
    rerank_results: bool = True,
    filter_clause_type: Optional[str] = None,
    filter_source_file: Optional[str] = None,
    search_method: str = "Hybrid (RRF Fusion)",
    query_enhancement: str = "None",
) -> Tuple[str, List[Dict]]:
    """Full RAG pipeline: retrieve → rerank → assemble context."""
    retrieval_k = max(top_k * 2, 15) if rerank_results else top_k
    
    candidates = retrieve(
        query=query,
        top_k=retrieval_k,
        filter_clause_type=filter_clause_type,
        filter_source_file=filter_source_file,
        search_method=search_method,
        query_enhancement=query_enhancement
    )
    
    if rerank_results and len(candidates) > 1:
        candidates = rerank(query, candidates)
        candidates = candidates[:top_k]
        
    context = assemble_context(candidates)
    return context, candidates


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Testing rag_pipeline.py with CrossEncoder reranking...")
    from collections import namedtuple
    MockChunk = namedtuple("MockChunk", ["text", "chunk_id", "source_file", "page_num"])
    test_chunks = [
        MockChunk("The liability cap under this agreement shall be fifty thousand dollars ($50,000).", "id_1", "test_contract.pdf", 1),
        MockChunk("This agreement will automatically renew unless terminated with 30 days written notice.", "id_2", "test_contract.pdf", 2)
    ]
    
    from modules.embedder import index_chunks
    index_chunks(test_chunks, topic_labels=None, clear_first=True)
    
    # Test retrieve
    query = "What is the liability limit?"
    context, chunks = rag_retrieve_and_assemble(query, top_k=2, rerank_results=True, filter_source_file="test_contract.pdf")
    
    print(f"Retrieved and reranked {len(chunks)} chunks")
    assert len(chunks) > 0, "Retrieve returned empty chunks!"
    assert "rerank_score" in chunks[0], "First chunk is missing 'rerank_score'!"
    print(f"Top chunk text: {chunks[0]['text']}")
    print(f"Top chunk rerank score: {chunks[0]['rerank_score']}")
    assert isinstance(chunks[0]['rerank_score'], float), "rerank_score is not a float!"
    
    # Test query enhancement helper defaults (without requiring API keys)
    print("Testing query expansion fallback...")
    expanded = generate_query_expansion("cancellation terms")
    assert len(expanded) >= 1
    print(f"Expanded queries: {expanded}")
    
    print("Testing HyDE document generation fallback...")
    hyde = generate_hyde_document("cancellation terms")
    assert len(hyde) > 0
    print(f"HyDE document: {hyde[:50]}...")
    
    print("All pipeline checks passed successfully!")
