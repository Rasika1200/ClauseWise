"""
Module: Telemetry Logger (telemetry.py)
Provides tracing for RAG queries. Always logs locally to SQLite (via db_manager)
and optionally logs to Langfuse cloud if API keys are present.
"""

import os
from typing import List, Dict, Any

class TelemetryLogger:
    """Telemetry logger supporting local SQLite log backups and optional Langfuse cloud tracking."""
    
    def __init__(self):
        self.langfuse = None
        self.use_langfuse = False
        
        # Check for Langfuse keys in environment
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if public_key and secret_key:
            try:
                # pyrefly: ignore [missing-import]
                from langfuse import Langfuse
                self.langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                self.use_langfuse = True
                print("🚀 Langfuse Observability initialized successfully!")
            except Exception as e:
                print(f"Failed to load or initialize Langfuse: {e}")
                self.langfuse = None
                self.use_langfuse = False

    def log_query_trace(
        self,
        query: str,
        response: str,
        retrieved_chunks: List[Dict[str, Any]],
        latency_ms: float,
        search_method: str,
        query_enhancement: str,
        rerank_enabled: bool
    ):
        """
        Always save trace to local SQLite database, 
        and optionally log spans and trace to Langfuse.
        """
        from modules import db_manager

        # 1. Always log locally to SQLite
        try:
            db_manager.save_rag_trace(
                query=query,
                response=response,
                retrieved_chunks=retrieved_chunks,
                latency_ms=latency_ms,
                search_method=search_method,
                query_enhancement=query_enhancement,
                rerank_enabled=rerank_enabled
            )
        except Exception as e:
            print(f"Error saving local RAG trace to SQLite: {e}")

        # 2. Push to Langfuse if enabled
        if self.use_langfuse and self.langfuse:
            try:
                # Top level trace
                trace = self.langfuse.trace(
                    name="clausewise-rag-query",
                    input=query,
                    output=response,
                    metadata={
                        "search_method": search_method,
                        "query_enhancement": query_enhancement,
                        "rerank_enabled": rerank_enabled,
                        "latency_ms": latency_ms
                    }
                )
                
                # Span for retrieval phase
                retrieval_span = trace.span(
                    name="Retrieval",
                    input={"query": query},
                    output={"chunks": [c.get("text", "")[:120] for c in retrieved_chunks]}
                )
                
                # Add chunks as sub-spans or items
                for c in retrieved_chunks:
                    retrieval_span.event(
                        name="Retrieved Chunk",
                        input={"rank": c.get("rank"), "score": c.get("score")},
                        output=c.get("text", "")
                    )
                retrieval_span.end()

                # Span for LLM generation phase
                generation_span = trace.span(
                    name="Generation",
                    input={"prompt_context_size": len(retrieved_chunks)},
                    output={"response": response}
                )
                generation_span.end()
                
            except Exception as e:
                # Failure must NEVER crash the main app
                print(f"Failed to log trace to Langfuse: {e}")

    def get_local_trace_summary(self) -> Dict[str, Any]:
        """Query local database for trace counts, average latency, and last trace logged date."""
        from modules import db_manager
        
        conn = db_manager.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as count, 
                    AVG(latency_ms) as avg_latency, 
                    MAX(logged_at) as last_logged 
                FROM rag_traces
            """)
            row = cursor.fetchone()
            if row:
                return {
                    "count": row["count"] or 0,
                    "avg_latency": round(row["avg_latency"] or 0.0, 2),
                    "last_logged": row["last_logged"] or "N/A"
                }
            return {"count": 0, "avg_latency": 0.0, "last_logged": "N/A"}
        except Exception as e:
            print(f"Error querying local trace summary: {e}")
            return {"count": 0, "avg_latency": 0.0, "last_logged": "N/A"}
        finally:
            conn.close()


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Testing telemetry.py logger...")
    
    logger = TelemetryLogger()
    chunks = [{"text": "Test chunk contents", "score": 0.92, "source": "dummy.pdf", "rank": 1}]
    
    logger.log_query_trace(
        query="What is the test question?",
        response="This is a test answer.",
        retrieved_chunks=chunks,
        latency_ms=150.0,
        search_method="Hybrid (RRF Fusion)",
        query_enhancement="None",
        rerank_enabled=True
    )
    
    summary = logger.get_local_trace_summary()
    print("Trace Summary:", summary)
    assert summary["count"] >= 1, "Failed to register local trace!"
    print("telemetry.py verification passed successfully!")
