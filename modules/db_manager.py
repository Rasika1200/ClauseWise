"""
Module: Database Manager (db_manager.py)
Handles all SQLite operations for contract analyses, clause extractions,
HITL feedback loops, and RAG query traces without any ORM.
"""

import os
import json
import sqlite3
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "feedback.db")

# Auto-create data directory on import
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Get a raw sqlite3 database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all required tables if they do not exist."""
    conn = get_connection()
    try:
        # 1. contract_analyses
        conn.execute("""
        CREATE TABLE IF NOT EXISTS contract_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_name TEXT,
            contract_hash TEXT UNIQUE,
            risk_profile TEXT,
            summary TEXT,
            assessment TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # 2. extracted_clauses
        conn.execute("""
        CREATE TABLE IF NOT EXISTS extracted_clauses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_name TEXT,
            clause_type TEXT,
            summary TEXT,
            risk_level TEXT,
            risk_explanation TEXT,
            confidence REAL,
            extracted_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # 3. hitl_feedback
        conn.execute("""
        CREATE TABLE IF NOT EXISTS hitl_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_name TEXT,
            clause_type TEXT,
            risk_level TEXT,
            decision TEXT,
            original_text TEXT,
            edited_text TEXT,
            decided_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # 4. rag_traces
        conn.execute("""
        CREATE TABLE IF NOT EXISTS rag_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            response TEXT,
            retrieved_chunks TEXT,
            latency_ms REAL,
            search_method TEXT,
            query_enhancement TEXT,
            rerank_enabled INTEGER,
            logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
    finally:
        conn.close()


# Run initialization on module import
init_db()


def save_contract_analysis(
    contract_name: str,
    contract_hash: str,
    risk_profile: str,
    summary: str,
    assessment: str,
):
    """
    Save or update a contract analysis. 
    Upserts on contract_hash conflict - updates risk_profile, summary, assessment.
    """
    conn = get_connection()
    try:
        conn.execute("""
        INSERT INTO contract_analyses (contract_name, contract_hash, risk_profile, summary, assessment)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(contract_hash) DO UPDATE SET
            contract_name=excluded.contract_name,
            risk_profile=excluded.risk_profile,
            summary=excluded.summary,
            assessment=excluded.assessment
        """, (contract_name, contract_hash, risk_profile, summary, assessment))
        conn.commit()
    finally:
        conn.close()


def get_contract_analysis(contract_name: str) -> Optional[Dict[str, Any]]:
    """Return the analysis row as a dictionary or None."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM contract_analyses WHERE contract_name = ? ORDER BY uploaded_at DESC LIMIT 1",
            (contract_name,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def save_extracted_clauses(contract_name: str, clauses: List[Dict[str, Any]]):
    """
    Delete all existing clauses for the contract_name, 
    then batch insert the new list of clauses.
    """
    conn = get_connection()
    try:
        # Delete existing
        conn.execute("DELETE FROM extracted_clauses WHERE contract_name = ?", (contract_name,))
        
        # Batch insert
        for c in clauses:
            conn.execute("""
            INSERT INTO extracted_clauses (
                contract_name, clause_type, summary, risk_level, risk_explanation, confidence, extracted_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                contract_name,
                c.get("clause_type"),
                c.get("summary"),
                c.get("risk_level"),
                c.get("risk_explanation"),
                c.get("confidence", 1.0),
                c.get("extracted_text")
            ))
        conn.commit()
    finally:
        conn.close()


def get_extracted_clauses(contract_name: str) -> List[Dict[str, Any]]:
    """Retrieve all extracted clauses for a given contract_name."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM extracted_clauses WHERE contract_name = ?", (contract_name,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_hitl_decision(
    contract_name: str,
    clause_type: str,
    risk_level: str,
    decision: str,
    original_text: str,
    edited_text: str,
):
    """Save a single human-in-the-loop audit decision (plain insert)."""
    conn = get_connection()
    try:
        conn.execute("""
        INSERT INTO hitl_feedback (contract_name, clause_type, risk_level, decision, original_text, edited_text)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (contract_name, clause_type, risk_level, decision, original_text, edited_text))
        conn.commit()
    finally:
        conn.close()


def get_all_hitl_decisions(contract_name: str) -> List[Dict[str, Any]]:
    """Return all HITL decisions logged for a given contract."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM hitl_feedback WHERE contract_name = ?", (contract_name,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_feedback_metrics() -> Dict[str, int]:
    """Return counts of accepted/edited/rejected decisions across all contracts."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT decision, COUNT(*) as cnt FROM hitl_feedback GROUP BY decision")
        rows = cursor.fetchall()
        
        metrics = {"accepted": 0, "edited": 0, "rejected": 0}
        for r in rows:
            dec = r["decision"]
            if dec in metrics:
                metrics[dec] = r["cnt"]
        return metrics
    finally:
        conn.close()


def save_rag_trace(
    query: str,
    response: str,
    retrieved_chunks: List[Dict[str, Any]],
    latency_ms: float,
    search_method: str,
    query_enhancement: str,
    rerank_enabled: bool,
):
    """Serialize retrieved_chunks to JSON and log the RAG telemetry trace."""
    conn = get_connection()
    try:
        chunks_json = json.dumps(retrieved_chunks)
        conn.execute("""
        INSERT INTO rag_traces (
            query, response, retrieved_chunks, latency_ms, search_method, query_enhancement, rerank_enabled
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            query,
            response,
            chunks_json,
            latency_ms,
            search_method,
            query_enhancement,
            1 if rerank_enabled else 0
        ))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    print("Testing db_manager.py...")
    
    # 1. Clean test entries if any to prevent state issues
    conn = get_connection()
    conn.execute("DELETE FROM contract_analyses WHERE contract_name='test_contract.pdf'")
    conn.execute("DELETE FROM extracted_clauses WHERE contract_name='test_contract.pdf'")
    conn.execute("DELETE FROM hitl_feedback WHERE contract_name='test_contract.pdf'")
    conn.commit()
    conn.close()

    # 2. Save and Retrieve contract analysis
    save_contract_analysis(
        contract_name="test_contract.pdf",
        contract_hash="hash_123",
        risk_profile="high",
        summary="Test Summary",
        assessment="Test Assessment"
    )
    analysis = get_contract_analysis("test_contract.pdf")
    assert analysis is not None, "Failed to retrieve analysis!"
    assert analysis["contract_name"] == "test_contract.pdf", "Contract name mismatch!"
    assert analysis["contract_hash"] == "hash_123", "Contract hash mismatch!"
    
    # Verify Upsert (save same hash, diff name/summary)
    save_contract_analysis(
        contract_name="test_contract.pdf",
        contract_hash="hash_123",
        risk_profile="medium",
        summary="Test Summary Updated",
        assessment="Test Assessment Updated"
    )
    analysis_updated = get_contract_analysis("test_contract.pdf")
    assert analysis_updated["risk_profile"] == "medium", "Upsert failed to update risk profile!"
    assert analysis_updated["summary"] == "Test Summary Updated", "Upsert failed to update summary!"

    # 3. Save test clauses and retrieve contract analysis (assert no error)
    test_clauses = [
        {"clause_type": "liability", "summary": "Cap at $50k", "risk_level": "medium", "risk_explanation": "Standard cap", "confidence": 0.95, "extracted_text": "Clause text 1"},
        {"clause_type": "termination", "summary": "30 days notice", "risk_level": "low", "risk_explanation": "Standard term", "confidence": 0.9, "extracted_text": "Clause text 2"}
    ]
    save_extracted_clauses("test_contract.pdf", test_clauses)
    
    # Retrieve via get_contract_analysis to verify contract analysis is still present and retrieve clauses too
    analysis_check = get_contract_analysis("test_contract.pdf")
    assert analysis_check is not None, "get_contract_analysis returned None after saving clauses!"
    clauses_retrieved = get_extracted_clauses("test_contract.pdf")
    assert len(clauses_retrieved) == 2, "Failed to retrieve correct number of clauses!"
    assert clauses_retrieved[0]["clause_type"] == "liability", "Clause type mismatch!"

    # 4. Save HITL decision and retrieve all
    save_hitl_decision(
        contract_name="test_contract.pdf",
        clause_type="liability",
        risk_level="medium",
        decision="accepted",
        original_text="Original text",
        edited_text=""
    )
    decisions = get_all_hitl_decisions("test_contract.pdf")
    assert len(decisions) >= 1, "HITL decisions list length is less than 1!"
    assert decisions[0]["decision"] == "accepted", "HITL decision value mismatch!"

    # 5. Save RAG trace
    test_chunks = [{"text": "Chunk text", "score": 0.85, "source": "test_contract.pdf", "rank": 0}]
    save_rag_trace(
        query="What is the cap?",
        response="It is $50k.",
        retrieved_chunks=test_chunks,
        latency_ms=120.5,
        search_method="Hybrid (RRF Fusion)",
        query_enhancement="None",
        rerank_enabled=True
    )
    
    metrics = get_feedback_metrics()
    assert metrics["accepted"] >= 1, "Feedback metrics calculation issue!"

    print("db_manager verification passed")
