import pytest
import numpy as np
from modules.clause_matcher import summarize_risk_profile, match_chunks

def test_summarize_risk_profile():
    # Test with empty matches
    res_empty = summarize_risk_profile([])
    assert res_empty["average_risk_score"] == 0.0
    assert res_empty["total_clauses"] == 0
    
    # Test with sample matches
    matches = [
        {"matched_clause": "liability", "risk_level": "high", "risk_score": 0.9},
        {"matched_clause": "termination", "risk_level": "medium", "risk_score": 0.7},
    ]
    res = summarize_risk_profile(matches)
    assert res["average_risk_score"] == 0.8
    assert res["high_risk_count"] == 1
    assert res["total_clauses"] == 2

def test_match_chunks(mocker):
    # Mock canonical names and embeddings matrix (14 clauses total in original, but we mock the mapping)
    mock_names = ["liability", "indemnification", "termination", "other"]
    mock_matrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.707, 0.707],
        [0.1, 0.1]
    ])
    
    mocker.patch("modules.clause_matcher.get_canonical_embeddings", return_value={
        "names": mock_names,
        "matrix": mock_matrix
    })
    
    # Mock target query embedding (matches liability vector [1.0, 0.0] exactly)
    # Patch the function where it resides in modules.embedder
    mocker.patch("modules.embedder.embed_texts", return_value=[[1.0, 0.0]])
    
    matches = match_chunks(["dummy chunk text"], threshold=0.5)
    assert len(matches) > 0
    assert matches[0]["matched_clause"] == "liability"
    assert matches[0]["match_score"] == pytest.approx(1.0)
