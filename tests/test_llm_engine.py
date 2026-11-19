# pyrefly: ignore [missing-import]
import pytest
from modules.llm_engine import verify_document_type, extract_clauses, answer_question, generate_dynamic_questions

def test_verify_document_type(mocker):
    # Mock call_llm to return a valid JSON string
    mocker.patch(
        "modules.llm_engine.call_llm",
        return_value='{"is_contract": true, "reason": "This is a lease agreement."}'
    )
    
    result = verify_document_type("Some random lease contract text")
    assert result["is_contract"] is True
    assert result["reason"] == "This is a lease agreement."

def test_extract_clauses(mocker):
    # Mock call_llm to return the complex JSON required by extract_clauses
    mock_json = """{
        "executive_summary": "Overall lease contract.",
        "completeness_assessment": "Standard lease protections present.",
        "risk_assessment": "No critical risks identified.",
        "risk_sources": ["Section 4"],
        "readability_assessment": "Clear language.",
        "fairness_summary": "Balanced",
        "fairness_details": "Balanced between tenant and landlord.",
        "fairness_sources": ["Rent clause"],
        "red_flags": [],
        "clauses": [
            {
                "clause_type": "payment",
                "summary": "Rent is $1,000 monthly.",
                "risk_level": "low",
                "extracted_text": "Tenant shall pay $1,000 per month.",
                "risk_explanation": "Standard rent provision.",
                "confidence_score": 0.95
            }
        ]
    }"""
    mocker.patch("modules.llm_engine.call_llm", return_value=mock_json)
    
    result = extract_clauses("Tenant shall pay $1,000 per month.")
    assert result.executive_summary == "Overall lease contract."
    assert len(result.clauses) == 1
    assert result.clauses[0].clause_type == "payment"
    assert result.clauses[0].risk_level == "low"
    assert result.clauses[0].confidence == 0.95

def test_answer_question(mocker):
    # Mock call_llm to return a text answer
    mocker.patch("modules.llm_engine.call_llm", return_value="The liability cap is $50,000.")
    
    # Mock chunks used to test sources extraction
    chunks = [
        {"metadata": {"clause_type": "liability", "source_file": "doc.pdf"}, "similarity": 0.85}
    ]
    
    result = answer_question("What is the liability cap?", "Context info", [], chunks_used=chunks)
    assert result.answer == "The liability cap is $50,000."
    assert result.confidence == 0.85
    assert "Liability" in result.sources

def test_generate_dynamic_questions(mocker):
    mocker.patch("modules.llm_engine.call_llm", return_value='{"questions": ["Q1", "Q2", "Q3"]}')
    
    result = generate_dynamic_questions("Lease summary")
    assert len(result) == 3
    assert result[0] == "Q1"
