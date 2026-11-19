from modules.doc_types import (detect_type_keywords, extract_key_terms, favorability_score,
                               compare_documents, retrieve_for_terms, DOC_TYPES)

CARD = [
    "Annual Percentage Rate (APR) for Purchases: 24.99% variable. This APR will vary with the market based on the Prime Rate.",
    "Penalty APR: up to 29.99%. This APR may be applied to your account if you make a late payment. The cardholder's minimum payment is due monthly.",
    "Fees. Annual Fee: None. Foreign Transaction Fee: 3% of each transaction in U.S. dollars. Late Payment: up to $40.",
    "Billing error rights: if you think there is an error on your statement, write to us within 60 days. Cardholder agreement credit limit balance transfer.",
    "Arbitration: any claim will be resolved by binding arbitration. You waive the right to participate in a class action. Grace period of at least 25 days.",
]

def test_detects_credit_card():
    assert detect_type_keywords(" ".join(CARD))["type"] == "credit_card"

def test_generic_text_is_commercial():
    assert detect_type_keywords("This agreement between the parties covers services.")["type"] == "commercial"

def test_retrieval_finds_term_chunks():
    hits = retrieve_for_terms(CARD, DOC_TYPES["credit_card"]["terms"])
    assert 0 in hits["purchase_apr"] and 2 in hits["foreign_fee"] and 4 in hits["arbitration"]

def test_extract_without_llm_returns_every_term():
    rows = extract_key_terms(CARD, "credit_card", use_llm=False)
    assert [r["key"] for r in rows] == [t["key"] for t in DOC_TYPES["credit_card"]["terms"]]
    assert "3%" in next(r for r in rows if r["key"] == "foreign_fee")["value"]

def test_extract_uses_llm_json(mocker):
    mocker.patch("modules.llm_engine.call_llm", return_value='{"terms": [{"key": "annual_fee", "value": "$0", "assessment": "favorable", "note": "No annual fee.", "quote": "Annual Fee: None"}]}')
    rows = extract_key_terms(CARD, "credit_card")
    fee = next(r for r in rows if r["key"] == "annual_fee")
    assert fee["value"] == "$0" and fee["assessment"] == "favorable"

def test_score_and_compare_without_llm():
    a = [{"label": "Fee", "value": "$0", "assessment": "favorable"}, {"label": "APR", "value": "18%", "assessment": "standard"}]
    b = [{"label": "Fee", "value": "$95", "assessment": "unfavorable"}, {"label": "APR", "value": "29%", "assessment": "unfavorable"}]
    assert favorability_score(a) == 75 and favorability_score(b) == 0
    verdict = compare_documents("Card A", a, "Card B", b, "credit_card", use_llm=False)
    assert verdict["better"] == "A" and verdict["differences"]
