"""
Module: Document-Type Packs
- Detects what kind of agreement a document is (LLM, with keyword fallback)
- Each type has a checklist of key terms that matter for that kind of document
- Extracts each term with targeted per-term retrieval (BM25 over the document's own chunks)
  plus one structured LLM call, then scores how favorable the document is for the party
  the pack protects (e.g. the cardholder, the tenant, the employee)
- Compares two documents of the same type term by term
"""

import json
import re
from typing import Dict, List, Optional

# ── Type packs ────────────────────────────────────────────────────────────────
# Each term: key, label, query (retrieval keywords), what to extract.
DOC_TYPES: Dict[str, Dict] = {
    "credit_card": {
        "label": "Credit Card Agreement",
        "perspective": "the cardholder",
        "keywords": ["cardholder", "annual percentage rate", "apr", "credit limit", "minimum payment",
                     "balance transfer", "cash advance", "billing cycle", "grace period", "penalty apr",
                     "foreign transaction", "card member", "cardmember", "statement balance"],
        "terms": [
            {"key": "purchase_apr", "label": "Purchase APR", "query": "annual percentage rate APR purchases variable prime rate margin"},
            {"key": "penalty_apr", "label": "Penalty APR & triggers", "query": "penalty APR late payment returned payment default rate how long"},
            {"key": "cash_advance", "label": "Cash advance APR & fee", "query": "cash advance APR fee transaction"},
            {"key": "balance_transfer", "label": "Balance transfer APR & fee", "query": "balance transfer APR fee intro introductory"},
            {"key": "annual_fee", "label": "Annual fee", "query": "annual fee membership fee"},
            {"key": "late_fee", "label": "Late payment fee", "query": "late payment fee up to returned payment fee"},
            {"key": "foreign_fee", "label": "Foreign transaction fee", "query": "foreign transaction fee currency conversion outside united states"},
            {"key": "grace_period", "label": "Grace period", "query": "grace period interest free days due date pay entire balance"},
            {"key": "disputes", "label": "Billing dispute & chargeback rights", "query": "billing error dispute rights notify us in writing 60 days unauthorized purchase chargeback"},
            {"key": "arbitration", "label": "Arbitration & class-action waiver", "query": "arbitration class action waiver jury trial opt out"},
            {"key": "rate_changes", "label": "Changing terms / rates", "query": "change terms notice 45 days amend agreement rates"},
        ],
    },
    "loan": {
        "label": "Loan / Credit Agreement",
        "perspective": "the borrower",
        "keywords": ["borrower", "lender", "principal", "interest rate", "promissory", "repayment",
                     "prepayment", "collateral", "default", "maturity date", "installment", "origination fee"],
        "terms": [
            {"key": "principal", "label": "Loan amount", "query": "principal amount loan amount credit facility"},
            {"key": "interest", "label": "Interest rate", "query": "interest rate APR fixed variable per annum"},
            {"key": "fees", "label": "Origination & other fees", "query": "origination fee closing costs fees charges"},
            {"key": "repayment", "label": "Repayment schedule", "query": "repayment installment monthly payment schedule maturity date"},
            {"key": "prepayment", "label": "Prepayment penalty", "query": "prepayment penalty pay early premium"},
            {"key": "late", "label": "Late fees", "query": "late charge late fee past due"},
            {"key": "default", "label": "Events of default & acceleration", "query": "event of default acceleration immediately due"},
            {"key": "collateral", "label": "Collateral / security", "query": "collateral security interest lien guarantee"},
            {"key": "arbitration", "label": "Arbitration & governing law", "query": "arbitration governing law jurisdiction jury waiver"},
        ],
    },
    "nda": {
        "label": "Non-Disclosure Agreement",
        "perspective": "the receiving party",
        "keywords": ["confidential information", "non-disclosure", "nondisclosure", "disclosing party",
                     "receiving party", "trade secret", "permitted disclosure"],
        "terms": [
            {"key": "definition", "label": "What counts as confidential", "query": "confidential information means includes definition marked"},
            {"key": "exclusions", "label": "Exclusions", "query": "shall not include publicly available independently developed already known exclusions"},
            {"key": "obligations", "label": "Use & protection obligations", "query": "receiving party shall use only purpose protect degree of care"},
            {"key": "duration", "label": "How long it lasts", "query": "term years survive termination duration obligations"},
            {"key": "return", "label": "Return / destruction", "query": "return or destroy materials upon request termination"},
            {"key": "compelled", "label": "Compelled disclosure", "query": "required by law subpoena court order notice"},
            {"key": "remedies", "label": "Remedies", "query": "injunctive relief irreparable harm damages remedies"},
            {"key": "non_solicit", "label": "Non-solicit / non-compete", "query": "non-solicitation non-compete hire employees"},
        ],
    },
    "employment": {
        "label": "Employment Agreement",
        "perspective": "the employee",
        "keywords": ["employee", "employer", "employment", "salary", "base compensation", "at-will",
                     "benefits", "severance", "duties", "non-compete", "termination for cause"],
        "terms": [
            {"key": "role", "label": "Role & duties", "query": "position title duties responsibilities report to"},
            {"key": "pay", "label": "Base pay", "query": "base salary compensation annual rate paid"},
            {"key": "bonus_equity", "label": "Bonus & equity", "query": "bonus incentive equity stock options vesting RSU"},
            {"key": "benefits", "label": "Benefits & PTO", "query": "benefits paid time off vacation health insurance"},
            {"key": "termination", "label": "Termination terms", "query": "termination at-will for cause without cause notice"},
            {"key": "severance", "label": "Severance", "query": "severance payment release months salary continuation"},
            {"key": "non_compete", "label": "Non-compete / non-solicit", "query": "non-compete non-solicitation restricted period competitor"},
            {"key": "ip", "label": "IP assignment", "query": "inventions intellectual property assignment work product"},
            {"key": "disputes", "label": "Dispute resolution", "query": "arbitration governing law disputes jury"},
        ],
    },
    "lease": {
        "label": "Lease Agreement",
        "perspective": "the tenant",
        "keywords": ["landlord", "tenant", "lessor", "lessee", "premises", "rent", "security deposit",
                     "lease term", "sublet", "occupancy"],
        "terms": [
            {"key": "rent", "label": "Rent & increases", "query": "monthly rent base rent increase escalation"},
            {"key": "term", "label": "Lease term & renewal", "query": "term commencement expiration renewal option"},
            {"key": "deposit", "label": "Security deposit", "query": "security deposit return deductions"},
            {"key": "late", "label": "Late fees", "query": "late fee grace period rent past due"},
            {"key": "maintenance", "label": "Repairs & maintenance", "query": "repairs maintenance responsible landlord tenant"},
            {"key": "early_exit", "label": "Early termination", "query": "early termination break lease penalty notice"},
            {"key": "sublet", "label": "Subletting / assignment", "query": "sublet sublease assign consent"},
            {"key": "entry", "label": "Landlord entry", "query": "landlord entry access notice inspect"},
        ],
    },
    "commercial": {
        "label": "Commercial Contract",
        "perspective": "a balanced reviewer (flag terms that are one-sided)",
        "keywords": ["agreement", "party", "parties", "services", "supplier", "customer", "company",
                     "indemnify", "warranty", "liability", "governing law"],
        "terms": [
            {"key": "parties", "label": "Parties & purpose", "query": "agreement between parties entered into purpose whereas"},
            {"key": "term", "label": "Term & termination", "query": "term termination terminate expiration notice"},
            {"key": "payment", "label": "Payment & costs", "query": "fees payment expenses costs invoice bear"},
            {"key": "liability", "label": "Liability cap", "query": "limitation of liability cap aggregate consequential damages"},
            {"key": "indemnity", "label": "Indemnification", "query": "indemnify hold harmless losses claims untrue statement"},
            {"key": "ip_conf", "label": "IP & confidentiality", "query": "intellectual property confidential information license"},
            {"key": "assignment", "label": "Assignment", "query": "assign assignment successors consent"},
            {"key": "law", "label": "Governing law & disputes", "query": "governing law jurisdiction arbitration jury waiver"},
        ],
    },
}

ASSESSMENTS = ("favorable", "standard", "unfavorable", "not_found")


def type_options() -> Dict[str, str]:
    return {k: v["label"] for k, v in DOC_TYPES.items()}


# ── Detection ─────────────────────────────────────────────────────────────────
def detect_type_keywords(text: str) -> Dict:
    """Score each pack by keyword hits; 'commercial' is the fallback."""
    t = text.lower()
    scores = {}
    for key, pack in DOC_TYPES.items():
        if key == "commercial":
            continue
        scores[key] = sum(t.count(kw) for kw in pack["keywords"])
    best = max(scores, key=scores.get) if scores else "commercial"
    if scores.get(best, 0) < 5:
        return {"type": "commercial", "confidence": 0.4, "method": "keywords"}
    total = sum(scores.values()) or 1
    return {"type": best, "confidence": round(min(0.95, scores[best] / total + 0.2), 2), "method": "keywords"}


def detect_document_type(text: str, use_llm: bool = True) -> Dict:
    """Detect the document type from its opening text. Returns {type, confidence, method}."""
    sample = text[:6000]
    kw = detect_type_keywords(sample)
    if not use_llm:
        return kw
    try:
        from modules.llm_engine import call_llm, _parse_json
        options = "\n".join(f"- {k}: {v['label']}" for k, v in DOC_TYPES.items())
        prompt = f"""Classify this document into exactly one type.
Types:
{options}
Use "commercial" for any business contract that is not one of the other types.

Document excerpt:
\"\"\"{sample[:4000]}\"\"\"

Respond ONLY with JSON: {{"type": "<one key from the list>", "confidence": 0.0}}"""
        data = _parse_json(call_llm(prompt, json_mode=True))
        dtype = str(data.get("type", "")).strip().lower()
        if dtype in DOC_TYPES:
            return {"type": dtype, "confidence": float(data.get("confidence", 0.8) or 0.8), "method": "llm"}
    except Exception as e:
        print(f"Type detection via LLM failed ({e}); using keyword detection.")
    return kw


# ── Per-term retrieval ────────────────────────────────────────────────────────
def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9%$]+", text.lower())


def retrieve_for_terms(chunk_texts: List[str], terms: List[Dict], per_term: int = 2) -> Dict[str, List[int]]:
    """BM25 over this document's own chunks, one query per checklist term. Returns term key -> chunk indices."""
    from rank_bm25 import BM25Okapi
    if not chunk_texts:
        return {t["key"]: [] for t in terms}
    bm25 = BM25Okapi([_tokenize(c) or ["empty"] for c in chunk_texts])
    out = {}
    for term in terms:
        scores = bm25.get_scores(_tokenize(term["query"]))
        ranked = sorted(range(len(chunk_texts)), key=lambda i: scores[i], reverse=True)
        out[term["key"]] = [i for i in ranked[:per_term] if scores[i] > 0]
    return out


def _best_sentence(text: str, query: str) -> str:
    q = set(_tokenize(query))
    sents = re.split(r"(?<=[.;])\s+", text)
    best = max(sents, key=lambda s: len(q & set(_tokenize(s))), default="")
    return best.strip()[:300]


# ── Extraction ────────────────────────────────────────────────────────────────
def _clean_quote(text: str) -> str:
    return re.sub(r"^[\s•\-–*\"“”]+", "", text or "").strip()


def extract_key_terms(chunk_texts: List[str], doc_type: str, use_llm: bool = True,
                      batch_size: int = 4) -> List[Dict]:
    """
    Returns one row per checklist term:
    {key, label, value, assessment, note, quote, method}  (method: "ai" or "keyword")
    Terms are extracted in small batches so each request stays well under free-tier token limits.
    """
    pack = DOC_TYPES.get(doc_type, DOC_TYPES["commercial"])
    terms = pack["terms"]
    hits = retrieve_for_terms(chunk_texts, terms)

    # Fallback rows (no LLM): best-matching sentence as evidence
    rows = []
    for t in terms:
        idx = hits.get(t["key"], [])
        quote = _clean_quote(_best_sentence(chunk_texts[idx[0]], t["query"])) if idx else ""
        rows.append({"key": t["key"], "label": t["label"],
                     "value": quote or "Not found in document",
                     "assessment": "standard" if quote else "not_found",
                     "note": "Keyword match only (AI extraction unavailable)." if quote else "",
                     "quote": quote, "method": "keyword"})
    if not use_llm:
        return rows

    from modules.llm_engine import call_llm, _parse_json
    by_key: Dict[str, Dict] = {}
    for start in range(0, len(terms), batch_size):
        batch = terms[start:start + batch_size]
        blocks = []
        for t in batch:
            excerpt = "\n...\n".join(chunk_texts[i][:550] for i in hits.get(t["key"], [])) or "(no matching text)"
            blocks.append(f"### {t['key']} — {t['label']}\n{excerpt}")
        prompt = f"""You are reviewing a {pack['label']} from the perspective of {pack['perspective']}.
For each term below, read ONLY its excerpts and extract what the document says.

{chr(10).join(blocks)}

Rules:
- value: the concrete answer in under 20 words (numbers, percentages, days, dollar amounts when present).
- assessment: "favorable", "standard", or "unfavorable" for {pack['perspective']}, or "not_found" if the excerpts don't cover it.
- note: one short sentence explaining the assessment.
- quote: a short exact phrase (under 25 words) copied from the excerpt that supports the value; "" if not_found.
Never invent numbers that are not in the excerpts.

Respond ONLY with JSON:
{{"terms": [{{"key": "...", "value": "...", "assessment": "...", "note": "...", "quote": "..."}}]}}"""
        try:
            data = _parse_json(call_llm(prompt, json_mode=True))
            for r in data.get("terms", []):
                if isinstance(r, dict) and str(r.get("key", "")).strip():
                    by_key[str(r["key"]).strip()] = r
        except Exception as e:
            print(f"Key-term extraction batch failed ({e}); keeping keyword evidence for {[t['key'] for t in batch]}.")

    merged = []
    for row in rows:
        r = by_key.get(row["key"])
        if r:
            a = str(r.get("assessment", "")).strip().lower().replace(" ", "_")
            merged.append({**row,
                           "value": str(r.get("value") or row["value"])[:200],
                           "assessment": a if a in ASSESSMENTS else "standard",
                           "note": str(r.get("note") or "")[:300],
                           "quote": _clean_quote(str(r.get("quote") or ""))[:300],
                           "method": "ai"})
        else:
            merged.append(row)
    return merged


def ai_coverage(rows: List[Dict]) -> float:
    """Share of rows extracted by the AI (the rest are keyword fallbacks)."""
    return sum(r.get("method") == "ai" for r in rows) / len(rows) if rows else 0.0


def favorability_score(rows: List[Dict]) -> Optional[int]:
    """0-100: share of found terms that are favorable (+1) / standard (0.5) / unfavorable (0)."""
    weights = {"favorable": 1.0, "standard": 0.5, "unfavorable": 0.0}
    found = [weights[r["assessment"]] for r in rows
             if r.get("assessment") in weights and r.get("method", "ai") != "keyword"]
    if not found:
        return None
    return round(100 * sum(found) / len(found))


# ── Comparison ────────────────────────────────────────────────────────────────
def compare_documents(name_a: str, rows_a: List[Dict], name_b: str, rows_b: List[Dict],
                      doc_type: str, use_llm: bool = True) -> Dict:
    """Return {better: 'A'|'B'|'similar'|'incomplete', summary, differences: [str], compared_terms}."""
    pack = DOC_TYPES.get(doc_type, DOC_TYPES["commercial"])
    # Only compare terms the AI extracted for BOTH documents; keyword fallbacks are evidence, not verdicts
    pairs = [(ra, rb) for ra, rb in zip(rows_a, rows_b)
             if ra.get("method", "ai") == "ai" and rb.get("method", "ai") == "ai"]
    if len(pairs) < max(2, len(rows_a) // 2):
        return {"better": "incomplete",
                "summary": "AI extraction didn't finish for enough terms in both documents to judge fairly. "
                           "The table shows what was found; run the comparison again in a minute.",
                "differences": [], "method": "incomplete", "compared_terms": len(pairs)}
    rows_a = [p[0] for p in pairs]
    rows_b = [p[1] for p in pairs]
    sa, sb = favorability_score(rows_a), favorability_score(rows_b)
    if sa is None or sb is None or abs(sa - sb) < 8:
        better = "similar"
    else:
        better = "A" if sa > sb else "B"
    fallback = {
        "better": better,
        "summary": (f"{name_a if better == 'A' else name_b} scores better for {pack['perspective']}."
                    if better != "similar" else "The two documents score about the same."),
        "differences": [f"{ra['label']}: {ra['value']} vs {rb['value']}"
                        for ra, rb in zip(rows_a, rows_b) if ra["assessment"] != rb["assessment"]][:5],
        "method": "score", "compared_terms": len(pairs),
    }
    if not use_llm:
        return fallback
    table = "\n".join(f"- {ra['label']}: A = {ra['value']} ({ra['assessment']}); B = {rb['value']} ({rb['assessment']})"
                      for ra, rb in zip(rows_a, rows_b))
    prompt = f"""Two {pack['label']}s were compared term by term from the perspective of {pack['perspective']}.
A = {name_a}
B = {name_b}
{table}

Which is better for {pack['perspective']}? Respond ONLY with JSON:
{{"better": "A" or "B" or "similar", "summary": "<2 sentences>", "differences": ["<up to 5 short bullets on the differences that matter most>"]}}"""
    try:
        from modules.llm_engine import call_llm, _parse_json
        data = _parse_json(call_llm(prompt, json_mode=True))
        b = str(data.get("better", "")).strip().upper()
        return {"better": b if b in ("A", "B") else "similar",
                "summary": str(data.get("summary") or fallback["summary"]),
                "differences": [str(d) for d in (data.get("differences") or fallback["differences"])][:5],
                "method": "llm", "compared_terms": len(pairs)}
    except Exception as e:
        print(f"Comparison via LLM failed ({e}); using score-based verdict.")
        return fallback
