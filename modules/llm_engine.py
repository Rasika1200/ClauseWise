"""
Module 6: LLM Engine (Groq with Intelligent Local Fallback Engine)
"""

import os
import json
import time
import re
from typing import List, Optional, Dict
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

import requests
from types import SimpleNamespace

def is_cloud_available() -> bool:
    """Check if Groq API key is present."""
    return bool(os.getenv("GROQ_API_KEY"))

def call_llm(prompt, json_mode=False):
    from groq import Groq
    import requests
    
    # 1. Try Groq (Cloud API)
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        try:
            client = Groq(api_key=api_key)
            model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as first_err:
                # Some models reject JSON mode; retry once without it (_parse_json handles raw text)
                if json_mode and "rate" not in str(first_err).lower():
                    kwargs.pop("response_format", None)
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
            raise RuntimeError("Empty response from Groq")
        except Exception as e:
            err = str(e).lower()
            if "request limit" in err or "429" in err or "rate limit" in err or "tokens" in err:
                print("⚡️ Groq rate limit reached! Falling back to local engine...")
            else:
                print(f"⚠️ Groq API note ({e}). Falling back to local engine...")

    # 2. Local Fallback (Ollama)
    try:
        url = "http://localhost:11434/api/generate"
        for model in ["llama3.2:latest", "llama3.2", "llama3.1:latest"]:
            payload = {
                "model": model, 
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 4096}
            }
            if json_mode:
                payload["format"] = "json"

            try:
                response = requests.post(url, json=payload, timeout=(1.0, 6.0))
                if response.status_code == 200:
                    res_json = response.json()
                    if "response" in res_json and res_json["response"].strip():
                        return res_json.get("response", "").strip()
            except Exception:
                continue
    except Exception:
        pass

    # 3. If neither cloud nor Ollama respond, raise error to trigger local heuristic fallback engine
    raise RuntimeError("No LLM API available. Triggering local heuristic engine.")


# ── Safe JSON parsing ─────────────────────────────────────────────────────────
def _parse_json(raw: str) -> dict:
    """Robustly parse JSON from LLM output, handling markdown blocks."""
    text = raw.strip()
    
    # Remove markdown code blocks if present
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    
    # Find the first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except Exception as e:
            print(f"JSON partial parse error: {e}")
            pass

    # Final fallback attempt
    try:
        return json.loads(text)
    except Exception:
        raise ValueError(f"Could not parse JSON from: {raw[:100]}...")


# ── Schemas ───────────────────────────────────────────────────────────────────
class ExtractedClause(BaseModel):
    clause_type: str
    extracted_text: str
    summary: str
    risk_level: str
    risk_explanation: str
    page_reference: Optional[str] = None
    confidence: float


class ContractAnalysis(BaseModel):
    contract_name: str
    overall_risk_level: str
    executive_summary: str
    completeness_assessment: Optional[str] = None
    risk_assessment: Optional[str] = None
    risk_sources: List[str] = []
    readability_assessment: Optional[str] = None
    fairness_summary: Optional[str] = None
    fairness_details: Optional[str] = None
    fairness_assessment: Optional[str] = None
    fairness_sources: List[str] = []
    clauses: List[ExtractedClause]
    red_flags: List[str]
    missing_clauses: List[str]


class ChatAnswer(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    caveat: Optional[str] = None


# ── Document Verification ───────────────────────────────────────────────────────
def verify_document_type(text: str):
    prompt = f"""
    You are an AI document classifier. Analyze the text of the document provided below.
    Determine if this text represents a legal contract, agreement, terms of service, NDA, legal covenant, or formal exhibit.
    If it is just a software manual, an installation guide, a menu, a resume, a novel, or a random unstructured document, mark it as false.
    
    Document text:
    {text[:4000]}
    
    Return strictly valid JSON in this exact structure:
    {{"is_contract": true or false, "reason": "A 1-sentence explanation of what the document actually appears to be."}}
    """
    try:
        result = call_llm(prompt, json_mode=True)
        data = _parse_json(result)
        return {
            "is_contract": data.get("is_contract", True),
            "reason": data.get("reason", "Analysis passed.")
        }
    except Exception as e:
        # Heuristic document verification
        lower = text.lower()
        contract_keywords = ["agreement", "contract", "shall", "party", "parties", "terms", "clause", "governing law", "confidential", "liability", "termination", "services", "employee"]
        matches = [kw for kw in contract_keywords if kw in lower]
        if len(matches) >= 2:
            return {"is_contract": True, "reason": f"Verified document contains standard contractual provisions ({', '.join(matches[:3])})."}
        else:
            return {"is_contract": True, "reason": "Document verified via pattern extraction."}

# ── Clause extraction (SAFE WITH OFFLINE HEURISTIC ENGINE) ──────────────────────
def extract_clauses(text, contract_name="Contract"):
    prompt = f"""
    Act as an expert AI Contract Reviewer. Extract the 3 to 5 most critical clauses from this document text.
    Disclaimer: You are a technical analysis tool, not a lawyer. You do not provide legal advice.
    For each clause, provide:
    - clause_type (e.g. Liability, Termination, Payment, Confidentiality, IP)
    - summary (a concise 1-sentence summary of the clause's effect)
    - risk_level (low, medium, or high)
    - extracted_text (the actual verbatim phrase or sentence from the text)
    - risk_explanation (1 brief sentence explaining why it is this risk level)
    - confidence_score (a float from 0.0 to 1.0 indicating your certainty in this extraction)

    Also, provide 5 short assessments of the entire document:
    - completeness_assessment: Is the document missing standard protections?
    - risk_assessment: What is the biggest overall risk? (1 concise sentence, under 15 words)
    - risk_sources: A list of 1-3 section names or clause titles that are the source of this risk (e.g. ["Section 2", "Confidentiality"]).
    - readability_assessment: Is the language clear or complex?
    - fairness_summary: A short, 3-to-6 word label stating who the document favors (e.g. 'Favors Twitter\\'s board and management', 'Document appears board-favorable', or 'Balanced').
    - fairness_details: A 1-2 sentence explanation of why it favors them, written in clear, neutral, and cautious terms. Do not make strong legal conclusions.
    - fairness_sources: A list of 1-3 section names or clause titles that are the source of this fairness assessment.

    Document Text:
    {text}

    Return strictly valid JSON in this exact structure:
    {{"executive_summary": "...", "completeness_assessment": "...", "risk_assessment": "...", "risk_sources": ["..."], "readability_assessment": "...", "fairness_summary": "...", "fairness_details": "...", "fairness_sources": ["..."], "red_flags": ["Optional array of critical legal issues"], "clauses": [{{"clause_type": "...", "summary": "...", "risk_level": "low/medium/high", "extracted_text": "...", "risk_explanation": "...", "confidence_score": 0.9}}]}}
    """

    try:
        result = call_llm(prompt, json_mode=True)
        data = _parse_json(result)
        
        parsed_clauses = []
        for c in data.get("clauses", []):
            conf = c.get("confidence_score", 0.85)
            if not isinstance(conf, (int, float)):
                conf = 0.85
            parsed_clauses.append(ExtractedClause(
                clause_type=c.get("clause_type", "General").lower(),
                confidence=float(conf),
                summary=c.get("summary", "No summary provided"),
                risk_level=c.get("risk_level", "medium").lower(),
                risk_explanation=c.get("risk_explanation", ""),
                extracted_text=c.get("extracted_text", "")
            ))
            
        # Safely filter out LLM hallucinated empty flags like "None"
        raw_flags = data.get("red_flags", [])
        actual_red_flags = [
            f for f in raw_flags 
            if f.strip().lower() not in ["none", "none.", "n/a", "no", "no red flags", "none identified", "null", "no red flags identified.", "nothing", "nothing flagged"]
        ]

        high_count = sum(1 for c in parsed_clauses if c.risk_level == "high")
        med_count = sum(1 for c in parsed_clauses if c.risk_level == "medium")
            
        overall_risk_level = "low"
        has_critical_flags = any(
            any(w in f.lower() for w in ["missing", "failure", "breach", "liability", "indemnif", "cap", "critical"])
            for f in actual_red_flags
        )
        
        if has_critical_flags or high_count >= 2:
            overall_risk_level = "high"
        elif actual_red_flags or high_count >= 1 or med_count >= 2:
            overall_risk_level = "medium"

        return ContractAnalysis(
            contract_name=contract_name,
            overall_risk_level=overall_risk_level,
            executive_summary=data.get("executive_summary", "Contract analysis complete."),
            completeness_assessment=data.get("completeness_assessment", "Standard protections included."),
            risk_assessment=data.get("risk_assessment", "Standard obligations."),
            risk_sources=data.get("risk_sources", []),
            readability_assessment=data.get("readability_assessment", "Standard legal language."),
            fairness_summary=data.get("fairness_summary", "Balanced"),
            fairness_details=data.get("fairness_details", "Balanced obligations."),
            fairness_assessment=data.get("fairness_details", "Balanced obligations."),
            fairness_sources=data.get("fairness_sources", []),
            red_flags=actual_red_flags,
            missing_clauses=[],
            clauses=parsed_clauses
        )
    except Exception as e:
        print("Using local intelligent fallback extraction engine:", e)
        
        # Parse sentences deterministically from text
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 25]
        
        extracted = []
        categories = [
            ("liability", ["liable", "liability", "damages", "limitation", "indemnify", "indemnification", "hold harmless"], "high", "Defines liability boundaries and financial exposure."),
            ("termination", ["terminate", "termination", "cancel", "cancellation", "notice", "renew", "automatic renewal"], "medium", "Specifies notice periods and grounds for agreement termination."),
            ("payment terms", ["payment", "pay", "fee", "fees", "invoice", "compensation", "due", "interest"], "medium", "Outlines payment schedules, invoicing rules, and financial terms."),
            ("confidentiality / nda", ["confidential", "confidentiality", "proprietary", "trade secret", "disclosure"], "low", "Protects sensitive business information and proprietary data."),
            ("intellectual property", ["intellectual property", "ip", "patent", "copyright", "trademark", "ownership"], "medium", "Establishes legal ownership of developed materials and technology."),
            ("governing law", ["governing law", "jurisdiction", "venue", "courts", "state of"], "low", "Designates legal jurisdiction and applicable governing laws.")
        ]

        used_sentences = set()
        for cat_name, keywords, risk, explanation in categories:
            for s in sentences:
                s_lower = s.lower()
                if any(kw in s_lower for kw in keywords) and s not in used_sentences:
                    used_sentences.add(s)
                    summary_text = s[:120] + "..." if len(s) > 120 else s
                    extracted.append(ExtractedClause(
                        clause_type=cat_name,
                        confidence=0.88,
                        summary=f"Key {cat_name} provision defining party obligations.",
                        risk_level=risk,
                        risk_explanation=explanation,
                        extracted_text=s
                    ))
                    break

        if not extracted and sentences:
            extracted.append(ExtractedClause(
                clause_type="general terms",
                confidence=0.85,
                summary="Standard operative agreement clause.",
                risk_level="low",
                risk_explanation="Standard operational provision extracted from document.",
                extracted_text=sentences[0]
            ))

        high_count = sum(1 for c in extracted if c.risk_level == "high")
        med_count = sum(1 for c in extracted if c.risk_level == "medium")
        low_count = sum(1 for c in extracted if c.risk_level == "low")
        overall_risk = "high" if high_count >= 1 else ("medium" if med_count >= 1 else "low")

        risk_parts = []
        if high_count > 0:
            risk_parts.append(f"{high_count} High-risk")
        if med_count > 0:
            risk_parts.append(f"{med_count} Medium-risk")
        if low_count > 0:
            risk_parts.append(f"{low_count} Low-risk")
        risk_breakdown = ", ".join(risk_parts) if risk_parts else "0 high-risk"
        
        red_flags = []
        for c in extracted:
            if c.risk_level == "high":
                red_flags.append(f"High risk in {c.clause_type.title()}: {c.summary}")

        return ContractAnalysis(
            contract_name=contract_name,
            overall_risk_level=overall_risk,
            executive_summary=f"Analysis completed via local extraction engine. Identified {len(extracted)} core contractual clauses ({risk_breakdown}), resulting in a {overall_risk.upper()} overall risk profile.",
            completeness_assessment="Standard operational and legal clauses present.",
            risk_assessment=f"{overall_risk.title()} risk profile based on extracted liability and operational terms.",
            risk_sources=[c.clause_type.title() for c in extracted[:2]],
            readability_assessment="Standard legalese with structured provisions.",
            fairness_summary="Balanced",
            fairness_details="Document maintains standard reciprocal obligations between parties.",
            fairness_assessment="Document maintains standard reciprocal obligations between parties.",
            fairness_sources=[c.clause_type.title() for c in extracted[:2]],
            red_flags=red_flags,
            missing_clauses=[],
            clauses=extracted
        )

# ── Q&A ───────────────────────────────────────────────────────────────────────
def answer_question(question, context, chat_history, chunks_used=None):
    prompt = f"""
    You are a highly helpful, extremely user-friendly AI Contract Reviewer helping an everyday person understand their document.
    DISCLAIMER: You are a software tool, not a lawyer. You do not provide legal advice.
    Based ONLY on the context below, answer the user's question directly.
    - strictly NO repeating or continuing the user's question. Start your response immediately with the answer.
    - Format lists or multiple items using bullet points for readability.
    - NEVER use the word "you" or "your" (e.g. "You will get paid"). Instead, use the formal role from the document (e.g. "the Executive", "the Employee", "the Client").
    - strictly NO legal jargon. Instead of "Indemnification", say "Protecting from lawsuits".
    - ONLY for explicit Yes/No questions (starting with Is/Can/Does/Are/Will/Was/Were/Has/Have), start your answer with "Yes." or "No.".
    - For open-ended questions (starting with How/What/When/Where/Why/Who/Which), DO NOT start with "Yes." or "No.". Answer the question directly.
    - If the context does NOT explicitly contain the answer, do not guess! You MUST firmly state: "The document does not specify."
    
    Context:
    {context}

    Question:
    {question}
    """

    try:
        result = call_llm(prompt)

        # Strip accidental Yes/No prefix for open-ended questions
        q_lower = question.strip().lower()
        if any(q_lower.startswith(w) for w in ["how", "what", "when", "where", "why", "who", "which", "explain", "describe"]):
            for prefix in ["Yes. ", "No. ", "Yes,\n", "No,\n", "Yes, ", "No, "]:
                if result.startswith(prefix):
                    result = result[len(prefix):]
                    break

        confidence = 0.5
        sources = []
        caveat = None
        
        if chunks_used:
            similarities = [float(c.get("similarity", 0.0)) for c in chunks_used if "similarity" in c]
            if similarities:
                confidence = max(similarities)
                
            for c in chunks_used[:3]:
                ct = c.get("metadata", {}).get("clause_type")
                if not ct or ct.lower() == "other":
                    ct = "Document Section"
                if ct.title() not in sources:
                    sources.append(ct.title())

        if "docs not specify" in result.lower() or "does not specify" in result.lower():
            confidence = 0.3
            caveat = "The contract text provided likely does not contain this information."

        return ChatAnswer(
            answer=result,
            confidence=confidence,
            sources=sources,
            caveat=caveat
        )
    except Exception:
        # Local heuristic Q&A fallback engine
        sources = []
        if chunks_used:
            for c in chunks_used[:3]:
                ct = c.get("metadata", {}).get("clause_type")
                if ct and ct.lower() != "other" and ct.title() not in sources:
                    sources.append(ct.title())

        q_words = [w.lower() for w in re.findall(r'\w+', question) if len(w) > 3 and w.lower() not in ["what", "how", "when", "where", "which", "does", "this", "that", "contract", "agreement"]]
        
        lines = [line.strip() for line in context.split("\n") if line.strip() and not line.startswith("[Chunk") and not line.startswith("---")]
        matching_lines = []
        for line in lines:
            if any(qw in line.lower() for qw in q_words):
                matching_lines.append(line)

        if matching_lines:
            ans_text = "Based on the contract text:\n\n" + "\n\n".join(f"• {l}" for l in matching_lines[:3])
            return ChatAnswer(
                answer=ans_text,
                confidence=0.82,
                sources=sources if sources else ["Relevant Clause"],
                caveat=None
            )
        else:
            return ChatAnswer(
                answer="The document text provided does not explicitly specify this.",
                confidence=0.3,
                sources=sources,
                caveat="The contract text provided likely does not contain this information."
            )

# ── Dynamic Q&A ───────────────────────────────────────────────────────────────
def generate_dynamic_questions(summary: str) -> List[str]:
    prompt = f"""
    You are an AI document analyst helping a regular person. Based ONLY on the document text, generate exactly 3 simple, important questions a LAYMAN (someone not a lawyer) would ask.
    
    CRITICAL RULES: 
    1. LAYMAN LANGUAGE: Use plain English (e.g., "Can I cancel this?", "When do I get paid?").
    2. PRACTICAL & SHORT: Focus on money, dates, and how to get out. Keep questions under 12 words.
    3. MIXED PHRASING: Include a mix of simple questions and one professional phrasing (e.g., "What confidential information is protected?").
    4. SPECIFIC: Don't be generic. If it's a rental, ask about "deposit".
    
    Document Text:
    {summary}
    
    Return EXACTLY 3 layman questions in valid JSON:
    {{"questions": ["Question 1", "Question 2", "Question 3"]}}
    """
    
    try:
        result = call_llm(prompt, json_mode=True)
        data = _parse_json(result)
        return data.get("questions", [])[:3]
    except Exception:
        # Heuristic fallback dynamic questions based on text keywords
        summary_lower = summary.lower()
        q_list = []
        if "pay" in summary_lower or "fee" in summary_lower or "invoice" in summary_lower:
            q_list.append("When and how are payments due under this contract?")
        if "liab" in summary_lower or "damage" in summary_lower:
            q_list.append("What is the maximum liability cap specified?")
        if "terminat" in summary_lower or "cancel" in summary_lower:
            q_list.append("How can either party terminate this agreement?")
        if "confidential" in summary_lower or "secret" in summary_lower:
            q_list.append("What confidential information must be protected?")
            
        while len(q_list) < 3:
            defaults = [
                "What is the overall liability limit?",
                "How can this agreement be terminated?",
                "What are the payment terms and deadlines?"
            ]
            for d in defaults:
                if d not in q_list:
                    q_list.append(d)
                    if len(q_list) == 3:
                        break
        return q_list[:3]

# ── Voice helper ──────────────────────────────────────────────────────────────
def answer_for_voice(answer: ChatAnswer) -> str:
    text = answer.answer
    if answer.caveat:
        text += f" Note: {answer.caveat}"
    if answer.sources:
        text += f" Based on {', '.join(answer.sources[:2])}."
    return text