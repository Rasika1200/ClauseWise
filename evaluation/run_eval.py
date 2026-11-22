"""
Reproducible evaluation of ClauseWise on the CUAD v1 benchmark.

Setup (one time):
    mkdir -p evaluation/cuad
    curl -L -o evaluation/cuad/data.zip https://github.com/TheAtticusProject/cuad/raw/main/data.zip
    unzip evaluation/cuad/data.zip -d evaluation/cuad        # gives CUADv1.json

Run:
    CLAUSEWISE_NEURAL=1 python evaluation/run_eval.py                 # all three parts
    CLAUSEWISE_NEURAL=1 python evaluation/run_eval.py --parts detect  # no API key needed

Parts (each runs the same code path the app uses):
  detect   Clause-type detection: modules.clause_matcher.match_chunks over every chunk of a
           contract, scored against CUAD labels for the taxonomy types CUAD can judge. No LLM.
  extract  LLM key-clause extraction: modules.llm_engine.extract_clauses on the first 8 chunks
           (as app/main_app.py does). Scores quote fidelity (is extracted_text really in the
           contract?) and clause-type precision against CUAD.
  qa       RAG Q&A: index contract -> rag_retrieve_and_assemble -> answer_question, for a balanced
           set of questions whose CUAD clause is present / absent. Scores retrieval recall of the
           gold span, false "clause exists" claims on absent clauses, and missed answers.

Every LLM result is cached in evaluation/eval_runs/ so an interrupted run (rate limits) resumes.
Results are written to evaluation/eval_results.json.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

CUAD_PATH = ROOT / "evaluation" / "cuad" / "CUADv1.json"
CACHE_DIR = ROOT / "evaluation" / "eval_runs"
OUT_PATH = ROOT / "evaluation" / "eval_results.json"

# ClauseWise taxonomy type -> CUAD categories that mark it present.
# Only types CUAD annotates directly are scored. Left out on purpose:
#   termination (CUAD labels only sub-types such as "Termination For Convenience", but nearly
#   every contract has *some* termination clause), and confidentiality, payment terms,
#   indemnification, force majeure, dispute resolution, generic liability (not in CUAD).
TAXONOMY_TO_CUAD = {
    "governing law": ["Governing Law"],
    "assignment": ["Anti-Assignment"],
    "limitation of liability": ["Cap On Liability"],
    "non-compete / non-solicitation": ["Non-Compete", "No-Solicit Of Employees", "No-Solicit Of Customers"],
    "intellectual property": ["Ip Ownership Assignment", "Joint Ip Ownership", "License Grant"],
    "warranty": ["Warranty Duration"],
}

# Free-text clause_type from the LLM -> taxonomy type (first match wins).
LLM_TYPE_RULES = [
    ("limitation of liability", ["limitation of liability", "liability cap", "cap on liability", "limitation on liability"]),
    ("governing law", ["governing law", "choice of law", "jurisdiction"]),
    ("assignment", ["assignment", "assign"]),
    ("non-compete / non-solicitation", ["non-compete", "noncompete", "non-solicit", "nonsolicit", "restrictive covenant", "competition"]),
    ("intellectual property", ["intellectual property", "ip ", "license", "licence", "ownership of"]),
    ("warranty", ["warrant"]),
]

# CUAD category -> the question a user would type in the app.
QA_QUESTIONS = {
    "Governing Law": "Which state's or country's law governs this contract?",
    "Anti-Assignment": "Can either party assign or transfer this contract to someone else without the other party's consent?",
    "Cap On Liability": "Is there a cap or limit on either party's liability under this contract?",
    "Non-Compete": "Does this contract restrict either party from competing with the other?",
    "Termination For Convenience": "Can a party terminate this contract without cause, just by giving notice?",
    "Exclusivity": "Does this contract give either party exclusive rights?",
    "Audit Rights": "Does either party have the right to audit the other's books or records?",
    "Insurance": "Is either party required to carry insurance?",
    "Change Of Control": "What happens under this contract if a party is acquired or undergoes a change of control?",
    "Liquidated Damages": "Does this contract set liquidated damages for any breach?",
}

FALLBACK_MARKER = "Analysis completed via local extraction engine"


# ── helpers ──────────────────────────────────────────────────────────────────
def load_cuad():
    data = json.loads(CUAD_PATH.read_text())["data"]
    contracts = []
    for item in data:
        para = item["paragraphs"][0]
        cats = {}
        for qa in para["qas"]:
            cat = qa["id"].split("__")[-1]
            cats[cat] = [a["text"] for a in qa["answers"]] if not qa.get("is_impossible") else []
        contracts.append({"id": item["title"], "text": para["context"], "cats": cats})
    return contracts


def gold_types(contract):
    return {t for t, cats in TAXONOMY_TO_CUAD.items() if any(contract["cats"].get(c) for c in cats)}


def norm(s):
    s = s.lower().replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def quote_coverage(quote, text_norm):
    """Fraction of the quote covered by its longest contiguous match in the contract (1.0 = verbatim)."""
    q = norm(quote)
    if not q:
        return 0.0
    if q in text_norm:
        return 1.0
    # longest common block, then check a window around it for near-verbatim matches
    m = SequenceMatcher(None, text_norm, q, autojunk=False).find_longest_match(0, len(text_norm), 0, len(q))
    start = max(0, m.a - m.b - 20)
    window = text_norm[start:start + len(q) + 40]
    blocks = SequenceMatcher(None, window, q, autojunk=False).get_matching_blocks()
    return round(sum(b.size for b in blocks) / len(q), 4)


def chunk_contract(contract):
    from modules.doc_processor import process_raw_text
    return process_raw_text(contract["text"], source_file=contract["id"], chunk_size=500, overlap=80, strategy="semantic")


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


class Cache:
    def __init__(self, name):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CACHE_DIR / f"{name}.jsonl"
        self.rows = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[row["key"]] = row

    def get(self, key):
        return self.rows.get(key)

    def put(self, row):
        self.rows[row["key"]] = row
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def llm_failed():
    from modules import llm_engine
    return bool(llm_engine.LAST_LLM_ERROR)


def reset_llm_error():
    from modules import llm_engine
    llm_engine.LAST_LLM_ERROR = ""


# ── Part 1: clause-type detection (no LLM) ───────────────────────────────────
def run_detect(contracts, threshold):
    from modules.clause_matcher import match_chunks
    from modules.embedder import embedding_backend

    per_type = {t: [0, 0, 0] for t in TAXONOMY_TO_CUAD}  # tp, fp, fn
    for i, c in enumerate(contracts, 1):
        chunks = chunk_contract(c)
        results = match_chunks([ch.text for ch in chunks], threshold=threshold)
        predicted = {r["matched_clause"] for r in results}
        gold = gold_types(c)
        for t in TAXONOMY_TO_CUAD:
            if t in predicted and t in gold:
                per_type[t][0] += 1
            elif t in predicted:
                per_type[t][1] += 1
            elif t in gold:
                per_type[t][2] += 1
        print(f"[detect {i}/{len(contracts)}] {len(chunks)} chunks", flush=True)

    by_type = {t: prf(*v) for t, v in per_type.items()}
    micro = prf(*(sum(v[k] for v in per_type.values()) for k in range(3)))
    macro = {m: round(sum(s[m] for s in by_type.values()) / len(by_type), 4) for m in ("precision", "recall", "f1")}
    return {
        "n_contracts": len(contracts),
        "embedding_backend": embedding_backend(),
        "match_threshold": threshold,
        "micro": micro,
        "macro": macro,
        "per_type": by_type,
        "gold_prevalence": {t: sum(t in gold_types(c) for c in contracts) for t in TAXONOMY_TO_CUAD},
    }


# ── Part 2: LLM key-clause extraction ────────────────────────────────────────
def run_extract(contracts, sleep):
    from modules.llm_engine import extract_clauses

    cache = Cache("extract")
    for i, c in enumerate(contracts, 1):
        if cache.get(c["id"]):
            continue
        chunks = chunk_contract(c)
        reset_llm_error()
        t0 = time.time()
        analysis = extract_clauses(" ".join(ch.text for ch in chunks[:8]), contract_name=c["id"])
        latency = time.time() - t0
        # Either the heuristic engine or the Ollama fallback answered instead of Groq
        fell_back = analysis.executive_summary.startswith(FALLBACK_MARKER) or llm_failed()
        if fell_back:
            # Don't cache: the LLM didn't answer (rate limit / error). Retry on the next run.
            from modules import llm_engine
            print(f"[extract {i}] LLM unavailable, skipped: {llm_engine.LAST_LLM_ERROR[:200]}", flush=True)
            time.sleep(sleep)
            continue
        cache.put({
            "key": c["id"],
            "latency_s": round(latency, 2),
            "clauses": [{"clause_type": cl.clause_type, "extracted_text": cl.extracted_text} for cl in analysis.clauses],
        })
        print(f"[extract {i}/{len(contracts)}] {len(analysis.clauses)} clauses, {latency:.1f}s", flush=True)
        time.sleep(sleep)

    by_id = {c["id"]: c for c in contracts}
    rows = [r for k, r in cache.rows.items() if k in by_id]
    n_quotes = exact = near = 0
    typed_tp = typed_fp = untyped = 0
    latencies = []
    for r in rows:
        c = by_id[r["key"]]
        text_norm = norm(c["text"])
        gold = gold_types(c)
        latencies.append(r["latency_s"])
        for cl in r["clauses"]:
            n_quotes += 1
            cov = quote_coverage(cl["extracted_text"], text_norm)
            exact += cov == 1.0
            near += cov >= 0.9
            ctype = cl["clause_type"].lower()
            mapped = next((t for t, kws in LLM_TYPE_RULES if any(k in ctype for k in kws)), None)
            if mapped is None:
                untyped += 1
            elif mapped in gold:
                typed_tp += 1
            else:
                typed_fp += 1
    latencies.sort()
    return {
        "n_contracts": len(rows),
        "n_clauses": n_quotes,
        "quote_exact_rate": round(exact / n_quotes, 4) if n_quotes else None,
        "quote_near_exact_rate_90": round(near / n_quotes, 4) if n_quotes else None,
        "type_precision_on_cuad_judgeable": round(typed_tp / (typed_tp + typed_fp), 4) if typed_tp + typed_fp else None,
        "n_type_judgeable": typed_tp + typed_fp,
        "n_type_not_judgeable": untyped,
        "latency_s_median": latencies[len(latencies) // 2] if latencies else None,
        "latency_s_p90": latencies[int(len(latencies) * 0.9)] if latencies else None,
    }


# ── Part 3: RAG Q&A ──────────────────────────────────────────────────────────
JUDGE_PROMPT = """You are grading an AI assistant's answer about a contract.
Question: {q}
Answer: {a}

Does the answer state or clearly imply that the contract DOES contain a provision on this topic?
- true: it describes such a provision (e.g. names a governing law, describes an assignment restriction, states a cap).
- false: it says the document does not specify / does not mention it, or that no such provision exists.
Return JSON only: {{"asserts_present": true or false}}"""


def build_qa_items(contracts, n_per_class, seed):
    rng = random.Random(seed)
    present, absent = [], []
    for c in contracts:
        for cat in QA_QUESTIONS:
            if cat not in c["cats"]:
                continue
            (present if c["cats"][cat] else absent).append((c["id"], cat))
    rng.shuffle(present)
    rng.shuffle(absent)
    return present[:n_per_class] + absent[:n_per_class]


def run_qa(contracts, n_per_class, seed, sleep):
    from modules.embedder import index_chunks
    from modules.clause_matcher import match_chunks
    from modules.rag_pipeline import rag_retrieve_and_assemble
    from modules.llm_engine import answer_question, call_llm, _parse_json

    by_id = {c["id"]: c for c in contracts}
    items = build_qa_items(contracts, n_per_class, seed)
    items.sort(key=lambda x: x[0])  # group by contract so each one is indexed once
    cache = Cache("qa")
    indexed = None

    for i, (cid, cat) in enumerate(items, 1):
        key = f"{cid}__{cat}"
        if cache.get(key):
            continue
        c = by_id[cid]
        if indexed != cid:
            chunks = chunk_contract(c)
            index_chunks(chunks, topic_labels=match_chunks([ch.text for ch in chunks]), clear_first=True)
            indexed = cid
        question = QA_QUESTIONS[cat]
        reset_llm_error()
        t0 = time.time()
        context, used = rag_retrieve_and_assemble(query=question, top_k=8, filter_source_file=cid, rerank_results=True)
        ans = answer_question(question, context, chat_history=[], chunks_used=used)
        latency = time.time() - t0
        if llm_failed():
            from modules import llm_engine
            print(f"[qa {i}] LLM unavailable, skipped: {llm_engine.LAST_LLM_ERROR[:200]}", flush=True)
            time.sleep(sleep)
            continue
        time.sleep(sleep)
        reset_llm_error()
        try:
            verdict = _parse_json(call_llm(JUDGE_PROMPT.format(q=question, a=ans.answer), json_mode=True))
            asserts_present = bool(verdict.get("asserts_present"))
        except Exception as e:
            print(f"[qa {i}] judge failed, skipped: {e}", flush=True)
            time.sleep(sleep)
            continue
        gold_spans = c["cats"][cat]
        ctx_norm = norm(context)
        span_retrieved = any(norm(s) and norm(s) in ctx_norm for s in gold_spans) if gold_spans else None
        cache.put({
            "key": key, "category": cat, "gold_present": bool(gold_spans), "span_retrieved": span_retrieved,
            "asserts_present": asserts_present, "answer": ans.answer, "latency_s": round(latency, 2),
        })
        print(f"[qa {i}/{len(items)}] {cat}: gold={bool(gold_spans)} model={asserts_present}", flush=True)
        time.sleep(sleep)

    keys = {f"{cid}__{cat}" for cid, cat in items}
    rows = [r for k, r in cache.rows.items() if k in keys]
    pos = [r for r in rows if r["gold_present"]]
    neg = [r for r in rows if not r["gold_present"]]
    tp = sum(r["asserts_present"] for r in pos)
    fp = sum(r["asserts_present"] for r in neg)
    latencies = sorted(r["latency_s"] for r in rows)
    per_cat = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in rows:
        per_cat[r["category"]]["n"] += 1
        per_cat[r["category"]]["correct"] += r["asserts_present"] == r["gold_present"]
    return {
        "n_questions": len(rows),
        "n_clause_present": len(pos),
        "n_clause_absent": len(neg),
        "accuracy": round((tp + len(neg) - fp) / len(rows), 4) if rows else None,
        "false_claim_rate_on_absent": round(fp / len(neg), 4) if neg else None,
        "miss_rate_on_present": round(1 - tp / len(pos), 4) if pos else None,
        "gold_span_retrieved_rate": round(sum(bool(r["span_retrieved"]) for r in pos) / len(pos), 4) if pos else None,
        "latency_s_median": latencies[len(latencies) // 2] if latencies else None,
        "per_category": dict(per_cat),
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="detect,extract,qa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-detect", type=int, default=100)
    ap.add_argument("--n-extract", type=int, default=25)
    ap.add_argument("--n-qa-per-class", type=int, default=30, help="questions where the clause is present, and the same number where it is absent")
    ap.add_argument("--threshold", type=float, default=0.25, help="match_chunks threshold (app default 0.25)")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between LLM calls")
    args = ap.parse_args()
    parts = set(args.parts.split(","))

    if not CUAD_PATH.exists():
        sys.exit(f"Missing {CUAD_PATH}. See the setup steps at the top of this file.")
    if parts & {"extract", "qa"} and not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set (add it to .env). Run with --parts detect to skip LLM parts.")

    contracts = load_cuad()
    rng = random.Random(args.seed)
    order = list(range(len(contracts)))
    rng.shuffle(order)
    sample = [contracts[i] for i in order]

    results = {}
    results["meta"] = {
        "dataset": "CUAD v1 (510 contracts), github.com/TheAtticusProject/cuad",
        "seed": args.seed,
        "llm": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "neural_models": os.getenv("CLAUSEWISE_NEURAL"),
        "date": time.strftime("%Y-%m-%d"),
    }
    if "detect" in parts:
        results["detect"] = run_detect(sample[: args.n_detect], args.threshold)
    if "extract" in parts:
        results["extract"] = run_extract(sample[: args.n_extract], args.sleep)
    if "qa" in parts:
        results["qa"] = run_qa(sample, args.n_qa_per_class, args.seed, args.sleep)

    # Merge with parts written by other runs (re-read now, in case one finished meanwhile)
    merged = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}
    merged.update(results)
    merged.setdefault("meta_by_part", {}).update({p: results["meta"] for p in results if p != "meta"})
    results = merged
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "meta"}, indent=2, default=str)[:4000])
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
