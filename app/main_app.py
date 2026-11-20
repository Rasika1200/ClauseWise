"""
AI Contract Review — Fixed UI
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys, json, datetime, sqlite3
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="ClauseWise Copilot",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

from modules.telemetry import TelemetryLogger
from modules import db_manager
from modules.llm_engine import is_cloud_available

@st.cache_resource
def get_telemetry_logger():
    return TelemetryLogger()

telemetry_logger = get_telemetry_logger()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* --- 1. CORE DISGUISE (HIDE STREAMLIT BRANDING) --- */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Base typography to custom modern font */
html, body, [class*="css"] {
    font-family: 'Outfit', -apple-system, sans-serif !important;
}

/* --- PRINT MODE CSS (CLEAN PDF EXPORTS) --- */
@media print {
    section[data-testid="stSidebar"], 
    header, 
    footer,
    iframe {
        display: none !important;
    }
    .stApp {
        background: #0f172a !important;
    }
    .block-container {
        max-width: 100% !important;
        padding-top: 10px !important;
        padding-left: 10px !important;
        padding-right: 10px !important;
    }
    .st-emotion-cache-1jicfl2, .st-emotion-cache-1104q3m { 
        width: 100% !important;
    }
}

/* --- 2. GLOBAL BACKGROUND (THE SAAS LOOK) --- */
.stApp {
    background: radial-gradient(125% 125% at 50% 0%, #0a0a0a 40%, #172554 100%) !important;
    color: #f3f4f6 !important;
}

/* --- 3. PREMIUM SIDEBAR --- */
section[data-testid="stSidebar"] {
    background: rgba(10, 15, 30, 0.6) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Sidebar navigation buttons */
div[data-testid="stSidebar"] .stButton > button {
    width: 100%; 
    text-align: left; 
    background: transparent;
    border: none; 
    padding: 14px 20px; 
    border-radius: 12px;
    font-size: 15px; 
    font-weight: 500; 
    margin-bottom: 8px;
    color: #9ca3af;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(59, 130, 246, 0.1); 
    color: #60a5fa !important;
    transform: translateX(6px);
    box-shadow: -4px 0px 0px 0px #3b82f6;
}
/* Active Sidebar Button */
div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(90deg, rgba(59, 130, 246, 0.2) 0%, transparent 100%);
    color: #ffffff !important;
    box-shadow: -4px 0px 0px 0px #60a5fa;
}

/* --- 4. GLASSMORPHISM METRIC CARDS --- */
[data-testid="stMetricValue"] {
    font-size: 32px !important;
    font-weight: 700 !important;
    background: linear-gradient(180deg, #ffffff 0%, #a1a1aa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
[data-testid="stMetricLabel"] {
    font-size: 13px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #9ca3af !important;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.8) 0%, rgba(10, 15, 30, 0.8) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.6);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}
[data-testid="stMetric"]:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 20px 40px -10px rgba(59, 130, 246, 0.3);
    border: 1px solid rgba(59, 130, 246, 0.4);
}
.dashboard-card {
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 24px;
    padding: 32px;
    height: 100%;
}

/* --- 5. NEON BUTTONS --- */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.3s ease !important;
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.5) !important;
}

/* --- 6. MODERN INPUT FIELDS --- */
.stTextArea textarea, .stTextInput input {
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background: rgba(15, 23, 42, 0.6) !important;
    color: #ffffff !important;
    padding: 16px !important;
    font-size: 15px !important;
    transition: all 0.2s ease !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #60a5fa !important;
    background: rgba(15, 23, 42, 0.9) !important;
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.15) !important;
}

/* --- 7. MASTERS-WORTHY CHAT INTERFACE --- */
.chat-container {
    padding: 20px 0;
    max-width: 900px;
    margin: 0 auto;
}
.user-msg {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px 20px 4px 20px;
    padding: 16px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    position: relative;
}
.bot-msg {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.4) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(96, 165, 250, 0.2);
    border-left: 5px solid #3b82f6;
    border-radius: 20px 20px 20px 4px;
    padding: 20px 28px;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.msg-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #60a5fa;
    margin-bottom: 8px;
    font-weight: 700;
}
.source-tag {
    background: rgba(59, 130, 246, 0.1);
    color: #93c5fd;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    margin-right: 6px;
    border: 1px solid rgba(59, 130, 246, 0.3);
}
.glass-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.4) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 35px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.4);
    margin-bottom: 25px;
}
.executive-header {
    background: linear-gradient(90deg, #3b82f6 0%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: -1px;
}
.metric-value-large {
    font-size: 3rem;
    font-weight: 500;
    color: #f8fafc;
    line-height: 1;
    letter-spacing: -1px;
}
.metric-label-small {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.audit-row {
    padding: 12px 16px; 
    border-radius: 12px;
    background: rgba(16, 185, 129, 0.05); 
    border: 1px solid rgba(16, 185, 129, 0.1);
    margin-bottom: 8px; 
    font-size: 14px;
    color: #6ee7b7;
    transition: all 0.2s;
}
.audit-row:hover {
    background: rgba(16, 185, 129, 0.1);
    transform: translateX(4px);
}

/* Clean up expanders */
.streamlit-expanderHeader, [data-testid="stExpander"] summary span {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #f1f5f9 !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
}

/* --- 8. HIGH-CONTRAST TYPOGRAPHY & WIDGET VISIBILITY --- */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-weight: 600 !important;
    letter-spacing: -0.03em;
}

p, span, label, div[data-testid="stMarkdownContainer"] p {
    color: #cbd5e1 !important;
}

[data-testid="stCaptionContainer"], .stCaption {
    color: #94a3b8 !important;
    font-size: 13px !important;
}

/* File Uploader Dark Dropzone */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 2px dashed rgba(59, 130, 246, 0.4) !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #cbd5e1 !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(30, 41, 59, 0.8) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
}

/* Alert / Notification Callout Styling */
.stAlert, [data-testid="stAlert"] {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
    border-radius: 12px !important;
}
.stAlert p, [data-testid="stAlert"] * {
    color: #93c5fd !important;
}

/* --- 9. SMOOTH PAGE TRANSITIONS --- */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.stApp {
    animation: fadeIn 0.5s ease-out;
}

hr {
    border-color: rgba(255, 255, 255, 0.05) !important;
}

/* --- 9. FIX COMPONENT IFRAME BACKGROUNDS --- */
iframe {
    background-color: transparent !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
_defaults = {
    "page": "upload", "contract_name": "", "chunks": [], "match_results": [],
    "topic_results": [], "contract_analysis": None, "indexed": False,
    "chat_history": [], "hitl_decisions": {}, "audit_log": [],
    "prefill_q": "", "auto_ask": False, "dynamic_questions": [],
    "rag_search_method": "Hybrid (RRF Fusion)",
    "rag_query_enhancement": "Query Expansion",
    "rag_rerank_enabled": True,
    "doc_type": "", "doc_type_conf": 0.0, "doc_type_for": "",
    "key_terms": None, "key_terms_for": "",
    "compare_result": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Key-term rendering helpers (document-type packs) ─────────────────────────
_ASSESS_STYLE = {
    "favorable":   ("#10b981", "Favorable"),
    "standard":    ("#94a3b8", "Standard"),
    "unfavorable": ("#ef4444", "Unfavorable"),
    "not_found":   ("#64748b", "Not found"),
}

def _badge(assessment: str) -> str:
    color, text = _ASSESS_STYLE.get(assessment, _ASSESS_STYLE["standard"])
    return (f"<span style='display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; "
            f"font-weight:700; color:{color}; border:1px solid {color}55; background:{color}1a;'>{text}</span>")

def _cell(row: dict) -> str:
    import html as _h
    val = _h.escape(row.get("value", ""))
    note = _h.escape(row.get("note", ""))
    quote = _h.escape(row.get("quote", ""))
    out = f"<div style='font-weight:600; color:#e5e7eb;'>{val}</div><div style='margin:4px 0;'>{_badge(row.get('assessment', 'standard'))}</div>"
    if note:
        out += f"<div style='font-size:12px; color:#9ca3af;'>{note}</div>"
    if quote:
        out += f"<div style='font-size:11px; color:#64748b; font-style:italic; margin-top:4px;'>“{quote}”</div>"
    return out

_TABLE_CSS = ("width:100%; border-collapse:collapse; font-size:14px;")
_TH = "text-align:left; padding:10px; color:#60a5fa; font-size:11px; letter-spacing:1px; border-bottom:1px solid rgba(255,255,255,0.1);"
_TD = "vertical-align:top; padding:10px; border-bottom:1px solid rgba(255,255,255,0.06);"

def render_key_terms(rows: list):
    import html as _h
    body = "".join(
        f"<tr><td style='{_TD} width:28%; color:#cbd5e1; font-weight:600;'>{_h.escape(r['label'])}</td><td style='{_TD}'>{_cell(r)}</td></tr>"
        for r in rows)
    st.markdown(f"<table style='{_TABLE_CSS}'><tr><th style='{_TH}'>TERM</th><th style='{_TH}'>WHAT THE DOCUMENT SAYS</th></tr>{body}</table>",
                unsafe_allow_html=True)

def render_comparison(name_a: str, rows_a: list, name_b: str, rows_b: list):
    import html as _h
    body = "".join(
        f"<tr><td style='{_TD} width:22%; color:#cbd5e1; font-weight:600;'>{_h.escape(ra['label'])}</td>"
        f"<td style='{_TD} width:39%;'>{_cell(ra)}</td><td style='{_TD} width:39%;'>{_cell(rb)}</td></tr>"
        for ra, rb in zip(rows_a, rows_b))
    st.markdown(f"<table style='{_TABLE_CSS}'><tr><th style='{_TH}'>TERM</th><th style='{_TH}'>A · {_h.escape(name_a)}</th>"
                f"<th style='{_TH}'>B · {_h.escape(name_b)}</th></tr>{body}</table>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 ClauseWise")
    st.caption("AI-powered document analysis")
    st.divider()
    if st.session_state.contract_name:
        st.success(f"📄 {st.session_state.contract_name[:28]}")
    else:
        st.info("No contract loaded")
    st.divider()
    for pid, label in [
        ("upload",    "📤  Upload Contract"),
        ("review",    "🔍  Review Contract"),
        ("qa",        "💬  Q&A"),
        ("dashboard", "📊  Dashboard"),
        ("compare",   "⚖️  Compare Documents"),
    ]:
        if st.button(label, key=f"nav_{pid}",
                     type="primary" if st.session_state.page == pid else "secondary",
                     use_container_width=True):
            st.session_state.page = pid
            st.rerun()
    
    st.divider()
    with st.expander("🛠️ Advanced RAG Config", expanded=False):
        st.markdown("<div style='font-size: 11px; color: #94a3b8; font-weight: 600; margin-bottom: 6px;'>RETRIEVAL STRATEGY</div>", unsafe_allow_html=True)
        search_method = st.selectbox(
            "Search Method",
            ["Dense (Semantic)", "Sparse (BM25)", "Hybrid (RRF Fusion)"],
            index=["Dense (Semantic)", "Sparse (BM25)", "Hybrid (RRF Fusion)"].index(
                st.session_state.get("rag_search_method", "Hybrid (RRF Fusion)")
            ),
            key="rag_search_method_ui",
            label_visibility="collapsed"
        )
        st.session_state.rag_search_method = search_method
        
        st.markdown("<div style='font-size: 11px; color: #94a3b8; font-weight: 600; margin-top: 10px; margin-bottom: 6px;'>QUERY ENHANCEMENT</div>", unsafe_allow_html=True)
        query_enh = st.selectbox(
            "Query Enhancement",
            ["None", "Query Expansion", "HyDE"],
            index=["None", "Query Expansion", "HyDE"].index(
                st.session_state.get("rag_query_enhancement", "Query Expansion")
            ),
            key="rag_query_enhancement_ui",
            label_visibility="collapsed"
        )
        st.session_state.rag_query_enhancement = query_enh

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        rerank_enabled = st.toggle(
            "Cross-Encoder Re-Ranker",
            value=st.session_state.get("rag_rerank_enabled", True),
            key="rag_rerank_enabled_ui"
        )
        st.session_state.rag_rerank_enabled = rerank_enabled
        try:
            from modules.embedder import embedding_backend, neural_models_enabled
            _emb = embedding_backend()
            _rr = "Cross-encoder (ms-marco-MiniLM)" if (rerank_enabled and neural_models_enabled() and _emb != "TF-IDF") else "Retrieval-score order"
            st.caption(f"Embeddings: {_emb} · Re-ranking: {_rr}")
        except Exception:
            pass

    st.divider()
    if is_cloud_available():
        st.markdown("<div style='font-size: 11px; color: #60a5fa; font-weight: 600; margin-bottom: 8px;'>⚡ Groq Cloud API Active</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size: 11px; color: #10b981; font-weight: 600; margin-bottom: 8px;'>🟢 Local Offline Engine Active</div>", unsafe_allow_html=True)
    if st.session_state.contract_name:
        if st.button("🗑 Clear & Start Over", use_container_width=True):
            for k, v in _defaults.items():
                st.session_state[k] = v
            try:
                from modules.embedder import clear_collection
                clear_collection()
            except ImportError:
                pass
            st.rerun()
    st.divider()
    st.caption("DATS 6501 Capstone | GWU")

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if page == "upload":
    st.markdown("""
        <div style='text-align: center; padding: 3rem 0 2rem 0;'>
            <h1 style='font-size: 3.5rem; color: #3b82f6; margin-bottom: 0.5rem;'>ClauseWise</h1>
            <p style='font-size: 1.25rem; color: #9ca3af; font-weight: 400;'>Instantly analyze, extract, and chat with your legal documents using advanced AI.</p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    if not is_cloud_available():
        st.info("💡 **Local Offline Engine Active:** Running zero-dependency mode. Add `GROQ_API_KEY` to `.env` anytime for optional cloud acceleration.")

    # ── Demo Mode: one click, no upload needed ────────────────────────────────
    DEMO_PDF = "data/demo_contracts/OpenAI-AMD.pdf"
    DEMO_JSON = "data/demo_contracts/openai_amd_prebaked.json"
    st.markdown(
        "<div style='text-align:center; margin: 0.5rem 0 0.75rem 0;'>"
        "<h3 style='margin-bottom:0.25rem;'>🚀 New here? Try the demo</h3>"
        "<p style='color:#94a3b8; margin:0;'>Loads a real AMD–OpenAI Registration Rights Agreement (Oct 2025) "
        "with its full risk review, then lets you ask it questions.</p></div>",
        unsafe_allow_html=True,
    )
    _d1, _d2, _d3 = st.columns([1, 2, 1])
    with _d2:
        if st.button("✨ Load Demo Contract", type="primary", use_container_width=True, key="load_demo"):
            from modules.doc_processor import process_contract
            from modules.clause_matcher import match_chunks
            from modules.embedder import index_chunks
            from modules.llm_engine import ContractAnalysis, ExtractedClause

            for k, v in _defaults.items():
                st.session_state[k] = v
            st.session_state.chat_history = []
            st.session_state.audit_log = []
            st.session_state.hitl_decisions = {}
            prog = st.progress(0)
            status = st.empty()
            try:
                status.info("📄 Step 1/3 — Reading the demo contract...")
                prog.progress(20)
                demo_chunks, _ = process_contract(DEMO_PDF, chunk_size=500, overlap=80, strategy="semantic")
                demo_name = os.path.basename(DEMO_PDF)
                for chunk in demo_chunks:
                    chunk.source_file = demo_name
                st.session_state.chunks = demo_chunks
                demo_texts = [c.text for c in demo_chunks]

                status.info("💾 Step 2/3 — Matching clauses and building the search index...")
                prog.progress(55)
                demo_matches = match_chunks(demo_texts)
                st.session_state.match_results = demo_matches
                index_chunks(demo_chunks, topic_labels=demo_matches)
                st.session_state.indexed = True

                status.info("🤖 Step 3/3 — Loading the saved risk review...")
                prog.progress(90)
                with open(DEMO_JSON, "r") as f:
                    demo_data = json.load(f)
                st.session_state.dynamic_questions = demo_data.pop("dynamic_questions", [])
                demo_data["clauses"] = [ExtractedClause(**c) for c in demo_data["clauses"]]
                st.session_state.contract_analysis = ContractAnalysis(**demo_data)
                st.session_state.contract_name = demo_name
                prog.progress(100)
                status.empty()
                prog.empty()
                st.session_state.page = "review"
                st.rerun()
            except Exception as e:
                status.empty()
                prog.empty()
                st.error(f"Couldn't load the demo contract: {e}")

    st.markdown("<p style='text-align:center; color:#64748b; margin: 1rem 0 0.25rem 0;'>— or analyze your own contract —</p>",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag and drop your contract here, or click Browse",
        type=["pdf", "docx"], help="Supported: PDF, Word (.docx)"
    )

    if uploaded_file:
        st.markdown(f"**File:** {uploaded_file.name}  |  **Size:** {uploaded_file.size // 1024} KB")
        if st.button("Analyze Contract", type="primary"):
            import tempfile
            from modules.doc_processor import process_contract
            from modules.topic_modeler import classify_chunks_llm
            from modules.clause_matcher import match_chunks
            from modules.embedder import index_chunks
            from modules.llm_engine import extract_clauses

            suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".docx"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            st.session_state.contract_name = uploaded_file.name
            st.session_state.chat_history = []  # Reset chat history for the new contract
            st.session_state.audit_log = []
            st.session_state.hitl_decisions = {}
            st.session_state.dynamic_questions = []  # Clear previous document's suggested queries
            prog = st.progress(0)
            status = st.empty()

            try:
                status.info("📄 Step 1/5 — Reading document...")
                prog.progress(15)
                chunks, doc_hash = process_contract(tmp_path, chunk_size=500, overlap=80, strategy="semantic")
                # Override temp filepath with the actual filename for proper RAG filtering
                for chunk in chunks:
                    chunk.source_file = uploaded_file.name
                st.session_state.chunks = chunks
                chunk_texts = [c.text for c in chunks]

                status.info("🛡️ Step 2/5 — Verifying document type...")
                prog.progress(35)
                from modules.llm_engine import verify_document_type
                verification = verify_document_type(" ".join(chunk_texts[:3]))
                if not verification.get("is_contract", True):
                    st.error(f"🛑 **Document Rejected:** This doesn't appear to be a contract. {verification.get('reason')}")
                    status.empty()
                    prog.empty()
                    st.stop()

                status.info("🔗 Step 3/5 — Matching clauses to legal taxonomy...")
                prog.progress(55)
                match_results = match_chunks(chunk_texts)
                st.session_state.match_results = match_results

                status.info("💾 Step 4/5 — Building search index...")
                prog.progress(75)
                index_chunks(chunks, topic_labels=match_results)
                st.session_state.indexed = True

                status.info("🤖 Step 5/5 — Running full analysis...")
                prog.progress(90)
                analysis = extract_clauses(" ".join(chunk_texts[:8]), contract_name=uploaded_file.name)
                st.session_state.contract_analysis = analysis
                prog.progress(100)
                status.empty()

                # Call db_manager.save_contract_analysis and save_extracted_clauses to persist the finalized analysis to SQLite
                db_manager.save_contract_analysis(
                    contract_name=uploaded_file.name,
                    contract_hash=doc_hash,
                    risk_profile=analysis.overall_risk_level,
                    summary=analysis.executive_summary,
                    assessment=analysis.risk_assessment
                )
                
                serialized_clauses = [
                    {
                        "clause_type": c.clause_type,
                        "summary": c.summary,
                        "risk_level": c.risk_level,
                        "risk_explanation": c.risk_explanation,
                        "confidence": c.confidence,
                        "extracted_text": c.extracted_text
                    }
                    for c in analysis.clauses
                ]
                db_manager.save_extracted_clauses(uploaded_file.name, serialized_clauses)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()
            finally:
                os.unlink(tmp_path)

            # ── FIX 1: navigate directly, no extra button needed ─────────────
            st.session_state.page = "review"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# REVIEW CONTRACT
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# REVIEW CONTRACT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "review":
    st.title("🔍 Review Contract")

    if not st.session_state.contract_analysis:
        st.info("Please upload a contract first.")
        if st.button("Go to Upload"):
            st.session_state.page = "upload"
            st.rerun()
        st.stop()

    analysis  = st.session_state.contract_analysis
    
    # Load existing decisions from SQLite on first load
    if not st.session_state.hitl_decisions and st.session_state.contract_name:
        existing = db_manager.get_all_hitl_decisions(st.session_state.contract_name)
        st.session_state.hitl_decisions = {
            d["clause_type"]: {"decision": d["decision"], "edited_text": d["edited_text"]}
            for d in existing
        }

    decisions = st.session_state.hitl_decisions
    clauses   = analysis.clauses
    risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(analysis.overall_risk_level.lower(), "⚪")
    reviewed = sum(1 for c in clauses if c.clause_type in decisions)

    # Routing Logic
    routing_departments = set(["Legal"])
    for c in clauses:
        text = (c.clause_type + " " + getattr(c, "summary", "")).lower()
        if any(w in text for w in ["data", "privacy", "security", "confidential", "sharing"]):
            routing_departments.add("Security")
        if any(w in text for w in ["payment", "fee", "liability", "indemnity", "financial", "cap"]):
            routing_departments.add("Finance")
    
    routing_html = "".join([f"<span class='source-tag' style='font-size:13px; font-weight:600;'>{dept}</span>" for dept in sorted(list(routing_departments))])

    st.markdown(f"**Contract:** {analysis.contract_name}")
    c1, c2 = st.columns([0.4, 0.6])
    
    risk_class = f"risk-{analysis.overall_risk_level.lower()}"
    with c1:
        st.markdown(f"**Overall Risk Profile**")
        st.markdown(f"<div class='risk-pill {risk_class}' style='font-size: 1.2rem; padding: 8px 12px; text-align: center; width: 100%; background: rgba(255,255,255,0.05); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);'>{risk_emoji} {analysis.overall_risk_level.upper()}</div>", unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"**Suggested Routing**")
        st.markdown(f"<div style='padding: 10px 0;'>{routing_html}</div>", unsafe_allow_html=True)

    st.divider()

    st.markdown(f"### 📄 Contract Summary")
    
    summary_text = getattr(analysis, 'executive_summary', 'No executive summary provided by model.')
    if "Analysis completed via local extraction engine" in summary_text and hasattr(analysis, "clauses") and analysis.clauses:
        h_cnt = sum(1 for c in analysis.clauses if c.risk_level == "high")
        m_cnt = sum(1 for c in analysis.clauses if c.risk_level == "medium")
        l_cnt = sum(1 for c in analysis.clauses if c.risk_level == "low")
        r_parts = []
        if h_cnt > 0: r_parts.append(f"{h_cnt} High-risk")
        if m_cnt > 0: r_parts.append(f"{m_cnt} Medium-risk")
        if l_cnt > 0: r_parts.append(f"{l_cnt} Low-risk")
        b_str = ", ".join(r_parts) if r_parts else "0 high-risk"
        summary_text = f"Analysis completed via local extraction engine. Identified {len(analysis.clauses)} core contractual clauses ({b_str}), resulting in a {analysis.overall_risk_level.upper()} overall risk profile."

    st.markdown(f"**TL;DR:** {summary_text}")
    st.divider()

    # ── Key terms for this document type ──────────────────────────────────────
    from modules.doc_types import DOC_TYPES, detect_document_type, extract_key_terms, favorability_score
    _cname = st.session_state.contract_name
    _texts = [c.text for c in st.session_state.chunks]
    _use_llm = is_cloud_available()
    if _texts and st.session_state.doc_type_for != _cname:
        with st.spinner("Identifying document type..."):
            _det = detect_document_type(" ".join(_texts[:12]), use_llm=_use_llm)
        st.session_state.doc_type = _det["type"]
        st.session_state.doc_type_conf = _det["confidence"]
        st.session_state.doc_type_for = _cname

    if _texts and st.session_state.doc_type:
        _keys = list(DOC_TYPES.keys())
        kt1, kt2 = st.columns([0.6, 0.4])
        with kt1:
            st.markdown("### 📑 Key Terms")
            st.caption("The checklist changes with the document type. Each value is pulled from the passages that cover that term.")
        with kt2:
            _chosen = st.selectbox(
                f"Document type (detected, {st.session_state.doc_type_conf:.0%} confidence)",
                _keys, index=_keys.index(st.session_state.doc_type),
                format_func=lambda k: DOC_TYPES[k]["label"], key=f"doc_type_select_{_cname}")
            if _chosen != st.session_state.doc_type:
                st.session_state.doc_type = _chosen
                st.rerun()

        _cache_key = f"{_cname}|{st.session_state.doc_type}"
        if st.session_state.key_terms_for != _cache_key:
            with st.spinner(f"Extracting key terms for a {DOC_TYPES[st.session_state.doc_type]['label']}..."):
                st.session_state.key_terms = extract_key_terms(_texts, st.session_state.doc_type, use_llm=_use_llm)
            st.session_state.key_terms_for = _cache_key

        _rows = st.session_state.key_terms or []
        _score = favorability_score(_rows)
        if _score is not None:
            st.markdown(f"**Favorability for {DOC_TYPES[st.session_state.doc_type]['perspective']}:** {_score}/100 "
                        f"· {sum(r['assessment']=='unfavorable' for r in _rows)} unfavorable · "
                        f"{sum(r['assessment']=='not_found' for r in _rows)} not found")
        render_key_terms(_rows)
        st.divider()

    # Add Red Flags and Missing Clauses to Review page for better visibility
    if analysis.red_flags:
        st.subheader("⚠️ Critical Red Flags")
        for f in analysis.red_flags:
            st.markdown(f"<div class='flag-card'>🚩 {f}</div>", unsafe_allow_html=True)
        st.divider()

    if analysis.missing_clauses:
        st.subheader("🔴 Missing Standard Clauses")
        for m in analysis.missing_clauses:
            st.markdown(f"<div class='missing-card'>⚠️ Not found: {m}</div>", unsafe_allow_html=True)
        st.divider()

    st.subheader("Key Clauses Extracted")
    st.caption("Review each clause. Edit if needed, then Accept or Reject.")

    # Use a cleaner one-column layout for review
    for i, clause in enumerate(clauses):
        icon  = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(clause.risk_level, "⚪")
        dec   = decisions.get(clause.clause_type, {})
        badge = {"accepted": " ✅", "edited": " ✏️", "rejected": " ❌"}.get(dec.get("decision", ""), "")
        with st.expander(
            f"{icon} **{clause.clause_type.title()}**{badge}  —  [Model Conf: {clause.confidence:.1%} | {len(clause.extracted_text.split())} tokens]",
            expanded=(clause.risk_level == "high" and clause.clause_type not in decisions)
        ):
            st.markdown(f"*{clause.summary}*")
            st.caption(f"Risk: **{clause.risk_level.upper()}** — {clause.risk_explanation}")
            edited = st.text_area("Clause text (you can edit)",
                                  value=clause.extracted_text, height=90,
                                  key=f"ct_{i}", label_visibility="collapsed")
            a1, a2, a3 = st.columns(3)
            with a1:
                if st.button("✅ Accept Validation", key=f"acc_{i}", use_container_width=True):
                    decisions[clause.clause_type] = {"decision": "accepted", "edited_text": edited}
                    st.session_state.audit_log.append({
                        "action": f"**{clause.clause_type}** accepted",
                        "time": datetime.datetime.now().strftime("%I:%M %p")})
                    db_manager.save_hitl_decision(
                        contract_name=analysis.contract_name,
                        clause_type=clause.clause_type,
                        risk_level=clause.risk_level,
                        decision="accepted",
                        original_text=clause.extracted_text,
                        edited_text=edited
                    )
                    st.rerun()
            with a2:
                if st.button("✏️ Edit & Save", key=f"edit_{i}", use_container_width=True):
                    decisions[clause.clause_type] = {"decision": "edited", "edited_text": edited}
                    st.session_state.audit_log.append({
                        "action": f"**{clause.clause_type}** edited",
                        "time": datetime.datetime.now().strftime("%I:%M %p")})
                    db_manager.save_hitl_decision(
                        contract_name=analysis.contract_name,
                        clause_type=clause.clause_type,
                        risk_level=clause.risk_level,
                        decision="edited",
                        original_text=clause.extracted_text,
                        edited_text=edited
                    )
                    st.rerun()
            with a3:
                if st.button("❌ False Positive", key=f"rej_{i}", use_container_width=True):
                    decisions[clause.clause_type] = {"decision": "rejected", "edited_text": ""}
                    st.session_state.audit_log.append({
                        "action": f"**{clause.clause_type}** rejected",
                        "time": datetime.datetime.now().strftime("%I:%M %p")})
                    db_manager.save_hitl_decision(
                        contract_name=analysis.contract_name,
                        clause_type=clause.clause_type,
                        risk_level=clause.risk_level,
                        decision="rejected",
                        original_text=clause.extracted_text,
                        edited_text=""
                    )
                    st.rerun()
            if clause.clause_type in decisions:
                st.caption({"accepted": "✅ Accepted", "edited": "✏️ Edited & Accepted",
                            "rejected": "❌ Rejected"}.get(dec.get("decision", ""), ""))

    st.divider()
    st.subheader("📥 Export Final Report")
    st.caption("Generate a downloadable DOCX report containing the extracted clauses, risk analysis, and review decisions.")
    
    def create_docx_report():
        import io
        from docx import Document
        from docx.shared import RGBColor
        
        doc = Document()
        doc.add_heading(f"Contract Review Report: {analysis.contract_name}", 0)
        
        doc.add_heading("Overall Risk Profile", level=1)
        doc.add_paragraph(f"Risk Level: {analysis.overall_risk_level.upper()}")
        doc.add_paragraph(analysis.executive_summary)

        # Routing tags
        routine_tags = ", ".join(list(routing_departments)) if routing_departments else "Legal"
        doc.add_heading("Suggested Routing", level=1)
        doc.add_paragraph(routine_tags)
        
        doc.add_heading("Extracted Clauses", level=1)
        for clause in clauses:
            dec = decisions.get(clause.clause_type, {})
            status = dec.get("decision", "pending")
            if status == "rejected":
                continue
                
            p = doc.add_paragraph()
            p.add_run(f"{clause.clause_type.title()} ").bold = True
            run = p.add_run(f"({status.upper()})")
            if clause.risk_level == 'high':
                run.font.color.rgb = RGBColor(255, 0, 0)
            
            doc.add_paragraph(f"Risk Level: {clause.risk_level.upper()}")
            doc.add_paragraph(f"Summary: {clause.summary}")
            
            text_to_show = dec.get("edited_text", clause.extracted_text)
            if text_to_show:
                doc.add_paragraph(f"Extracted Text:\n{text_to_show}")
            
        bio = io.BytesIO()
        doc.save(bio)
        return bio.getvalue()

    st.download_button(
        label="📄 Generate & Download DOCX Report",
        data=create_docx_report(),
        file_name=f"ReviewReport_{analysis.contract_name}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type="primary",
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# Q&A PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "qa":
    st.title("💬 Contract Q&A")

    if not st.session_state.contract_analysis:
        st.info("Please upload a contract first.")
        if st.button("Go to Upload"):
            st.session_state.page = "upload"
            st.rerun()
        st.stop()

    st.markdown(f"Ask anything about **{st.session_state.contract_name}** in plain English.")
    st.divider()

    # Initialize dynamic questions state if absent
    if "dynamic_questions" not in st.session_state:
        st.session_state.dynamic_questions = []

    # If dynamic questions haven't been loaded, generate them
    if not st.session_state.dynamic_questions and st.session_state.contract_analysis:
        from modules.llm_engine import generate_dynamic_questions
        with st.spinner("Generating custom contract queries..."):
            # Extract clause types present in document
            clause_types = [c.clause_type.lower() for c in getattr(st.session_state.contract_analysis, "clauses", [])]
            
            # Map clause types directly to guaranteed high-precision questions
            custom_q = []
            for c_type in clause_types:
                if "liab" in c_type:
                    custom_q.append("What is the liability cap under this contract?")
                elif "terminat" in c_type:
                    custom_q.append("How can this agreement be terminated?")
                elif "intellectual" in c_type or "ip" in c_type:
                    custom_q.append("Who owns the intellectual property created?")
                elif "confidential" in c_type or "nda" in c_type:
                    custom_q.append("What confidentiality obligations apply under this agreement?")
                elif "pay" in c_type or "fee" in c_type:
                    custom_q.append("What are the payment schedule and fee terms?")
                elif "compete" in c_type or "solicit" in c_type:
                    custom_q.append("What non-compete restrictions apply under this agreement?")
                elif "governing" in c_type or "jurisdiction" in c_type:
                    custom_q.append("What governing law applies to this agreement?")

            # Deduplicate while preserving order
            seen_q = set()
            dedup_q = []
            for q in custom_q:
                if q not in seen_q:
                    seen_q.add(q)
                    dedup_q.append(q)

            if len(dedup_q) >= 3:
                st.session_state.dynamic_questions = dedup_q[:3]
            else:
                if "chunks" in st.session_state and st.session_state.chunks:
                    source_text = "\n".join([c.text for c in st.session_state.chunks[:4]])
                else:
                    source_text = st.session_state.contract_analysis.executive_summary
                
                llm_q = generate_dynamic_questions(source_text)
                for q in llm_q:
                    if q not in seen_q:
                        seen_q.add(q)
                        dedup_q.append(q)
                
                defaults = [
                    "What is the liability cap under this contract?",
                    "How can this agreement be terminated?",
                    "What key restrictions or obligations apply?"
                ]
                for d in defaults:
                    if d not in seen_q:
                        seen_q.add(d)
                        dedup_q.append(d)
                
                st.session_state.dynamic_questions = dedup_q[:3]

    # Suggested questions — more compact design
    st.markdown("<div style='font-size: 0.85rem; font-weight: 500; color: #60a5fa;'>SUGGESTED QUERIES</div>", unsafe_allow_html=True)
    suggestions = st.session_state.dynamic_questions
    
    cols = st.columns(3)
    for idx, full_q in enumerate(suggestions[:3]):
        # Display the complete question without truncation
        if cols[idx % 3].button(full_q, key=f"sq_{idx}", use_container_width=True):
            st.session_state.prefill_q = full_q
            st.session_state.auto_ask  = True

    st.divider()

    # Voice section — record in the browser, transcribe with Groq Whisper, auto-ask
    col_v, _ = st.columns([1, 2])
    with col_v:
        st.markdown("**🎤 Ask by voice**")
        if not hasattr(st, "audio_input"):
            st.caption("Voice input needs Streamlit 1.39 or newer.")
        elif not is_cloud_available():
            st.caption("Voice input needs a GROQ_API_KEY (used for Whisper transcription).")
        else:
            audio_in = st.audio_input("Record your question", key="voice_q", label_visibility="collapsed")
            if audio_in is not None:
                import hashlib
                audio_bytes = audio_in.getvalue()
                audio_id = hashlib.md5(audio_bytes).hexdigest()
                # Transcribe each recording once (Streamlit reruns keep the widget value)
                if audio_bytes and st.session_state.get("last_voice_id") != audio_id:
                    st.session_state.last_voice_id = audio_id
                    from modules.voice_handler import transcribe_audio_bytes, is_valid_question
                    with st.spinner("Transcribing..."):
                        try:
                            spoken, _ = transcribe_audio_bytes(audio_bytes, audio_format="wav")
                            if is_valid_question(spoken):
                                st.session_state.prefill_q = spoken
                                st.session_state.auto_ask = True
                                st.session_state.voice_heard = spoken
                                st.rerun()
                            else:
                                st.warning("Didn't catch that. Try again a little closer to the mic.")
                        except Exception as e:
                            st.error(f"Voice error: {e}")
            if st.session_state.get("voice_heard"):
                st.caption(f'Heard: "{st.session_state.voice_heard}"')

    st.divider()

    # Initialize edit state
    if "editing_msg_idx" not in st.session_state:
        st.session_state.editing_msg_idx = None

    # Render Chat History
    for idx, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            if st.session_state.editing_msg_idx == idx:
                with st.form(key=f"edit_form_{idx}"):
                    edit_q = st.text_input("Edit question", value=msg["content"], label_visibility="collapsed")
                    ec1, ec2 = st.columns([1,1])
                    if ec1.form_submit_button("✅ Save"):
                        st.session_state.chat_history = st.session_state.chat_history[:idx]
                        st.session_state.prefill_q = edit_q
                        st.session_state.auto_ask = True
                        st.session_state.editing_msg_idx = None
                        st.rerun()
                    if ec2.form_submit_button("❌"):
                        st.session_state.editing_msg_idx = None
                        st.rerun()
            else:
                st.markdown(f"""
                <div class='user-msg'>
                    <div style='display: flex; justify-content: space-between;'>
                        <div class='msg-label'>YOU</div>
                    </div>
                    <div>{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Assistant Message with premium styling
            sources_html = "".join([f"<span class='source-tag'>{s}</span>" for s in msg.get("sources", [])[:3]])
            _conf = float(msg.get("confidence", 0) or 0)
            # Show how strongly the answer is grounded in specific passages, as a tier rather than a raw score
            if _conf > 0.7:
                conf_color, conf_label = "#10b981", "STRONG"
            elif _conf > 0.4:
                conf_color, conf_label = "#f59e0b", "MODERATE"
            else:
                conf_color, conf_label = "#94a3b8", "BROAD (answer draws on several sections)"

            import html as _html, re as _re
            _lines = [l.strip() for l in msg['content'].replace('**', '').split("\n") if l.strip()]
            # Some models put bullets on one line ("- a - b"); split those too
            if len(_lines) == 1 and _lines[0].startswith("- ") and " - " in _lines[0]:
                _lines = ["- " + x.strip() for x in _re.split(r"\s+-\s+", _lines[0][2:]) if x.strip()]
            _parts, _items = [], []
            for _l in _lines:
                if _re.match(r"^([-*•]|\d+[.)])\s+", _l):
                    _items.append("<li>" + _html.escape(_re.sub(r"^([-*•]|\d+[.)])\s+", "", _l)) + "</li>")
                else:
                    if _items:
                        _parts.append("<ul style='margin:4px 0 8px 18px; padding:0;'>" + "".join(_items) + "</ul>"); _items = []
                    _parts.append("<p style='margin:0 0 8px 0;'>" + _html.escape(_l) + "</p>")
            if _items:
                _parts.append("<ul style='margin:4px 0 8px 18px; padding:0;'>" + "".join(_items) + "</ul>")
            answer_html = "".join(_parts)
            
            st.markdown(f"""
            <div class='bot-msg'>
                <div class='msg-label'>AI ASSISTANT</div>
                <div style='margin-bottom: 12px;'>{answer_html}</div>
                <div style='display: flex; align-items: center; gap: 10px; opacity: 0.8;'>
                    <div style='font-size: 11px; color: {conf_color}; font-weight: 700;' title='How closely the top retrieved passage matches the question (cross-encoder relevance: {_conf:.0%})'>
                        SOURCE MATCH: {conf_label}
                    </div>
                    <div style='border-left: 1px solid rgba(255,255,255,0.1); height: 12px;'></div>
                    <div>{sources_html}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if msg.get("audio_out"):
                # Voice-asked answers play automatically once, right after they arrive
                st.audio(msg["audio_out"], format="audio/mp3", autoplay=bool(msg.pop("autoplay", False)))
            
            if msg.get("latency_ms") is not None:
                st.caption(f"⏱ Response time: {msg['latency_ms']:.0f}ms")
            
            if msg.get("chunks_used"):
                with st.expander("🔍 Legal Citations", expanded=False):
                    for i, chunk in enumerate(msg["chunks_used"][:3]):
                        st.caption(f"Context Block {i+1}")
                        st.text(chunk["text"][:300] + "...")
                        if i < 2: st.divider()

    if st.session_state.chat_history:
        if st.button("🗑️", help="Clear Conversation"):
            st.session_state.chat_history = []
            st.rerun()

    # Auto-scroll to bottom and button placement
    import streamlit.components.v1 as components
    components.html(
        """
        <script>
            const doc = window.parent.document;
            
            // Auto-scroll logic
            const scrollNode = doc.querySelector('.main') || doc.querySelector('.stApp');
            if (scrollNode) {
                window.parent.requestAnimationFrame(() => {
                    scrollNode.scrollTo({top: scrollNode.scrollHeight, behavior: 'smooth'});
                });
            }

            // Position clear button inside the chat input box
            function enhanceChat() {
                const buttons = doc.querySelectorAll('button');
                let clearBtn = null;
                buttons.forEach(btn => {
                    if (btn.innerText.includes('🗑️')) {
                        clearBtn = btn;
                    }
                });
                
                const chatInput = doc.querySelector('div[data-testid="stChatInput"]');
                if (clearBtn && chatInput) {
                    const chatRect = chatInput.getBoundingClientRect();
                    
                    // Hide the wrapper so it doesn't take up space in the normal layout
                    const wrapper = clearBtn.closest('div[data-testid="element-container"]');
                    if (wrapper && wrapper.style.position !== 'absolute') {
                        wrapper.style.position = 'absolute';
                        wrapper.style.visibility = 'hidden';
                    }
                    
                    // Detach button visually
                    clearBtn.style.position = 'fixed';
                    clearBtn.style.visibility = 'visible';
                    clearBtn.style.zIndex = '99999';
                    
                    // Position it inside the right side of the chat box, just left of the send button
                    clearBtn.style.left = (chatRect.right - 90) + 'px';
                    clearBtn.style.top = (chatRect.top + chatRect.height / 2 - 18) + 'px';
                    
                    // Premium styling
                    clearBtn.style.borderRadius = '50%';
                    clearBtn.style.width = '36px';
                    clearBtn.style.height = '36px';
                    clearBtn.style.background = 'rgba(30, 41, 59, 0.9)';
                    clearBtn.style.border = '1px solid rgba(255, 255, 255, 0.1)';
                    clearBtn.style.display = 'flex';
                    clearBtn.style.alignItems = 'center';
                    clearBtn.style.justifyContent = 'center';
                    clearBtn.style.boxShadow = '0 2px 8px rgba(0,0,0,0.3)';
                    clearBtn.style.cursor = 'pointer';
                    
                    // Add hover effects using JS since we're manipulating styles directly
                    clearBtn.onmouseover = () => {
                        clearBtn.style.background = 'rgba(239, 68, 68, 0.2)';
                        clearBtn.style.borderColor = 'rgba(239, 68, 68, 0.5)';
                    };
                    clearBtn.onmouseout = () => {
                        clearBtn.style.background = 'rgba(30, 41, 59, 0.9)';
                        clearBtn.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                    };
                    
                    // Pad the textarea so text doesn't hide behind our button
                    const textArea = chatInput.querySelector('textarea');
                    if (textArea) {
                        textArea.style.paddingRight = '100px';
                    }
                }
            }
            
            enhanceChat();
            window.parent.addEventListener('resize', enhanceChat);
            setInterval(enhanceChat, 500);
        </script>
        """,
        height=0
    )

    # The chat input
    prompt = st.chat_input("Ask a question about the contract...")
    should_answer = prompt or (st.session_state.auto_ask and st.session_state.prefill_q)

    if should_answer:
        final_question = prompt if prompt else st.session_state.prefill_q
        asked_by_voice = (not prompt) and final_question == st.session_state.get("voice_heard")
        st.session_state.prefill_q = ""
        st.session_state.auto_ask  = False
        
        st.session_state.chat_history.append({"role": "user", "content": final_question})
        
        from modules.rag_pipeline import rag_retrieve_and_assemble
        from modules.llm_engine import answer_question

        with st.spinner("Analyzing contract text..."):
            try:
                import time
                start_time = time.time()
                context, chunks_used = rag_retrieve_and_assemble(
                    query=final_question, 
                    top_k=8,
                    filter_source_file=st.session_state.contract_name,
                    search_method=st.session_state.rag_search_method,
                    query_enhancement=st.session_state.rag_query_enhancement,
                    rerank_results=st.session_state.rag_rerank_enabled
                )
                answer = answer_question(
                    question=final_question,
                    context=context,
                    chat_history=st.session_state.chat_history,
                    chunks_used=chunks_used)
                
                # Ensure open-ended questions don't start with Yes/No
                q_lower = final_question.strip().lower()
                if any(q_lower.startswith(w) for w in ["how", "what", "when", "where", "why", "who", "which", "explain", "describe"]):
                    for pfx in ["Yes. ", "No. ", "Yes,\n", "No,\n", "Yes, ", "No, "]:
                        if answer.answer.startswith(pfx):
                            answer.answer = answer.answer[len(pfx):]
                            break
                
                latency_ms = (time.time() - start_time) * 1000

                # Formatted retrieved chunks for telemetry logging
                formatted_chunks = [
                    {
                        "text": c.get("text", ""),
                        "score": float(c.get("similarity", 0.0)),
                        "source": c.get("metadata", {}).get("source_file", st.session_state.contract_name),
                        "rank": idx
                    }
                    for idx, c in enumerate(chunks_used)
                ]

                # Log trace using telemetry logger
                try:
                    telemetry_logger.log_query_trace(
                        query=final_question,
                        response=answer.answer,
                        retrieved_chunks=formatted_chunks,
                        latency_ms=latency_ms,
                        search_method=st.session_state.rag_search_method,
                        query_enhancement=st.session_state.rag_query_enhancement,
                        rerank_enabled=st.session_state.rag_rerank_enabled
                    )
                except Exception as telemetry_err:
                    print(f"Telemetry logging error: {telemetry_err}")

                # Spoken reply only for voice questions (keeps typed answers fast)
                audio_out = None
                if asked_by_voice:
                    try:
                        from modules.voice_handler import text_to_speech, answer_for_voice
                        audio_out = text_to_speech(answer_for_voice(answer))
                    except Exception:
                        pass
                    st.session_state.voice_heard = ""

                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer.answer,
                    "confidence": answer.confidence,
                    "sources": answer.sources,
                    "audio_out": audio_out,
                    "autoplay": audio_out is not None,
                    "chunks_used": chunks_used,
                    "latency_ms": latency_ms
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    
    # ══════════════════════════════════════════════════════════════════════════════
    # DASHBOARD — FIX 4: imports inside block, safe .get() everywhere
    # ══════════════════════════════════════════════════════════════════════════════
elif page == "dashboard":
    st.markdown("<h1 class='executive-header'>Risk Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 30px;'>Multi-dimensional audit analytics and vector-based risk assessment.</p>", unsafe_allow_html=True)

    if not st.session_state.match_results:
        st.info("Upload a contract to see the dashboard.")
        if st.button("Go to Upload"):
            st.session_state.page = "upload"
            st.rerun()
        st.stop()

    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    from modules.clause_matcher import summarize_risk_profile

    mr       = st.session_state.match_results
    profile  = summarize_risk_profile(mr)
    analysis = st.session_state.contract_analysis
    re       = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
        analysis.overall_risk_level.lower(), "⚪") if analysis else "⚪"

    # 1. PREMIUM KPI ROW
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class='glass-card' style='padding: 20px; text-align: center; height: 135px; display: flex; flex-direction: column; justify-content: center;'>
            <div class='metric-label-small' style='font-size: 0.7rem; letter-spacing: 1.5px; margin-bottom: 8px;'>OVERALL PROFILE</div>
            <div class='metric-value-large' style='color: {"#f43f5e" if analysis.overall_risk_level=="high" else "#10b981"}; margin-bottom: 6px;'>
                {analysis.overall_risk_level[0].upper()}
            </div>
            <div style='font-size: 0.75rem; font-weight: 600; color: #64748b; letter-spacing: 0.5px;'>{analysis.overall_risk_level.upper()} RISK</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class='glass-card' style='padding: 20px; text-align: center; height: 135px; display: flex; flex-direction: column; justify-content: center;'>
            <div class='metric-label-small' style='font-size: 0.7rem; letter-spacing: 1.5px; margin-bottom: 8px;'>SECTIONS SCANNED</div>
            <div class='metric-value-large' style='margin-bottom: 6px;'>{profile.get("total_clauses", 0)}</div>
            <div style='font-size: 0.75rem; font-weight: 600; color: #64748b; letter-spacing: 0.5px;'>VECTOR CHUNKS</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class='glass-card' style='padding: 20px; text-align: center; height: 135px; display: flex; flex-direction: column; justify-content: center;'>
            <div class='metric-label-small' style='font-size: 0.7rem; letter-spacing: 1.5px; margin-bottom: 8px;'>AGGREGATE SCORE</div>
            <div class='metric-value-large' style='margin-bottom: 6px;'>{profile.get('average_risk_score', 0):.2f}</div>
            <div style='font-size: 0.75rem; font-weight: 600; color: #64748b; letter-spacing: 0.5px;'>MAX 1.00</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        clauses = st.session_state.contract_analysis.clauses if st.session_state.contract_analysis else []
        decisions = st.session_state.get("hitl_decisions", {})
        reviewed = sum(1 for c in clauses if c.clause_type in decisions)
        rate_p = reviewed / max(len(clauses), 1)
        st.markdown(f"""
        <div class='glass-card' style='padding: 20px; text-align: center; height: 135px; display: flex; flex-direction: column; justify-content: center;'>
            <div class='metric-label-small' style='font-size: 0.7rem; letter-spacing: 1.5px; margin-bottom: 8px;'>VERIFICATION</div>
            <div class='metric-value-large' style='margin-bottom: 6px;'>{rate_p:.0%}</div>
            <div style='font-size: 0.75rem; font-weight: 600; color: #64748b; letter-spacing: 0.5px;'>AUDIT PROGRESS</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. ANALYST SUMMARY CARD
    if analysis:
        st.markdown(f"""
        <div class='glass-card' style='padding: 24px 30px; border-left: 3px solid #3b82f6; background: linear-gradient(135deg, rgba(15, 23, 42, 0.6) 0%, rgba(30, 41, 59, 0.4) 100%); border-radius: 16px;'>
            <div class='msg-label' style='color: #60a5fa; margin-bottom: 12px; font-size: 10px; letter-spacing: 2px;'>PRIMARY RISK ASSESSMENT</div>
            <div style='font-size: 1.05rem; line-height: 1.7; color: #cbd5e1; font-weight: 300;'>
                {analysis.executive_summary}
            </div>
        </div>
        """, unsafe_allow_html=True)
    # 3. INTUITIVE ANALYTICS
    st.markdown("<br>", unsafe_allow_html=True)
    v1 = st.container()
    import streamlit.components.v1 as components

    with v1:
        c0, c2 = st.columns([0.8, 0.2])
        with c0:
            st.markdown("<div class='msg-label' style='margin-bottom: 20px; color: #60a5fa;'>CONTRACT RISK METER</div>", unsafe_allow_html=True)
        tech_mode = False  # Disabled UI toggle to enforce standard business metrics
        with c2:
            components.html("""
            <style>
                .pdf-btn {
                    padding: 4px 10px;
                    background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(30, 64, 175, 0.2));
                    border: 1px solid rgba(96, 165, 250, 0.5);
                    color: #93c5fd;
                    border-radius: 6px;
                    font-family: -apple-system, system-ui, sans-serif;
                    font-size: 13px;
                    font-weight: 500;
                    cursor: pointer;
                    width: 100%;
                    transition: all 0.2s ease;
                }
                .pdf-btn:hover {
                    background: rgba(59, 130, 246, 0.4);
                    color: #ffffff;
                }
            </style>
            <button class="pdf-btn" onclick="window.parent.print()">
                📄 PDF
            </button>
            """, height=40)

        if mr:
            import numpy as np
            # 1. Calculate Risk Score
            avg_risk = profile.get('average_risk_score', 0)
            llm_risk_level = getattr(analysis, 'overall_risk_level', 'medium').lower()
            
            # Anchor Risk Score strictly to LLM holistic assessment
            if llm_risk_level == "high":
                risk_score = int(min(100, max(75, 75 + (avg_risk * 25))))
                risk_str = "High"
            elif llm_risk_level == "low":
                risk_score = int(min(30, max(0, avg_risk * 30)))
                risk_str = "Low"
            else:
                risk_score = int(min(74, max(31, 31 + (avg_risk * 38))))
                risk_str = "Medium"
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                number={'suffix': "/100", 'font': {'size': 48, 'color': '#f8fafc', 'family': 'Outfit'}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.2)"},
                    'bar': {'color': "rgba(255,255,255,0.9)", 'thickness': 0.15},
                    'bgcolor': "rgba(255,255,255,0.05)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.75)"},   # Green
                        {'range': [30, 75], 'color': "rgba(251, 191, 36, 0.75)"},  # Yellow
                        {'range': [75, 100], 'color': "rgba(244, 63, 94, 0.75)"}   # Red
                    ],
                    'threshold': {
                        'line': {'color': "#ffffff", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_score
                    }
                }
            ))
            
            fig.update_layout(
                height=220, margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Outfit", color="#94a3b8")
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # --- 2. Additional Breakdown Cards (Two-Track Logic) ---
            df_full = pd.DataFrame(mr)
            found_count = len(df_full['matched_clause'].unique())
            completeness = min(100, int((found_count / 14) * 100)) # 14 canonical clauses standard
            
            try:
                avg_chunk_len = int(np.mean([len(str(c.get('text', ''))) for c in mr]))
                clarity = max(40, min(98, 100 - int(avg_chunk_len / 50)))
            except:
                clarity = 78
                
            risk_col = "#fca5a5" if risk_str=="High" else ("#fde047" if risk_str=="Medium" else "#6ee7b7")

            # Dynamic Dictionaries for display modes
            if completeness >= 90:
                b_comp_t, b_comp_v = "✅ COMPLETENESS", "All standard clauses included ✓"
            elif completeness >= 60:
                b_comp_t, b_comp_v = "✅ COMPLETENESS", f"Coverage: {completeness}%"
            else:
                b_comp_t, b_comp_v = "❌ COMPLETENESS", "Missing key clauses"
            b_comp_s = getattr(analysis, "completeness_assessment", None) or ("Dynamic completeness overview (Re-upload contract to generate).")
            t_comp_t, t_comp_v, t_comp_s = "✅ COMPLETENESS (NLP)", f"{completeness}% Match", f"Found: {found_count} / 14 canonical vectors. Missing: {len(analysis.missing_clauses or [])} core components."

            b_clar_s = getattr(analysis, "readability_assessment", None) or "Dynamic linguistic overview (Re-upload contract to generate)."
            _rtxt = b_clar_s.lower()
            # Headline follows the AI's written assessment when it has one, so the two never contradict
            if any(w in _rtxt for w in ("dense", "complex", "legalese", "difficult", "technical", "hard to")):
                b_clar_t, b_clar_v = "💎 READABILITY", "Complex"
            elif any(w in _rtxt for w in ("moderate", "mixed", "fairly")):
                b_clar_t, b_clar_v = "💎 READABILITY", "Moderate"
            elif any(w in _rtxt for w in ("clear", "plain", "readable", "simple", "easy")):
                b_clar_t, b_clar_v = "💎 READABILITY", "Highly Readable"
            elif clarity >= 80:
                b_clar_t, b_clar_v = "💎 READABILITY", "Highly Readable"
            elif clarity >= 60:
                b_clar_t, b_clar_v = "💎 READABILITY", "Moderate"
            else:
                b_clar_t, b_clar_v = "💎 READABILITY", "Complex"
            t_clar_t, t_clar_v, t_clar_s = "🔍 CLARITY SCORE", f"{clarity}%", f"Avg semantic cluster span: {avg_chunk_len} chars."

            if risk_str == "Low":
                b_risk_t, b_risk_v = "⚠️ RISK LEVEL", "Safe (Low Risk)"
            elif risk_str == "Medium":
                b_risk_t, b_risk_v = "⚠️ RISK LEVEL", "Moderate Risk"
            else:
                b_risk_t, b_risk_v = "⚠️ RISK LEVEL", "High Risk"
            b_risk_s = getattr(analysis, "risk_assessment", None) or ("Dynamic risk overview (Re-upload contract to generate).")
            t_risk_t, t_risk_v, t_risk_s = "⚠️ RISK LEVEL", f"{risk_str} ({avg_risk:.2f})", f"Aggregate ML risk mapped across {found_count} logic vectors."

            if risk_score > 70:
                b_bal_t, b_bal_v = "⚖️ FAIRNESS", "One-Sided"
            elif risk_score <= 40:
                b_bal_t, b_bal_v = "⚖️ FAIRNESS", "Balanced"
            else:
                b_bal_t, b_bal_v = "⚖️ FAIRNESS", "Slightly Skewed"
            _fair_sum = getattr(analysis, "fairness_summary", None)
            if _fair_sum:
                b_bal_v = _fair_sum
            b_bal_s = (getattr(analysis, "fairness_details", None)
                       or getattr(analysis, "fairness_assessment", None)
                       or "Dynamic fairness overview (Re-upload contract to generate).")
            balance_tech = "Company Skew" if risk_score > 70 else ("Neutral" if risk_score <= 40 else "Warning")
            t_bal_t, t_bal_v, t_bal_s = "⚖️ BALANCE BIAS", balance_tech, "Analyzed via asymmetry in risk matrices."

            # Render logic
            comp_t, comp_v, comp_s = (t_comp_t, t_comp_v, t_comp_s) if tech_mode else (b_comp_t, b_comp_v, b_comp_s)
            clar_t, clar_v, clar_s = (t_clar_t, t_clar_v, t_clar_s) if tech_mode else (b_clar_t, b_clar_v, b_clar_s)
            risk_t, risk_v, risk_s = (t_risk_t, t_risk_v, t_risk_s) if tech_mode else (b_risk_t, b_risk_v, b_risk_s)
            bal_t, bal_v, bal_s = (t_bal_t, t_bal_v, t_bal_s) if tech_mode else (b_bal_t, b_bal_v, b_bal_s)

            value_font_size = "18px" if not tech_mode else "20px"
            info_icon = "<span style='display:inline-block; border: 1px solid rgba(148,163,184,0.5); border-radius: 50%; width: 14px; height: 14px; text-align: center; line-height: 13px; font-size: 9px; font-weight: 600; font-family: sans-serif; margin-left: 4px; cursor: help;'>i</span>"

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""
                <div title='Percentage of standard contract clauses successfully identified.' style='background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255,255,255,0.05); padding: 15px 20px; border-radius: 12px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);'>
                    <div style='font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; display: flex; align-items: center;'>{comp_t}{info_icon}</div>
                    <div style='font-size: {value_font_size}; color: #f8fafc; font-weight: 700;'>{comp_v}</div>
                    <div style='font-size: 12px; color: #94a3b8; margin-top: 4px; line-height: 1.4;'>{comp_s}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div title='Readability metric derived from average semantic clause span (legalese density).' style='background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255,255,255,0.05); padding: 15px 20px; border-radius: 12px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);'>
                    <div style='font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; display: flex; align-items: center;'>{clar_t}{info_icon}</div>
                    <div style='font-size: {value_font_size}; color: #f8fafc; font-weight: 700;'>{clar_v}</div>
                    <div style='font-size: 12px; color: #94a3b8; margin-top: 4px; line-height: 1.4;'>{clar_s}</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div title='Calculated aggregate risk severity indicating signing safety.' style='background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255,255,255,0.05); border-right: 3px solid {risk_col}; padding: 15px 20px; border-radius: 12px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);'>
                    <div style='font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; display: flex; align-items: center;'>{risk_t}{info_icon}</div>
                    <div style='font-size: {value_font_size}; color: {risk_col}; font-weight: 700;'>{risk_v}</div>
                    <div style='font-size: 12px; color: #94a3b8; margin-top: 4px; line-height: 1.4;'>{risk_s}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div title='Asymmetry analysis determining if the contract heavily favors the counterparty.' style='background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255,255,255,0.05); padding: 15px 20px; border-radius: 12px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);'>
                    <div style='font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; display: flex; align-items: center;'>{bal_t}{info_icon}</div>
                    <div style='font-size: {value_font_size}; color: #f8fafc; font-weight: 700;'>{bal_v}</div>
                    <div style='font-size: 12px; color: #94a3b8; margin-top: 4px; line-height: 1.4;'>{bal_s}</div>
                </div>
                """, unsafe_allow_html=True)



    # 4. CAPSTONE-WORTHY RED FLAGS & VULNERABILITIES
    if analysis.red_flags or analysis.missing_clauses:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='msg-label' style='color: #f43f5e; margin-bottom: 5px; font-size: 14px;'>IDENTIFIED LEGAL VULNERABILITIES</div>", unsafe_allow_html=True)
        st.markdown("<div style='height: 2px; width: 100%; background: linear-gradient(90deg, #f43f5e 0%, transparent 100%); margin-bottom: 20px;'></div>", unsafe_allow_html=True)
        
        for f in analysis.red_flags:
            fw = f.lower()
            if any(w in fw for w in ["missing", "failure", "breach", "liability", "indemnif", "cap"]):
                icon, level, color = "🔴", "CRITICAL", "#fca5a5"
                bg_col = "rgba(252, 165, 165, 0.05)"
            elif any(w in fw for w in ["unclear", "ambiguous", "notice", "term", "auto-renewal"]):
                icon, level, color = "🟡", "WARNING", "#fde047"
                bg_col = "rgba(253, 224, 71, 0.05)"
            else:
                icon, level, color = "🟠", "MODERATE", "#fdba74"
                bg_col = "rgba(253, 186, 116, 0.05)"
                
            # 2. Heuristic Recommendation Mapping
            rec = "Ensure manual review by legal counsel prior to signature."
            if level == "CRITICAL":
                rec = "Stop execution. Flag immediately to the executive team for renegotiation of core terms."
            elif any(w in fw for w in ["financial", "loan", "payment", "fee", "cost"]):
                rec = "Have the Finance team review the payment schedule and default triggers."
            elif any(w in fw for w in ["termination", "default", "breach"]):
                rec = "Negotiate a 30-day cure period (grace period) for non-payment defaults."
            elif any(w in fw for w in ["unclear", "ambiguous", "vague"]):
                rec = "Mandate Legal to explicitly define these vague technical terms to prevent litigation."

            # 3. Best Clause Match (Simulated Context)
            clause_match = None
            for c in clauses:
                if c.risk_level.lower() == level.lower() or ("high" in c.risk_level.lower() and level=="CRITICAL"):
                    clause_match = c
                    break
            if not clause_match and clauses:
                clause_match = clauses[0]
                
            with st.container():
                st.markdown(f"""
                <div style='background: {bg_col}; border-left: 3px solid {color}; padding: 15px 20px; border-radius: 8px; margin-bottom: 5px; border: 1px solid rgba(255,255,255,0.05);'>
                    <div style='font-size: 1.05rem; color: #e2e8f0; line-height: 1.5;'>
                        {icon} <span style='color: {color}; font-weight: 700; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-right: 8px;'>{level}</span> {f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("↳ View clause context"):
                    if tech_mode:
                        st.caption(f"Relevant context vector retrieved regarding: *{f}*. Review the extracted clauses tab or execute a semantic search query for exact phrasing.")
                    else:
                        st.markdown(f"**📍 Identified in Clause:** {clause_match.clause_type.title() if clause_match else 'General Provisions'}")
                        if clause_match:
                            st.info(f'"{clause_match.extracted_text[:180]}..."')
                        st.markdown(f"**⚠️ Actionable Recommendation:** {rec}")
        
        for m in (analysis.missing_clauses or []):
            with st.container():
                st.markdown(f"""
                <div style='background: rgba(252, 165, 165, 0.05); border-left: 3px solid #fca5a5; padding: 15px 20px; border-radius: 8px; margin-bottom: 5px; border: 1px solid rgba(255,255,255,0.05);'>
                    <div style='font-size: 1.05rem; color: #e2e8f0; line-height: 1.5;'>
                        🔴 <span style='color: #fca5a5; font-weight: 700; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-right: 8px;'>CRITICAL</span> Missing explicitly standardized clause: <b>{m}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("↳ Remediation Step"):
                    st.caption(f"This contract lacks a detected **{m}** clause. Consider routing to Legal for insertion.")

    st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# COMPARE DOCUMENTS — two agreements of the same type, term by term
# ══════════════════════════════════════════════════════════════════════════════
elif page == "compare":
    from modules.doc_types import (DOC_TYPES, detect_document_type, extract_key_terms,
                                   favorability_score, compare_documents, ai_coverage)
    from modules import llm_engine as _llm
    from modules.doc_processor import process_contract
    import tempfile, glob

    st.title("⚖️ Compare Documents")
    st.caption("Put two agreements of the same kind side by side: two credit card agreements, two leases, two job offers, "
               "two vendor contracts. ClauseWise detects the type, pulls the terms that matter for it, and says which one is better and why.")

    SAMPLE_DIR = "data/demo_contracts/compare"
    samples = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.pdf")) + glob.glob(os.path.join(SAMPLE_DIR, "*.docx")))

    def _doc_label(path_or_name: str) -> str:
        import re as _re
        name = os.path.splitext(os.path.basename(path_or_name))[0]
        name = _re.sub(r"^[A-Z]{2,}\d+[_\- ]+", "", name)          # filing codes like COL00040_
        name = _re.sub(r"[_\- ]+\d{5,}$", "", name)                # trailing database IDs
        name = _re.sub(r"^Credit[-_ ]Card[-_ ]Agreement[-_ ]for[-_ ]", "", name, flags=_re.I)
        return name.replace("_", " ").replace("-", " ").strip()

    src_a = src_b = None
    if len(samples) >= 2:
        use_samples = st.toggle("Use sample documents", value=True, key="cmp_use_samples",
                                help="Real public agreements bundled with the demo.")
    else:
        use_samples = False

    ca, cb = st.columns(2)
    if use_samples:
        with ca:
            src_a = st.selectbox("Document A", samples, index=0, format_func=_doc_label, key="cmp_sample_a")
        with cb:
            src_b = st.selectbox("Document B", samples, index=1, format_func=_doc_label, key="cmp_sample_b")
    else:
        with ca:
            up_a = st.file_uploader("Document A", type=["pdf", "docx"], key="cmp_up_a")
        with cb:
            up_b = st.file_uploader("Document B", type=["pdf", "docx"], key="cmp_up_b")
        src_a, src_b = up_a, up_b

    type_keys = ["auto"] + list(DOC_TYPES.keys())
    forced = st.selectbox("Document type", type_keys,
                          format_func=lambda k: "Auto-detect" if k == "auto" else DOC_TYPES[k]["label"], key="cmp_type")

    def _chunk_texts(src) -> tuple:
        """Return (display_name, chunk_texts) for a sample path or an uploaded file."""
        if isinstance(src, str):
            chunks, _ = process_contract(src, chunk_size=500, overlap=80, strategy="semantic")
            return _doc_label(src), [c.text for c in chunks]
        suffix = ".pdf" if src.name.lower().endswith(".pdf") else ".docx"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(src.getvalue())
            tmp_path = tmp.name
        try:
            chunks, _ = process_contract(tmp_path, chunk_size=500, overlap=80, strategy="semantic")
        finally:
            os.unlink(tmp_path)
        return _doc_label(src.name), [c.text for c in chunks]

    can_run = src_a is not None and src_b is not None
    if st.button("⚖️ Compare", type="primary", disabled=not can_run, use_container_width=True):
        use_llm = is_cloud_available()
        prog, status = st.progress(0), st.empty()
        try:
            status.info("📄 Reading both documents...")
            name_a, texts_a = _chunk_texts(src_a)
            prog.progress(20)
            name_b, texts_b = _chunk_texts(src_b)
            if name_a == name_b:
                name_a, name_b = name_a + " (A)", name_b + " (B)"
            prog.progress(35)

            status.info("🧭 Identifying document type...")
            det_a = detect_document_type(" ".join(texts_a[:12]), use_llm=use_llm)
            det_b = detect_document_type(" ".join(texts_b[:12]), use_llm=use_llm)
            dtype = det_a["type"] if forced == "auto" else forced
            prog.progress(50)

            status.info(f"📑 Extracting key terms ({DOC_TYPES[dtype]['label']})...")
            rows_a = extract_key_terms(texts_a, dtype, use_llm=use_llm)
            prog.progress(70)
            rows_b = extract_key_terms(texts_b, dtype, use_llm=use_llm)
            prog.progress(88)

            status.info("⚖️ Comparing...")
            verdict = compare_documents(name_a, rows_a, name_b, rows_b, dtype, use_llm=use_llm)
            prog.progress(100)
            st.session_state.compare_result = {
                "llm_error": _llm.LAST_LLM_ERROR if (ai_coverage(rows_a) < 1 or ai_coverage(rows_b) < 1) else "",
                "type": dtype, "type_mismatch": forced == "auto" and det_a["type"] != det_b["type"],
                "det_b": det_b["type"], "name_a": name_a, "name_b": name_b,
                "rows_a": rows_a, "rows_b": rows_b, "verdict": verdict,
            }
        except Exception as e:
            st.error(f"Comparison failed: {e}")
        finally:
            status.empty()
            prog.empty()

    res = st.session_state.compare_result
    if res:
        pack = DOC_TYPES[res["type"]]
        st.divider()
        if res["type_mismatch"]:
            st.warning(f"These look like different document types (A: {pack['label']}, "
                       f"B: {DOC_TYPES[res['det_b']]['label']}). Comparing both as a {pack['label']}.")
        cov_a, cov_b = ai_coverage(res["rows_a"]), ai_coverage(res["rows_b"])
        if cov_a < 1 or cov_b < 1:
            st.warning(f"AI extraction covered {cov_a:.0%} of terms for A and {cov_b:.0%} for B. Rows marked "
                       "'Keyword match only' are raw evidence and are left out of the scores and the verdict."
                       + (f" Last AI error: {res['llm_error']}" if res.get("llm_error") else ""))
        v = res["verdict"]
        winner = {"A": res["name_a"], "B": res["name_b"]}.get(v["better"])
        if v["better"] == "incomplete":
            headline = "⏳ Not enough terms extracted to pick a winner"
        elif winner:
            headline = f"🏆 {winner} is better for {pack['perspective']}"
        else:
            headline = f"🤝 Roughly even for {pack['perspective']}"
        st.markdown(f"### {headline}")
        st.markdown(v["summary"])
        if v.get("differences"):
            st.markdown("\n".join(f"- {d}" for d in v["differences"]))

        sa, sb = favorability_score(res["rows_a"]), favorability_score(res["rows_b"])
        m1, m2, m3 = st.columns(3)
        m1.metric(f"A · favorability", f"{sa}/100" if sa is not None else "n/a")
        m2.metric(f"B · favorability", f"{sb}/100" if sb is not None else "n/a")
        m3.metric("Document type", pack["label"])
        st.caption("Favorability = share of the checklist terms that are favorable (1), standard (0.5) or unfavorable (0) "
                   f"for {pack['perspective']}; terms not found or only keyword-matched are left out. "
                   f"Verdict based on {v.get('compared_terms', '?')} terms extracted for both documents.")

        render_comparison(res["name_a"], res["rows_a"], res["name_b"], res["rows_b"])

        import pandas as pd
        csv = pd.DataFrame([{"term": ra["label"], "A_value": ra["value"], "A_assessment": ra["assessment"],
                             "B_value": rb["value"], "B_assessment": rb["assessment"]}
                            for ra, rb in zip(res["rows_a"], res["rows_b"])]).to_csv(index=False)
        st.download_button("📥 Download comparison (CSV)", csv, file_name="clausewise_comparison.csv", mime="text/csv")
