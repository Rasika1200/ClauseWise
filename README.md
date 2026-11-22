# ClauseWise: AI-Powered Contract Review & Risk Analysis

## 📌 Project Overview
**ClauseWise Copilot** is a high-performance, full-stack enterprise application designed to ingest, analyze, and interrogate complex legal contracts instantly. Built for the DATS 6501 Capstone project at George Washington University, the platform replaces hours of manual legal review with a robust Retrieval-Augmented Generation (RAG) pipeline and a human-in-the-loop validation workflow.

## 📁 Project Structure
```text
DATS6501_Capstone_Project/
├── README.md               # Main project documentation
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
├── app/                    # Streamlit Dashboard UI
├── api/                    # FastAPI Extension Backend
├── modules/                # Core AI & RAG Logic
├── evaluation/             # Performance Metrics & CUAD Benchmarks
├── notebooks/              # Evaluation & Demo Notebooks
├── extension/              # Chrome Sidebar Source Code
├── data/                   # Persistence & Demo Contracts
├── screenshots/            # UI/UX Previews
└── docs/                   # Technical Architecture Documentation
```

## 🛠 Technical Architecture
ClauseWise utilizes a decoupled architecture to support both a high-end analytical dashboard and a lightweight browser extension. For a deep dive, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

```mermaid
graph TD
    A[PDF/DOCX/Webpage] --> B(Document Processor)
    B --> C{Chunking & Embedding}
    C --> D[(ChromaDB Vector Store)]
    D --> E[RAG Pipeline]
    E --> F[Groq gpt-oss-20b]
    F --> G[Streamlit Dashboard]
    F --> H[Chrome Sidebar Extension]
    G --> I[Human-in-the-Loop Validation]
    I --> J[(SQLite Audit Log)]
```

## 🚀 Key Features
1. **Semantic RAG Engine:** Uses `all-MiniLM-L6-v2` local embeddings and ChromaDB to retrieve contract context with high precision.
2. **Dynamic Risk Dashboarding:** Heuristically scores liability vectors and visualizes contract "health" via Plotly.
3. **Chrome Sidebar Extension:** A FastAPI-backed extension for analyzing contracts directly from the browser.
4. **Hybrid System Resiliency:** Automated fallback from Groq Cloud to local **Ollama (Llama 3.2)** for 100% uptime.
5. **Human-In-The-Loop (HITL):** Legal operators validate AI findings, creating a verifiable audit trail in SQLite.
6. **Voice-Enabled Interface:** Integrated STT/TTS using **OpenAI Whisper** for natural language interrogation.

## 📊 Performance & Evaluation
Measured on the **CUAD v1** benchmark (510 lawyer-annotated commercial contracts) with `openai/gpt-oss-20b` on Groq, November 2026:

| Metric | Result | Sample |
| :--- | :--- | :--- |
| **Clause-type detection F1** (micro, 6 CUAD-scorable types) | **0.71** (P 0.69 / R 0.74) | 510 contracts |
| **Extracted clause quotes found verbatim in the contract** | **82%** (4% invented) | 122 clauses |
| **Q&A accuracy on "does the contract cover X?"** | **85%** | 60 questions |
| **Q&A false "clause exists" claims** | **13%** | 30 questions |
| **Retrieval: gold passage in top-8 context** | **93%** | 30 questions |
| **Median Groq latency** | **~1.2 s** | |

### How I Evaluated
[`evaluation/run_eval.py`](evaluation/run_eval.py) runs the same code paths the app uses against CUAD's labels:
* **Clause detection:** the embedding-based clause matcher on every chunk of every contract, scored per clause type.
* **LLM extraction:** whether each clause the model quotes actually appears in the contract.
* **RAG Q&A:** a balanced set of questions about clauses CUAD marks present or absent, checking retrieval and whether the answer claims clauses that aren't there.

Full results, per-type breakdowns, failure analysis and limitations are in [EVALUATION_RESULTS.md](evaluation/EVALUATION_RESULTS.md). Human-in-the-loop decisions are logged to SQLite, but no user study has been run yet.

## ⚙️ Setup & Installation

### 1. Clone & Environment
```bash
git clone https://github.com/Rasika1200/DATS6501_Capstone_Project.git
cd DATS6501_Capstone_Project
pip install -r requirements.txt
cp .env.example .env # Add your GROQ_API_KEY
```

### 2. Run Commands
**Terminal 1 (Dashboard):** `streamlit run app/main_app.py`  
**Terminal 2 (Extension API):** `uvicorn api.main:app --reload`

---
**Author:** Rasika Shrikant Nilatkar  
**Course:** DATS 6501 Capstone | The George Washington University
