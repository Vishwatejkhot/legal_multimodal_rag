<p align="center">
  <img src="assets/logo.svg" alt="JusticeAI Logo" width="520"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Claude-Sonnet%204.6-blueviolet?style=flat-square&logo=anthropic&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.56-red?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
</p>

<p align="center">
  <b>Multimodal RAG system for UK legal caseworkers.</b><br/>
  Upload evidence. Ask a question. Get cited UK law, conflict warnings, and evidence strength — in seconds.
</p>

---

A production-grade legal evidence analysis system that combines document understanding, semantic search, and large language models to help UK caseworkers analyse evidence and find relevant law in minutes.

---

## What it does

Upload photos, scanned letters, PDFs, or Word documents alongside a legal question. JusticeAI:

- Reads images and scanned documents using **Claude Sonnet vision**
- Searches 35+ UK legal texts using **FAISS + BM25 + Reciprocal Rank Fusion**
- Re-ranks results with a **cross-encoder** for precision
- Detects contradictions between evidence using **conflict detection**
- Streams a structured legal answer with citations
- Scores evidence strength as **High / Medium / Low** using **XGBoost**
- Extracts a chronological **timeline** and flags statutory deadline breaches

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vision | Claude Sonnet 4.6 (Anthropic) |
| Text generation | Claude Sonnet 4.6 (Anthropic) |
| Embeddings | all-MiniLM-L6-v2 (local, free) |
| Semantic search | FAISS |
| Keyword search | BM25 (rank-bm25) |
| Fusion | Reciprocal Rank Fusion |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector store | LanceDB |
| Evidence scoring | XGBoost |
| UI | Streamlit |
| PDF parsing | PyMuPDF |

---

## Data Sources (all free)

| Source | Content |
|---|---|
| legislation.gov.uk API | 10 UK Acts (Housing Act 1988, Equality Act 2010, etc.) |
| BAILII (embedded) | 10 landmark housing/tenancy judgments |
| Sentencing Council | Offence guidelines (HTML) |
| HMCTS / justice.gov.uk | Civil Procedure Rules Parts 1-65 |
| LexGLUE (HuggingFace) | 300 labelled European court decisions |
| Synthetic generator | 500 UK housing case scenarios |

---

## Setup

### 1. Clone and create environment

```bash
git clone <repo-url>
cd legal_multimodal_rag
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
# Edit .env:
# ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at https://console.anthropic.com

### 3. Fetch legal data

```bash
python scripts/fetch_data.py
python scripts/fetch_training.py
python scripts/fetch_images.py
```

### 4. Build indexes

```bash
python scripts/build_index.py
```

### 5. Launch

```bash
streamlit run app.py
```

Open http://localhost:8501

---

## Project Structure

```
legal_multimodal_rag/
├── app.py                      # Streamlit UI
├── config.py                   # Settings and paths
├── requirements.txt
│
├── ingestion/                  # Data collection
│   ├── legislation_scraper.py  # legislation.gov.uk API
│   ├── bailii_scraper.py       # UK case law (embedded)
│   ├── sentencing_scraper.py   # Sentencing Council HTML
│   ├── hmcts_scraper.py        # CPR from justice.gov.uk
│   ├── pdf_parser.py           # PyMuPDF
│   ├── chunker.py              # Overlapping text chunks
│   └── deduplicator.py         # MD5 deduplication
│
├── image_pipeline/             # Multimodal input
│   ├── gpt4o_vision.py         # Claude vision API
│   ├── image_handler.py        # Routes files to correct handler
│   └── image_to_text.py        # Vision output to chunks
│
├── retrieval/                  # Search
│   ├── embedder.py             # Local sentence-transformers
│   ├── faiss_index.py          # Semantic search
│   ├── bm25_index.py           # Keyword search
│   ├── lancedb_store.py        # Multimodal vector store
│   ├── rrf_fusion.py           # Reciprocal Rank Fusion
│   └── reranker.py             # Cross-encoder re-ranking
│
├── agent/                      # Reasoning
│   ├── react_agent.py          # Main pipeline orchestration
│   ├── conflict_detector.py    # Contradiction detection
│   ├── synthesiser.py          # Answer generation + streaming
│   ├── confidence_scorer.py    # Per-chunk confidence 0-100%
│   └── timeline_extractor.py   # Date extraction + deadline checks
│
├── scorer/                     # Evidence strength
│   ├── xgboost_trainer.py      # Training on synthetic data
│   ├── xgboost_predictor.py    # High / Medium / Low prediction
│   └── feature_extractor.py    # 10 signal features
│
├── reliability/                # Production ops
│   ├── cache.py                # DiskCache 24-hour layer
│   ├── rate_limiter.py         # Token bucket + retry
│   └── logger.py               # Structured logging
│
├── scripts/
│   ├── fetch_data.py           # Fetch all legal text
│   ├── fetch_training.py       # Fetch training data
│   ├── fetch_images.py         # Fetch test images
│   └── build_index.py          # Build all indexes
│
└── data/
    ├── legal_text/             # Downloaded legal text (gitignored)
    ├── images/                 # Test images (gitignored)
    └── training/               # XGBoost training data (gitignored)
```

---

## Usage

1. Open http://localhost:8501
2. Upload evidence files (PDFs, photos, scanned letters, Word documents)
3. Type your legal question
4. Click Analyse Evidence

Example questions:
- Is this Section 21 notice legally valid?
- Does this evidence support a housing disrepair claim under the Landlord and Tenant Act 1985?
- Is the deposit protection compliant with the Deregulation Act 2015?
- What are the tenant's rights given this evidence?

Results include:
- Streamed legal analysis with cited Acts and case names
- Evidence strength badge (High / Medium / Low)
- Conflict warnings when evidence contradicts itself
- Chronological timeline with statutory deadline checks
- Source breakdown by legislation, case law, and procedure

---

## Cost

| Service | Cost |
|---|---|
| Anthropic API (vision + text) | ~0.02-0.05 GBP per query |
| Embeddings | Free (local model) |
| All legal data | Free |
| All libraries | Free |

---

## Requirements

- Python 3.12+
- Anthropic API key (https://console.anthropic.com)
- 2GB disk space (models + indexes)
- Internet connection for first run (model download + data fetch)
