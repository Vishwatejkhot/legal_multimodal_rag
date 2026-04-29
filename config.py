import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

VISION_MODEL  = "claude-sonnet-4-6"
REASON_MODEL  = "claude-sonnet-4-6"
EMBED_MODEL   = "all-MiniLM-L6-v2"   # local sentence-transformers, no API needed

DATA_DIR       = BASE_DIR / "data"
LEGAL_TEXT_DIR = DATA_DIR / "legal_text"
LEGISLATION_DIR= LEGAL_TEXT_DIR / "legislation"
CASE_LAW_DIR   = LEGAL_TEXT_DIR / "case_law"
SENTENCING_DIR = LEGAL_TEXT_DIR / "sentencing"
HMCTS_DIR      = LEGAL_TEXT_DIR / "hmcts"
IMAGES_DIR     = DATA_DIR / "images"
TRAINING_DIR   = DATA_DIR / "training"
INDEXES_DIR    = BASE_DIR / "indexes"
CACHE_DIR      = BASE_DIR / ".cache"

FAISS_INDEX_PATH  = str(INDEXES_DIR / "faiss.index")
FAISS_META_PATH   = str(INDEXES_DIR / "faiss_meta.json")
BM25_INDEX_PATH   = str(INDEXES_DIR / "bm25.pkl")
LANCEDB_PATH      = str(INDEXES_DIR / "lancedb")
XGBOOST_MODEL_PATH= str(INDEXES_DIR / "xgboost_evidence.json")

CHUNK_SIZE          = 600
CHUNK_OVERLAP       = 100
TOP_K               = 10
CONFIDENCE_THRESHOLD= 40.0
CACHE_TTL_SECONDS   = 86400

LEGISLATION_TARGETS = [
    {"type": "ukpga", "year": 1988, "number": 50, "title": "Housing Act 1988"},
    {"type": "ukpga", "year": 1996, "number": 52, "title": "Housing Act 1996"},
    {"type": "ukpga", "year": 1985, "number": 70, "title": "Landlord and Tenant Act 1985"},
    {"type": "ukpga", "year": 1977, "number": 43, "title": "Protection from Eviction Act 1977"},
    {"type": "ukpga", "year": 2010, "number": 15, "title": "Equality Act 2010"},
    {"type": "ukpga", "year": 1998, "number": 42, "title": "Human Rights Act 1998"},
    {"type": "ukpga", "year": 2004, "number": 34, "title": "Housing Act 2004"},
    {"type": "ukpga", "year": 2002, "number": 15, "title": "Commonhold and Leasehold Reform Act 2002"},
    {"type": "ukpga", "year": 2016, "number": 22, "title": "Housing and Planning Act 2016"},
    {"type": "ukpga", "year": 1954, "number": 56, "title": "Landlord and Tenant Act 1954"},
]

BAILII_SEARCH_TERMS = [
    "illegal eviction Section 21",
    "housing disrepair",
    "unlawful eviction damages",
    "rent arrears possession",
    "harassment landlord tenant",
    "leasehold service charges",
    "antisocial behaviour eviction",
]
