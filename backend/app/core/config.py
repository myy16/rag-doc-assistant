import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "doc"}
MAX_FILE_SIZE_MB = 20
UPLOAD_DIR = "uploads"

# ============================================================================
# SENSITIVE CREDENTIALS (no defaults - must be set in .env or environment)
# ============================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Required for LLM generation via Groq API
HF_TOKEN = os.getenv("HF_TOKEN")  # Optional: Required only for private HuggingFace models

# ============================================================================
# NORMAL CONFIGURATION (safe to have defaults)
# ============================================================================
# RAG / Vector store
RAG_DATA_DIR = os.getenv("RAG_DATA_DIR", "data")
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(RAG_DATA_DIR, "chroma"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "docs_chunks")
CHROMA_TOP_K = int(os.getenv("CHROMA_TOP_K", "8"))
RETRIEVER_MIN_RELEVANCE_SCORE = float(os.getenv("RETRIEVER_MIN_RELEVANCE_SCORE", "0.10"))
RETRIEVER_MIN_CONTEXT_RECALL = float(os.getenv("RETRIEVER_MIN_CONTEXT_RECALL", "0.15"))
RETRIEVER_MIN_CONFIDENCE_SCORE = float(os.getenv("RETRIEVER_MIN_CONFIDENCE_SCORE", "0.15"))
RETRIEVER_MAX_DYNAMIC_TOP_K = int(os.getenv("RETRIEVER_MAX_DYNAMIC_TOP_K", "10"))
RETRIEVER_MAX_CANDIDATE_K = int(os.getenv("RETRIEVER_MAX_CANDIDATE_K", "30"))
RETRIEVER_RRF_K = int(os.getenv("RETRIEVER_RRF_K", "60"))

# Models
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

# Text cleaning
MAX_CONSECUTIVE_BLANK_LINES = 2  # collapse runs longer than this

# Chunking
CHUNK_SIZE = 1000       # max characters per chunk
CHUNK_OVERLAP = 100     # characters shared between consecutive chunks

# LLM runtime resilience
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_RETRY_BASE_DELAY_SECONDS = float(os.getenv("LLM_RETRY_BASE_DELAY_SECONDS", "1.0"))
LLM_RETRY_MAX_DELAY_SECONDS = float(os.getenv("LLM_RETRY_MAX_DELAY_SECONDS", "6.0"))
LLM_CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("LLM_CIRCUIT_BREAKER_THRESHOLD", "3"))
LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS = float(os.getenv("LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS", "45"))

# ============================================================================
# VALIDATION HELPERS: Validate required credentials only when needed
# ============================================================================
def require_groq_api_key():
    """Return the configured Groq API key or raise when Groq features are used."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY must be set in .env or environment. "
            "Without it, RAG generation and chat endpoints will fail."
        )
    return GROQ_API_KEY
