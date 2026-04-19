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
CHROMA_TOP_K = int(os.getenv("CHROMA_TOP_K", "5"))

# Models
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")

# Text cleaning
MAX_CONSECUTIVE_BLANK_LINES = 2  # collapse runs longer than this

# Chunking
CHUNK_SIZE = 500        # max characters per chunk
CHUNK_OVERLAP = 50      # characters shared between consecutive chunks

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
