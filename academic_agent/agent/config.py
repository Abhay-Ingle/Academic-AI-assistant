"""Central configuration. Every value can be overridden via environment variables or a .env file."""
import os
from datetime import date
from pathlib import Path

try:  # python-dotenv is optional
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLE_DIR = DATA_DIR / "sample_notes"
INDEX_DIR = DATA_DIR / "index"
MEMORY_FILE = DATA_DIR / "memory.json"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "outputs"))

# ---- LLM -------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()      # groq | gemini | openai | ollama
LLM_MODEL = os.getenv("LLM_MODEL", "")                         # blank -> provider default
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# ---- Agent behaviour -------------------------------------------------------
MAX_STEPS = int(os.getenv("MAX_STEPS", "7"))                   # hard cap on tool calls per request
ENABLE_REFLECTION = os.getenv("ENABLE_REFLECTION", "true").lower() == "true"
OBSERVATION_MAX_CHARS = int(os.getenv("OBSERVATION_MAX_CHARS", "2200"))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))

# ---- RAG -------------------------------------------------------------------
USE_DENSE = os.getenv("USE_DENSE", "true").lower() == "true"  # dense embeddings + Chroma
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DENSE_MAX_DISTANCE = float(os.getenv("DENSE_MAX_DISTANCE", "0.75"))  # cosine distance cut-off

# ---- Safety ----------------------------------------------------------------
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4000"))


def today() -> date:
    """Current date (can be frozen with FAKE_TODAY=YYYY-MM-DD for reproducible tests/demos)."""
    fake = os.getenv("FAKE_TODAY")
    return date.fromisoformat(fake) if fake else date.today()


def ensure_dirs() -> None:
    for d in (DATA_DIR, UPLOAD_DIR, SAMPLE_DIR, INDEX_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
