import os
from dotenv import load_dotenv

load_dotenv()

def getenv_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v and v.strip().isdigit() else default

API_KEY = os.getenv("API_KEY", "").strip()
PORT = getenv_int("PORT", 8001)
RATE_LIMIT = os.getenv("RATE_LIMIT", "60/minute").strip()

MODEL_PATH = os.getenv("MODEL_PATH", "").strip()
MODEL_URL = os.getenv("MODEL_URL", "").strip()

MODEL_N_CTX = getenv_int("MODEL_N_CTX", 4096)
MODEL_N_THREADS = getenv_int("MODEL_N_THREADS", 8)
MODEL_N_GPU_LAYERS = getenv_int("MODEL_N_GPU_LAYERS", 0)
MODEL_N_BATCH = getenv_int("MODEL_N_BATCH", 256)

RAG_DB_PATH = os.getenv("RAG_DB_PATH", "./data/rag.db").strip()
RAG_TOP_K = getenv_int("RAG_TOP_K", 5)
RAG_USE_LLM = os.getenv("RAG_USE_LLM", "1").strip() in ("1", "true", "True", "yes", "YES")
