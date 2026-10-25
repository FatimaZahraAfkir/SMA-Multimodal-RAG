import os
from dotenv import load_dotenv

load_dotenv()


def _list(name: str, default: str) -> list:
    return [p.strip() for p in os.getenv(name, default).split(",") if p.strip()]


class Settings:
    GROQ_API_KEY:   str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")  

    LLM_PROVIDERS: list = _list("LLM_PROVIDERS", "groq,gemini")
    GROQ_MODEL:   str = os.getenv("GROQ_MODEL",   "openai/gpt-oss-120b")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    LLM_TIMEOUT:  int = int(os.getenv("LLM_TIMEOUT", "30"))

    JUDGE_PROVIDERS: list = _list("JUDGE_PROVIDERS", "gemini,groq")
    VALIDATION_MIN_SCORE: float = float(os.getenv("VALIDATION_MIN_SCORE", "6"))

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    EMBEDDING_DIM:   int = int(os.getenv("EMBEDDING_DIM", "768"))
    EMBED_BATCH_SIZE: int   = int(os.getenv("EMBED_BATCH_SIZE", "50"))
    EMBED_PAUSE_S:    float = float(os.getenv("EMBED_PAUSE_S", "35"))

    GENERATION_TEMPERATURE: float = float(os.getenv("GENERATION_TEMPERATURE", "0.1"))
    MAX_REGENERATIONS: int = int(os.getenv("MAX_REGENERATIONS", "1"))

    CORS_ORIGINS: list = _list("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "15"))
    MAX_MESSAGE_CHARS: int = int(os.getenv("MAX_MESSAGE_CHARS", "4000"))
    SESSION_TTL_S: int = int(os.getenv("SESSION_TTL_S", "7200"))

    FAISS_PATH: str = os.getenv("FAISS_PATH", "database/faiss_index")


settings = Settings()