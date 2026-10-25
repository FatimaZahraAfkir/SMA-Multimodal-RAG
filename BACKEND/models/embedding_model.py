"""Embeddings via API Gemini : plus de torch ni de sentence-transformers."""
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import settings


def load_embeddings():
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY absente du fichier .env (nécessaire pour les embeddings)")
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        output_dimensionality=settings.EMBEDDING_DIM,
    )