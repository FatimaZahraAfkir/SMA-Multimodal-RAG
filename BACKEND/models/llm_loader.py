from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _groq(temperature: float):
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=temperature,
        max_retries=1,
        timeout=settings.LLM_TIMEOUT,
    )


def _gemini(temperature: float):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=temperature,
        max_retries=1,
        timeout=settings.LLM_TIMEOUT,
    )

_PROVIDERS = {
    "groq":   (_groq,   lambda: settings.GROQ_API_KEY),
    "gemini": (_gemini, lambda: settings.GEMINI_API_KEY),
}


def load_llm(temperature: float = 0.0, providers: list = None):
    """providers : ordre de priorité (défaut : settings.LLM_PROVIDERS)."""
    order = providers if providers is not None else settings.LLM_PROVIDERS
    available = []
    for name in order:
        if name not in _PROVIDERS:
            logger.warning(f"Fournisseur LLM inconnu ignoré : {name}")
            continue
        build, get_key = _PROVIDERS[name]
        if not get_key():
            logger.warning(f"{name} : clé API absente, fournisseur ignoré")
            continue
        available.append((name, build(temperature)))

    if not available:
        raise RuntimeError(
            "Aucun fournisseur LLM configuré. "
            "Ajoute GROQ_API_KEY et/ou GEMINI_API_KEY dans le fichier .env"
        )

    logger.info("LLM : " + " -> ".join(name for name, _ in available))
    primary = available[0][1]
    backups = [model for _, model in available[1:]]
    return primary.with_fallbacks(backups) if backups else primary


def to_text(content) -> str:
    """Certains modèles renvoient une liste de blocs au lieu d'une chaîne."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)

if __name__ == "__main__":
    llm = load_llm()
    res = llm.invoke("Réponds uniquement par le mot : OK")
    print("Réponse :", to_text(res.content))