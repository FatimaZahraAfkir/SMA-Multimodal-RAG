from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def clean_query(query: str) -> str:
    """Retire l'historique/prompt système que l'ancienne interface collait dans la question."""
    return query.split("\nUtilisateur :")[-1].strip()[:200]


def _tavily(q: str, n: int) -> list:
    try:
        from tavily import TavilyClient
        data = TavilyClient(api_key=settings.TAVILY_API_KEY).search(q, max_results=n)
        return [{"title": r.get("title", ""), "body": r.get("content", ""), "url": r.get("url", "")}
                for r in data.get("results", [])]
    except Exception as e:
        logger.warning(f"Tavily indisponible : {e}")
        return []


def _ddg(q: str, n: int) -> list:
    try:
        from ddgs import DDGS
        for region in ("fr-fr", "wt-wt"):         
            results = DDGS().text(q, region=region, max_results=n)
            if results:
                return [{"title": r.get("title", ""), "body": r.get("body", ""), "url": r.get("href", "")}
                        for r in results]
            logger.warning(f"DuckDuckGo : 0 résultat (région {region})")
        return []
    except ImportError:
        logger.warning("Paquet `ddgs` non installé : pip install ddgs")
        return []
    except Exception as e:
        logger.warning(f"DuckDuckGo indisponible : {type(e).__name__}: {e}")
        return []


def web_search(query: str, max_results: int = 5) -> list:
    """-> [{"title", "body", "url"}, ...]"""
    q = clean_query(query)
    if not q:
        return []
    results = _tavily(q, max_results) if settings.TAVILY_API_KEY else []
    if not results:
        results = _ddg(q, max_results)
    if not results and not settings.TAVILY_API_KEY:
        logger.warning("Recherche web vide. Pour une recherche plus fiable, ajoute TAVILY_API_KEY dans .env (gratuit)")
    logger.info(f"Recherche web : {len(results)} résultats")
    return results


def format_results(results: list, limit: int = 4000) -> str:
    text = ""
    for i, r in enumerate(results, 1):
        text += f"{i}. {r['title']}\n{r['body']}\n(source: {r['url']})\n\n"
    return text[:limit]


def web_sources(results: list) -> list:
    """Format de source structuré pour l'interface."""
    return [{"type": "web", "title": r["title"], "url": r["url"], "snippet": r["body"][:200]}
            for r in results]