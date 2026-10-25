import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrievers.text_retriever import TextRetriever
from models.llm_loader import to_text
from utils.logger import get_logger
from utils.web_search import web_search, format_results, web_sources

logger = get_logger(__name__)

MIN_CONTEXT_LENGTH = 100

MAX_DISTANCE = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "1.0"))

RELEVANCE_PROMPT = """Question de l'utilisateur : {query}

Extraits de documents :
{excerpts}

Ces extraits contiennent-ils des informations qui permettent de repondre directement a la question ?
Reponds uniquement par : oui ou non"""


def _doc_meta(d) -> tuple:
    """(titre du fichier, numéro de page ou None) à partir des métadonnées d'un extrait."""
    meta = getattr(d, "metadata", None) or {}
    title = os.path.basename(str(meta.get("source", "document")))
    page = meta.get("page")
    page = page + 1 if isinstance(page, int) else None      
    return title, page


def _doc_sources(docs: list) -> list:
    sources, seen = [], set()
    for d in docs:
        title, page = _doc_meta(d)
        key = (title, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"type": "document", "title": title, "page": page,
                        "snippet": d.page_content[:200]})
    return sources


def _format_docs(docs: list) -> str:
    """Contexte donné au LLM : chaque extrait est précédé de sa source (fichier + page),
    pour que la réponse puisse citer précisément d'où vient chaque information."""
    blocks = []
    for d in docs:
        title, page = _doc_meta(d)
        label = f"{title}, p. {page}" if page else title
        blocks.append(f"[Source : {label}]\n{d.page_content}")
    return "\n\n".join(blocks)


class TextAgent:
    def __init__(self, grader=None):
        self._grader = grader
        try:
            self.retriever = TextRetriever()
        except Exception as e:
            logger.warning(f"TextRetriever non disponible : {e}")
            self.retriever = None

    def _is_relevant(self, query: str, context: str) -> bool:
        try:
            if self._grader is None:
                from models.llm_loader import load_llm
                self._grader = load_llm(0.0)
            out = to_text(self._grader.invoke(
                RELEVANCE_PROMPT.format(query=query[:500], excerpts=context[:1500])).content)
            return out.strip().strip("*`'\". ").lower().startswith("oui")
        except Exception as e:
            logger.warning(f"Controle de pertinence indisponible ({e}) : extraits conserves")
            return True

    def run(self, state):
        query = state.get("query", "")

        faiss_context, faiss_len, sources = "", 0, []
        try:
            if self.retriever and self.retriever.db is not None:
                pairs = self.retriever.db.similarity_search_with_score(query, k=4)
                logger.info("FAISS distances : " + ", ".join(f"{sc:.2f}" for _, sc in pairs)
                            + f" (seuil {MAX_DISTANCE})")
                docs = [d for d, sc in pairs if sc <= MAX_DISTANCE]
                faiss_context = _format_docs(docs)                      
                faiss_len = sum(len(d.page_content.strip()) for d in docs)   
                sources = _doc_sources(docs)
                logger.info(f"FAISS : {len(docs)}/{len(pairs)} documents pertinents ({faiss_len} chars)")
            else:
                logger.warning("FAISS non disponible -> recherche web directe")
        except Exception as e:
            logger.warning(f"Erreur FAISS : {e} -> recherche web")

        if faiss_context.strip() and not self._is_relevant(query, faiss_context):
            logger.info("FAISS : extraits hors sujet -> ignores, recherche web")
            faiss_context, faiss_len, sources = "", 0, []

        if faiss_len < MIN_CONTEXT_LENGTH:
            logger.info(f"FAISS insuffisant -> recherche web pour : '{query[:60]}'")
            results = web_search(query)
            web_context = format_results(results, limit=4000)
            if web_context:
                if faiss_context.strip():
                    ctx = f"[SOURCE FAISS]\n{faiss_context}\n\n[SOURCE WEB]\n{web_context}"
                else:
                    ctx = f"[SOURCE WEB]\n{web_context}"
                return {"context": ctx, "sources": sources + web_sources(results)}
            
            return {"context": faiss_context or "[TEXT] Aucune information trouvee.", "sources": sources}

        return {"context": faiss_context, "sources": sources}