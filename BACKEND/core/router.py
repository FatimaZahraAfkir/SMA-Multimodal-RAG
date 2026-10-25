from core.textutil import norm
from models.llm_loader import to_text

VALID_ROUTES = ("text", "graph", "image")
STRONG_GRAPH = ["graphe", "graph", "neo4j", "visualise", "schema des relations"]
WEAK_GRAPH = ["relation", "lien", "connexion", "associe", "interaction"]

ROUTER_PROMPT = """Classe cette question d'utilisateur en un seul mot.
- graph : il demande explicitement les relations / liens entre entités médicales (maladies, traitements, médicaments) ou une visualisation.
- text  : toute autre question médicale (symptômes, médicament, explication, posologie...).
Réponds uniquement par : graph ou text

Question : {query}
Réponse :"""


def route_query(query: str, has_file: bool, forced=None, llm=None):
    """-> (route, raison)"""
    if forced in VALID_ROUTES:
        return forced, "choix de l'utilisateur"
    if has_file:
        return "image", "fichier joint"

    q = norm(query)
    if any(k in q for k in STRONG_GRAPH):
        return "graph", "demande de graphe"
    if llm is not None and any(k in q for k in WEAK_GRAPH):
        try:
            label = to_text(llm.invoke(ROUTER_PROMPT.format(query=query[:500])).content).strip().lower()
            return ("graph" if label.startswith("graph") else "text"), "classifieur LLM"
        except Exception:
            return "text", "repli : classifieur indisponible"
    return "text", "question médicale (RAG)"
