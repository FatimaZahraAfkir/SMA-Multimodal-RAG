import os
import re
import sys
import json
import unicodedata
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrievers.graph_retriever import GraphRetriever
from models.llm_loader import load_llm, to_text
from utils.logger import get_logger
from utils.web_search import web_search, web_sources

logger = get_logger(__name__)

GRAPH_KEYWORDS = [
    "graphe", "graph", "visualise", "visualisation", "schema",
    "sous forme", "montre", "affiche", "relations", "lien", "connexion"
]

MIN_GROUNDED_RELATIONS = 3     

def _tokens(text: str) -> list:
    t = unicodedata.normalize("NFKD", (text or "").lower().replace("’", "'"))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return [w for w in re.findall(r"\w+", t) if len(w) > 2]


def _is_grounded(preuve: str, source_text: str, threshold: float = 0.7) -> bool:
    """Vrai si l'extrait cité comme preuve se retrouve (à ~70 %) dans le texte réellement lu."""
    toks = _tokens(preuve)
    if len(toks) < 3:
        return False
    pool = set(_tokens(source_text))
    return sum(1 for w in toks if w in pool) / len(toks) >= threshold


class GraphAgent:
    def __init__(self):
        try:
            self.retriever = GraphRetriever()
        except Exception as e:
            logger.warning(f"GraphRetriever indisponible : {e}")
            self.retriever = None
        self.llm = load_llm()

    def run(self, state: dict) -> dict:
        query = state.get("query", "")
        wants_graph = any(kw in query.lower() for kw in GRAPH_KEYWORDS)

        entity = self._extract_entity(query)
        logger.info(f"Entite extraite : '{entity}'")

        triplets, stats = self._neo4j_lookup(entity)

        if triplets:
            if wants_graph:
                context = self._build_graph_context(triplets, entity, stats)
            else:
                relations = "\n".join([f"  {t}" for t in triplets])
                context = (
                    f"Relations du graphe pour '{entity}' :\n{relations}\n\n"
                    f"[Graphe : {stats['nodes']} noeuds, {stats['edges']} relations]"
                )
            return {"context": context,
                    "sources": [{"type": "graph", "title": "Graphe médical (Neo4j)"}]}

        logger.info(f"Neo4j : 0 resultats -> recherche web pour '{entity}'")
        context, results = self._web_search_fallback(entity, query, wants_graph)
        return {"context": context, "sources": web_sources(results)}

    def _neo4j_lookup(self, entity: str):
        if self.retriever is None:
            return [], None
        try:
            triplets = self.retriever.retrieve(entity) or self._fuzzy_search(entity)
            stats = self.retriever.stats() if triplets else None
            return triplets, stats
        except Exception as e:
            logger.error(f"Neo4j indisponible : {e} -> recherche web")
            return [], None

    def _build_graph_context(self, triplets: list, entity: str, stats: dict) -> str:
        relations = []
        for t in triplets[:15]:
            parts = [p.strip() for p in t.split("->")]   # "src -> rel -> dst"
            if len(parts) == 3:
                relations.append({
                    "source": parts[0],
                    "cible":  parts[2],
                    "type":   parts[1].upper().replace(" ", "_")
                })
        if not relations:
            return f"[TEXT] Aucune relation formatee pour '{entity}'."

        graph_json = json.dumps({"relations": relations}, ensure_ascii=False)
        return (
            f"[GRAPH] {graph_json} [/GRAPH]\n\n"
            f"Source : Neo4j ({stats['nodes']} noeuds, {stats['edges']} relations)"
        )

    def _web_search_fallback(self, entity: str, original_query: str, wants_graph: bool):
        """-> (contexte, résultats web)"""
        results = web_search(f"{entity} maladie symptômes traitement")
        if not results:
            return f"[TEXT] Aucune information trouvee pour '{entity}' (graphe medical et web).", []

        web_text = "\n".join(f"{r['title']}: {r['body']}" for r in results)
        if wants_graph:
            return self._extract_relations_as_graph(entity, web_text, original_query), results
        return f"[TEXT] [SOURCE WEB]\n\n{web_text[:3000]}", results

    def _extract_relations_as_graph(self, entity: str, web_text: str, query: str) -> str:
        seen_text = web_text[:3000]
        prompt = f"""A partir de ce texte medical, extrais les relations entre concepts medicaux.
Retourne UNIQUEMENT ce JSON, sans texte avant ou apres :
{{"relations":[{{"source":"ConceptA","cible":"ConceptB","type":"TRAITE_PAR","preuve":"extrait copie du texte"}}]}}

SENS DES RELATIONS (source -> cible), a respecter strictement :
- TRAITE_PAR        : source = maladie,            cible = traitement ou medicament
- SYMPTOME_DE       : source = symptome,           cible = maladie
- CAUSE             : source = cause,              cible = effet ou maladie
- FACTEUR_DE_RISQUE : source = facteur de risque,  cible = maladie
- COMPLICATION_DE   : source = complication,       cible = maladie
- DIAGNOSTIQUE_PAR  : source = maladie,            cible = examen
- CONTRE_INDIQUE    : source = medicament,         cible = situation ou maladie
- EST_UN            : source = cas particulier,    cible = categorie (ex. diabete de type 1 EST_UN diabete)

Regles :
- Maximum 10 relations, uniquement celles ecrites dans le texte (n'invente rien)
- source et cible : 1 a 4 mots
- preuve : court extrait (12 mots maximum) COPIE du texte qui justifie la relation
- Inclure '{entity}' comme noeud principal

Texte :
{seen_text}

JSON:"""
        try:
            raw = to_text(self.llm.invoke(prompt).content).strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                relations = json.loads(match.group()).get("relations", [])
                kept = [
                    {"source": r["source"], "cible": r["cible"], "type": r["type"]}
                    for r in relations
                    if isinstance(r, dict) and all(k in r for k in ("source", "cible", "type"))
                    and _is_grounded(str(r.get("preuve", "")), seen_text)
                ]
                logger.info(f"Relations : {len(kept)} gardees, {len(relations) - len(kept)} rejetees "
                            f"(preuve absente du texte)")
                if len(kept) >= MIN_GROUNDED_RELATIONS:
                    graph_json = json.dumps({"relations": kept}, ensure_ascii=False)
                    return (
                        f"[GRAPH] {graph_json} [/GRAPH]\n\n"
                        f"Relations extraites automatiquement de pages web ({len(kept)} retrouvees dans "
                        f"les textes sources). A confirmer avec les sources ci-dessous."
                    )
        except Exception as e:
            logger.error(f"Erreur extraction relations web : {e}")
        return f"[TEXT] [SOURCE WEB]\n\n{web_text[:2000]}"

    def _extract_entity(self, query: str) -> str:
        try:
            prompt = (
                "Extrais le concept medical principal (maladie, symptome, medicament, organe) de cette question.\n"
                "Reponds avec UNIQUEMENT 1 a 3 mots, sans ponctuation, sans explication.\n"
                "Exemples :\n"
                "  Question : 'quelles sont les relations du diabete ?'\n"
                "  Reponse  : diabete\n"
                "  Question : 'montre le lien entre hypertension et insuffisance renale'\n"
                "  Reponse  : hypertension\n\n"
                f"Question : {query}\n"
                "Reponse (1 a 3 mots) :"
            )
            raw = to_text(self.llm.invoke(prompt).content).strip()
            entity = raw.strip('"\'.,!?\n ').split("\n")[0].strip()
            entity = " ".join(entity.split()[:3])
            return entity if entity else self._fallback_entity(query)
        except Exception:
            return self._fallback_entity(query)

    def _fallback_entity(self, query: str) -> str:
        stop_words = {
            "quelles", "sont", "les", "relations", "de", "du", "la", "le",
            "un", "une", "des", "et", "ou", "comment", "fonctionne", "explique",
            "montre", "donne", "moi", "liste", "dans", "pour", "avec", "par",
            "sur", "que", "qui", "quoi", "est", "ce", "entre", "donner", "sous",
            "forme", "graphe", "chacun", "leurs", "lien", "visualise"
        }
        words = query.lower().strip("?.,!").split()
        significant = [w for w in words if w not in stop_words and len(w) > 2]
        if significant:
            return " ".join(significant[:2]) if len(significant) >= 2 else significant[0]
        return words[-1] if words else query

    def _fuzzy_search(self, entity: str) -> list:
        all_triplets, seen = [], set()
        for keyword in [entity] + [w for w in entity.split() if len(w) > 2]:
            for t in self.retriever.db.query_contains(keyword):
                if t not in seen:
                    seen.add(t)
                    all_triplets.append(t)
        logger.info(f"Fuzzy '{entity}' -> {len(all_triplets)} triplets")
        return all_triplets[:20]