"""
ValidationAgent léger : plus de torch / transformers / BERTScore.

1. Toute réponse écrite par le LLM est contrôlée, y compris un [GRAPH] ou [TABLE] généré.
   (Les données lues directement dans Neo4j ou un fichier ne passent pas par ici : voir pipeline.)
2. Juge LLM (VALIDATION_PROMPT) : fidélité, pertinence, cohérence, sécurité.
   Idéalement un autre modèle que le générateur (settings.JUDGE_PROVIDERS).
3. Règles de sécurité déterministes (diagnostic définitif, langue, avertissement...).
4. ROUGE-L (précision) calculé en Python pur : signal de « grounding » bon marché.

Si le juge est indisponible ou renvoie du JSON illisible, la réponse n'est PAS déclarée
valide à la légère : on se rabat sur les règles et on la marque « non vérifiée ».
"""
import os
import re
import sys
import unicodedata

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from evaluation.confidence_score import compute_confidence, infer_source_level
from models.llm_loader import load_llm, to_text
from utils.helpers import safe_json_parse
from utils.logger import get_logger
from utils.prompt_templates import VALIDATION_PROMPT

logger = get_logger(__name__)

STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "en", "au", "aux", "ce",
    "ces", "est", "sont", "que", "qui", "dans", "pour", "par", "sur", "avec", "ne", "pas",
    "se", "sa", "son", "ses", "il", "elle", "ils", "plus", "peut", "etre", "a", "d", "l",
}
FR_WORDS = ["le", "la", "les", "de", "du", "et", "est", "que", "une", "pour", "dans", "avec", "sur", "par"]
DISCLAIMER_HINTS = ["consultez", "professionnel de sante", "medecin", "educatif", "pharmacien"]
DANGEROUS = ["vous avez certainement", "vous avez definitivement", "c'est forcement",
             "diagnostic certain", "vous souffrez de", "vous etes atteint de",
             "aucun signe evident d'anomalie", "aucune anomalie visible", "aucune anomalie detectee",
             "pas d'anomalie structurale", "ne montre aucune anomalie", "examen normal",
             "irm normale", "scanner normal", "radiographie normale"]


def _norm(text: str) -> str:
    """Minuscules + sans accents (fièvre == fievre)."""
    text = (text or "").lower().replace("’", "'")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def _tokens(text: str) -> list:
    return [t for t in re.findall(r"\w+", _norm(text)) if t not in STOPWORDS and len(t) > 1]


def _lcs_len(a: list, b: list) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l_precision(context: str, response: str):
    """Part des mots de la réponse retrouvés dans le contexte, dans le même ordre (LCS).
    Implémentation maison : le tokenizer de rouge-score casse les mots accentués."""
    ctx, resp = _tokens(context)[:1200], _tokens(response)[:300]
    if not ctx or not resp:
        return None
    return _lcs_len(ctx, resp) / len(resp)


def check_rules(response: str) -> dict:
    text, toks = _norm(response), set(_tokens(response)) | set(re.findall(r"\w+", _norm(response)))
    r = {
        "has_disclaimer": any(k in text for k in DISCLAIMER_HINTS) or "⚠️" in response,
        "has_format": bool(re.match(r"\s*\[(TEXT|TABLE|GRAPH|IMAGE)\]", response)),
        "is_french": sum(1 for w in FR_WORDS if w in toks) >= 5,
        "substantial": len(response.strip()) > 100,
        "dangerous": [p for p in DANGEROUS if p in text],
    }
    r["score"] = (0.30 * r["has_disclaimer"] + 0.15 * r["has_format"] + 0.15 * r["is_french"]
                  + 0.15 * r["substantial"] + 0.25 * (not r["dangerous"]))
    return r


class ValidationAgent:
    def __init__(self):
        self._judge = None
        logger.info("ValidationAgent léger initialisé (juge LLM + règles + ROUGE-L)")

    def _get_judge(self):
        if self._judge is None:
            self._judge = load_llm(0.0, providers=settings.JUDGE_PROVIDERS)
        return self._judge

    def _run_judge(self, query: str, context: str, response: str):
        block = context.strip()[:3000] or "(aucun contexte récupéré)"
        if query.strip():
            block = (f"QUESTION POSEE PAR L'UTILISATEUR : {query.strip()[:500]}\n"
                     f"(Evalue en priorité si la réponse répond bien à CETTE question : "
                     f"une réponse hors sujet est INVALID.)\n\nCONTEXTE RECUPERE :\n{block}")
        prompt = VALIDATION_PROMPT.format(
            context_sources=block,
            generated_answer=response[:2000],
        )
        try:
            raw = to_text(self._get_judge().invoke(prompt).content)
        except Exception as e:
            logger.warning(f"Juge LLM indisponible : {e}")
            return None

        data = safe_json_parse(raw)
        if not isinstance(data, dict) or "parsing_error" in (data.get("errors") or []):
            logger.warning("Juge LLM : JSON illisible -> réponse non vérifiée")
            return None
        try:
            score = max(0.0, min(10.0, float(data.get("score", 0))))
        except (TypeError, ValueError):
            score = 0.0
        return {
            "status": str(data.get("status", "")).upper(),
            "score": score,
            "metrics": data.get("metrics") if isinstance(data.get("metrics"), dict) else {},
            "correction": str(data.get("correction_instruction", "aucune")),
        }

    def validate(self, state: dict) -> dict:
        response = (state.get("response") or "").strip()
        context = state.get("context") or ""
        iters = state.get("iterations", 0)

        clean = re.sub(r"\[/?(TEXT|TABLE|GRAPH|IMAGE)\]", "", response)
        rules = check_rules(response)
        rouge = rouge_l_precision(context, clean) if len(context.strip()) >= 100 else None
        judge = self._run_judge(state.get("query") or "", context, response)

        if judge:
            is_valid = (judge["status"] == "VALID"
                        and judge["score"] >= settings.VALIDATION_MIN_SCORE
                        and not rules["dangerous"])
            score_10 = judge["score"]
        else:
            is_valid = rules["score"] >= 0.6 and not rules["dangerous"]
            score_10 = round(rules["score"] * 10, 1)

        fixes = []
        if judge and judge["correction"].strip().lower() not in ("", "aucune"):
            fixes.append(judge["correction"])
        if rules["dangerous"]:
            fixes.append("Ne formule aucun diagnostic définitif et ne conclus jamais qu'un examen d'imagerie est normal ou anormal : décris seulement, reste prudent.")
        if not rules["has_format"]:
            fixes.append("Commence la réponse par [TEXT], [TABLE] ou [GRAPH].")
        try:
            low_safety = float((judge or {}).get("metrics", {}).get("securite", 10)) < 7
        except (TypeError, ValueError):
            low_safety = False
        if not rules["has_disclaimer"] and (judge is None or low_safety):
            fixes.append("Ajoute : ⚠️ Cette information est à but éducatif. Consultez un professionnel de santé.")
        feedback = " | ".join(fixes) or "Reste fidèle au contexte fourni et ne rien inventer."

        confidence = compute_confidence(judge=judge, rouge=rouge, rules=rules["score"],
                                        source_level=infer_source_level(context))

        logger.info(f"Validation : {'VALID' if is_valid else 'INVALID'} | juge={'oui' if judge else 'NON'} | question={'oui' if (state.get('query') or '').strip() else 'NON'} "
                    f"| score={score_10} | rouge={None if rouge is None else round(rouge, 2)} "
                    f"| règles={rules['score']:.2f} | confiance={confidence['score']} ({confidence['label']})")

        return {
            "is_valid": is_valid,
            "validation_score": score_10,
            "iterations": iters + 1,
            "validation_feedback": "" if is_valid else feedback,
            "confidence": confidence,
            "metrics": {
                "judge": judge["metrics"] if judge else None,
                "judge_status": judge["status"] if judge else None,
                "rouge_l_precision": None if rouge is None else round(rouge, 3),
                "rules": round(rules["score"], 2),
                "dangerous_phrases": rules["dangerous"],
                "verified": judge is not None,
            },
        }