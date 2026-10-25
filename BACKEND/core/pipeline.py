import os
import re
import time
from typing import Iterator, Optional, Tuple

from config.settings import settings
from core.output_parser import parse_output
from core.router import route_query
from core.safety import detect_alert
from core.sessions import SessionStore
from models.llm_loader import to_text
from utils.logger import get_logger
from utils.prompt_templates import GENERATOR_PROMPT

logger = get_logger(__name__)

STRUCTURED_CONTEXT = ("[GRAPH]", "[TABLE]", "[TABLEAU]", "[IMAGE_B64]")
FILE_ERROR = re.compile(r"\[TEXT\]\s*(Impossible|Format non supporte)")
NO_INFO = re.compile(r"\[TEXT\]\s*Aucune information")       
RETRIEVAL_LABEL = {
    "text": "Recherche dans les documents médicaux",
    "graph": "Consultation du graphe médical",
    "image": "Analyse du fichier",
}
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")

ABSTAIN_BELOW_SCORE = float(os.getenv("ABSTAIN_BELOW_SCORE", "4"))

ABSTAIN_NO_SOURCE = ("Je ne trouve pas cette information dans mes documents ni dans les sources consultées, "
                     "et je préfère ne pas inventer une réponse médicale. Essayez de reformuler votre question "
                     "avec d'autres termes, ou consultez un professionnel de santé.")
ABSTAIN_REJECTED = ("Je n'ai pas pu produire une réponse suffisamment fiable à partir des sources trouvées : "
                    "la vérification automatique l'a rejetée. Je préfère ne pas risquer une information médicale "
                    "erronée. Essayez de reformuler votre question, ou consultez un professionnel de santé.")

CITATION_RULE = ("\n\nConsigne de citation : les extraits du contexte commencent par [Source : fichier, p. N]. "
                 "Quand tu utilises une information d'un extrait, indique sa source entre parenthèses à la fin de "
                 "la phrase, par exemple (Source : guide.pdf, p. 12). Cite uniquement les sources présentes dans "
                 "le contexte et n'invente jamais une source ou un numéro de page.")


SCOPE_PROMPT = """Tu tries les messages envoyés à un assistant médical (maladies, symptômes, médicaments, \
pharmacologie, anatomie, physiologie, soins, santé publique, santé mentale).

Conversation récente :
{history}

Nouveau message : {query}

Réponds par UN SEUL mot :
- MEDICAL : le message concerne la santé ou la médecine, ou prolonge naturellement la conversation médicale \
(ex. « et pour les enfants ? », « explique plus simplement »).
- TRANSFORMER : le message demande de traduire, reformuler, simplifier, résumer ou expliquer autrement \
la réponse précédente de l'assistant (ex. « traduis ça en arabe », « résume », « explique plus simplement »). \
Uniquement s'il existe une conversation récente.
- SALUTATION : bonjour, merci, au revoir, ou question sur ce que tu sais faire.
- HORS_SUJET : tout le reste (religion, cuisine, jeux, sport, politique, programmation, divertissement, etc.)."""

STANDALONE_PROMPT = """Conversation récente :
{history}

Nouvelle question : {query}

Réécris la nouvelle question pour qu'elle soit compréhensible seule, en reprenant le sujet médical de la \
conversation si la question y fait référence (« et pour les enfants ? », « et son dosage ? »). \
Si elle est déjà autonome, recopie-la telle quelle. Réponds uniquement par la question, en français."""

TRANSFORM_RULE = ("\n\nConsigne : l'utilisateur demande de transformer ta réponse précédente (traduction, "
                  "reformulation, simplification ou résumé). Applique cette demande à la dernière réponse de "
                  "l'assistant figurant dans la conversation ci-dessus, dans la langue demandée par l'utilisateur. "
                  "N'ajoute aucune information médicale nouvelle et ne change pas le sens, les doses ni les "
                  "avertissements.")

OUT_OF_SCOPE_REPLY = ("Je suis un assistant médical : je ne peux répondre qu'aux questions de santé "
                      "(maladies, symptômes, médicaments, anatomie, etc.). Votre question sort de ce domaine. "
                      "N'hésitez pas à me poser une question médicale ou à joindre un document de santé à analyser.")
SMALLTALK_REPLY = ("Bonjour ! Je suis l'assistant médical SMA. Posez-moi une question sur une maladie, un symptôme "
                   "ou un médicament, ou joignez un document de santé à analyser. "
                   "Mes réponses sont à but éducatif et ne remplacent pas l'avis d'un professionnel de santé.")


def _cap_image_confidence(conf):
    """La jauge mesure la cohérence entre la réponse et l'analyse du modèle de vision ; elle ne peut pas
    vérifier que ce modèle a bien lu l'image. Pour une image on plafonne donc à « Moyenne » (60)."""
    if not conf or conf.get("score") is None:
        return conf
    c = dict(conf)
    c["reasons"] = ["image interprétée par un modèle de vision : non vérifiée"] + list(c.get("reasons", []))
    c["score"] = min(c["score"], 60)
    if c.get("level") == "high":
        c["level"], c["label"] = "medium", "Moyenne"
    return c


UNVALIDATED_WARNING = ("Cette réponse n'a pas pu être validée automatiquement. "
                       "Vérifiez-la auprès d'un professionnel de santé.")


class _TokenGate:
    """Décide d'après le début de la réponse s'il faut streamer du texte ([TEXT] ou sans balise)
    ou bufferiser (TABLE / GRAPH / IMAGE : le JSON ne doit pas défiler à l'écran)."""

    BUFFERED = ("[TABLE]", "[TABLEAU]", "[GRAPH]", "[IMAGE]", "[IMAGE_B64]")

    def __init__(self):
        self.buf, self.mode = "", None      

    def feed(self, chunk: str) -> str:
        if self.mode == "text":
            return chunk
        if self.mode == "buffer":
            return ""
        self.buf += chunk
        s = self.buf.lstrip()
        if not s:
            return ""
        if s.startswith("["):
            end = s.find("]")
            if end == -1:
                if len(s) < 14:
                    return ""                      
                self.mode = "text"
                return self.buf
            tag = s[:end + 1]
            if tag == "[TEXT]":
                self.mode = "text"
                return s[end + 1:].lstrip(" ")
            if tag in self.BUFFERED:
                self.mode = "buffer"
                return ""
            self.mode = "text"                     
            return self.buf
        self.mode = "text"
        return self.buf

    def finish(self) -> str:
        if self.mode is None and self.buf:
            self.mode = "text"
            return self.buf
        return ""


def _status(step: str, state: str, label: str, **extra) -> Tuple[str, dict]:
    return "status", {"step": step, "state": state, "label": label, **extra}


def _abstention(message: str, route: str, sources: list, **extra) -> dict:
    """Résultat structuré d'une abstention (aucune réponse médicale inventée)."""
    return {"output": {"type": "text", "text": message, "table": None, "graph": None, "image_b64": None},
            "validated": True, "verified": False, "abstained": True,
            "confidence": {"score": None, "label": "Non évaluée", "level": "n/a"},
            "sources": sources, "route": route, "regenerated": False, **extra}


class Pipeline:
    def __init__(self, text_agent, graph_agent, image_agent, validator, llm,
                 router_llm=None, sessions: Optional[SessionStore] = None):
        self.agents = {"text": text_agent, "graph": graph_agent, "image": image_agent}
        self.validator, self.llm, self.router_llm = validator, llm, router_llm
        self.sessions = sessions or SessionStore(ttl_s=settings.SESSION_TTL_S)

    # ── utilitaires ──────────────────────────────────────────────────────────
    def _scope(self, query: str, history: str) -> str:
        """'medical' | 'greeting' | 'off_topic'. En cas de doute ou d'erreur : 'medical' (on ne bloque pas)."""
        if self.router_llm is None:
            return "medical"
        try:
            out = to_text(self.router_llm.invoke(SCOPE_PROMPT.format(
                history=history[-800:] or "(aucune)", query=query[:500])).content)
        except Exception as e:
            logger.warning(f"Contrôle du domaine indisponible ({type(e).__name__}) : question acceptée")
            return "medical"
        out = out.strip().strip("*`'\". ").upper()
        logger.info(f"Domaine de la question : {out[:15]}")
        if out.startswith("HORS"):
            return "off_topic"
        if out.startswith("SALUT"):
            return "greeting"
        if out.startswith("TRANSF") and history.strip():
            return "transform"
        return "medical"

    def _standalone(self, query: str, history: str) -> str:
        """Réécrit une question de suivi (« et pour les enfants ? ») en question autonome pour la recherche."""
        if not history.strip() or self.router_llm is None or len(query.split()) > 14:
            return query
        try:
            out = to_text(self.router_llm.invoke(STANDALONE_PROMPT.format(
                history=history[-1500:], query=query[:500])).content).strip().strip('"')
        except Exception as e:
            logger.warning(f"Réécriture de la question indisponible ({type(e).__name__})")
            return query
        if not out or len(out) > 400:
            return query
        logger.info(f"Question de recherche : {out!r}")
        return out

    def health(self) -> dict:
        retriever = getattr(self.agents.get("text"), "retriever", None)
        return {"llm_providers": [p for p in settings.LLM_PROVIDERS
                                  if (settings.GROQ_API_KEY if p == "groq" else settings.GEMINI_API_KEY)],
                "faiss_index_loaded": getattr(retriever, "db", None) is not None,
                "active_sessions": len(self.sessions)}

    @staticmethod
    def _prompt(query: str, context: str, history: str, feedback: str, route: str = "text",
                transform: bool = False) -> str:
        if transform:
            return GENERATOR_PROMPT.format(
                context=f"Conversation récente :\n{history[-3500:]}\n", query=query + TRANSFORM_RULE)
        ctx = f"Conversation récente :\n{history}\n\n" if history else ""
        if route == "image":
        
            ctx += ("Analyse du fichier que l'utilisateur vient de joindre (réalisée par un modèle de vision ; "
                    "le fichier a bien été fourni, ne dis jamais le contraire). Pour une image médicale : décris "
                    "seulement, ne conclus JAMAIS qu'un examen est normal ou anormal et ne pose aucun diagnostic :\n")
        else:
            ctx += "Contexte récupéré :\n"
        ctx += context[:5000] or "(aucun)"
        q = (query + "\n\nConsigne : reponds precisement a cette question. Si le contexte ne permet pas "
                     "d'y repondre, dis-le clairement au lieu de resumer le contexte.")
        if route == "text":
            q += CITATION_RULE
        if feedback and feedback.strip().lower() != "aucune":
            q += f"\n\n[INSTRUCTION DE CORRECTION] : {feedback}"
        return GENERATOR_PROMPT.format(context=ctx, query=q)

    def _summary_for_history(self, output: dict) -> str:
        if output["type"] == "graph":
            return "(graphe affiché) " + output["text"]
        if output["type"] == "table":
            return "(tableau affiché) " + output["text"]
        return output["text"]

    def run(self, query: str, session_id: str, file=None, forced_route: Optional[str] = None
            ) -> Iterator[Tuple[str, dict]]:
        t0 = time.monotonic()

        alert = detect_alert(query)
        if alert and alert["level"] == "support":
            yield "meta", {"session_id": session_id, "route": "safety", "reason": "message de soutien"}
            yield "result", {"output": {"type": "text", "text": alert["message"], "table": None,
                                        "graph": None, "image_b64": None},
                             "safety": "support", "validated": True, "verified": False,
                             "confidence": {"score": None, "label": "Non évaluée", "level": "n/a"},
                             "sources": [], "route": "safety", "regenerated": False}
            yield "done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)}
            return
        if alert:
            yield "alert", alert

        transform, history = False, self.sessions.history_text(session_id)
        if file is None and not alert:
            scope = self._scope(query, history)
            transform = scope == "transform"
            if scope not in ("medical", "transform"):
                off = scope == "off_topic"
                msg = OUT_OF_SCOPE_REPLY if off else SMALLTALK_REPLY
                logger.info(f"Question hors domaine médical ({scope}) -> réponse directe")
                yield "meta", {"session_id": session_id, "route": "out_of_scope",
                               "reason": "hors du domaine médical" if off else "salutation"}
                yield "result", _abstention(msg, "out_of_scope", [], abstained=False, out_of_scope=off)
                yield "done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)}
                return

        if transform:

            route, context, sources = "text", history, []
            yield "meta", {"session_id": session_id, "route": route, "reason": "reformulation de la réponse précédente"}
            yield _status("routing", "done", "Analyse de la question", detail="reformulation")
        else:

            search_query = query if file is not None else self._standalone(query, history)
            yield _status("routing", "start", "Analyse de la question")
            route, reason = route_query(query, file is not None, forced_route, self.router_llm)
            yield "meta", {"session_id": session_id, "route": route, "reason": reason}
            yield _status("routing", "done", "Analyse de la question", detail=reason)

            label = RETRIEVAL_LABEL[route]
            yield _status("retrieval", "start", label)
            try:
                out = self.agents[route].run({"query": search_query, "image": file, "iterations": 0})
                context, sources = out.get("context", "") or "", out.get("sources", []) or []
            except Exception as e:
                logger.exception("Erreur agent de récupération")
                context, sources = "", []
                yield "warning", {"message": f"La récupération a échoué ({type(e).__name__}). "
                                             "La réponse sera peu fiable."}
            if route == "image":
                logger.info(f"Analyse du fichier (début) : {context[:200]!r}")
            yield _status("retrieval", "done", label, detail=f"{len(sources)} source(s)")
            if sources:
                yield "sources", {"sources": sources}


        if route == "image" and FILE_ERROR.match(context):
            yield "error", {"message": re.sub(r"^\[TEXT\]\s*", "", context)}
            yield "done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)}
            return

        if context.lstrip().startswith(STRUCTURED_CONTEXT):
            output = parse_output(context)
            yield "result", {"output": output, "validated": True, "verified": False,
                             "confidence": {"score": None, "label": "Non évaluée", "level": "n/a"},
                             "sources": sources, "route": route, "regenerated": False}
            self.sessions.add_turn(session_id, query, self._summary_for_history(output))
            yield "done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)}
            return

        if route == "graph" and not sources:
            msg = ("Je n'ai trouvé aucune donnée fiable pour construire ce graphe : le graphe médical n'est pas "
                   "disponible et la recherche web n'a rien renvoyé. Je préfère ne pas en inventer un.")
            yield "result", {"output": {"type": "text", "text": msg, "table": None, "graph": None,
                                        "image_b64": None},
                             "validated": True, "verified": False, "no_data": True,
                             "confidence": {"score": None, "label": "Non évaluée", "level": "n/a"},
                             "sources": [], "route": route, "regenerated": False}
            yield "done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)}
            return
        
        if route == "text" and (not context.strip() or NO_INFO.match(context.strip())):
            logger.info("Aucune source trouvée -> abstention")
            result = _abstention(ABSTAIN_NO_SOURCE, route, sources, no_data=True)
            yield "result", result
            self.sessions.add_turn(session_id, query, ABSTAIN_NO_SOURCE)
            yield "done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)}
            return

        feedback, regen = "", 0
        while True:
            step, step_label = ("regeneration", "Nouvelle rédaction (correction)") if regen else \
                               ("generation", "Rédaction de la réponse")
            yield _status(step, "start", step_label)
            raw, gate = "", _TokenGate()
            try:
                for chunk in self.llm.stream(self._prompt(query, context, history, feedback, route, transform)):
                    piece = to_text(chunk.content)
                    raw += piece
                    shown = gate.feed(piece)
                    if shown:
                        yield "token", {"text": shown}
                tail = gate.finish()
                if tail:
                    yield "token", {"text": tail}
            except Exception as e:
                logger.exception("Erreur de génération")
                yield "error", {"message": f"La génération a échoué : {type(e).__name__}. Réessayez dans un instant."}
                yield "done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)}
                return
            if not raw.strip():
                yield "error", {"message": "Le modèle a renvoyé une réponse vide."}
                yield "done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)}
                return
            yield _status(step, "done", step_label)

            yield _status("validation", "start", "Vérification de la réponse")
            try:
                if transform:   
                    v = {"is_valid": True, "metrics": {"verified": False}, "validation_feedback": "",
                         "confidence": {"score": None, "label": "Non évaluée", "level": "n/a"}}
                else:
                    v = self.validator.validate({"query": query, "response": raw, "context": context, "iterations": regen})
            except Exception as e:
                logger.exception("Erreur de validation")
                v = {"is_valid": True, "metrics": {"verified": False},
                     "validation_feedback": "",
                     "confidence": {"score": None, "label": "Non vérifiée", "level": "unverified"}}
            yield _status("validation", "done", "Vérification de la réponse",
                          detail="valide" if v["is_valid"] else "à corriger")

            if v["is_valid"] or regen >= settings.MAX_REGENERATIONS:
                break
            regen += 1
            feedback = v.get("validation_feedback", "")
            yield "reset", {"reason": "correction automatique"}

        if (route == "text" and not v["is_valid"]
                and v.get("metrics", {}).get("verified")
                and v.get("validation_score", 10) < ABSTAIN_BELOW_SCORE):
            logger.info(f"Réponse rejetée (score {v.get('validation_score')} < {ABSTAIN_BELOW_SCORE}) -> abstention")
            yield "reset", {"reason": "abstention"}
            result = _abstention(ABSTAIN_REJECTED, route, sources, regenerated=regen > 0)
            result["regenerated"] = regen > 0
            yield "result", result
            self.sessions.add_turn(session_id, query, ABSTAIN_REJECTED)
            yield "done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)}
            return

        conf = v.get("confidence")
        if route == "image" and str(getattr(file, "name", "")).lower().endswith(IMAGE_EXT):
            conf = _cap_image_confidence(conf)
        output = parse_output(raw)
        result = {"output": output, "validated": bool(v["is_valid"]),
                  "verified": bool(v.get("metrics", {}).get("verified")),
                  "confidence": conf, "metrics": v.get("metrics"),
                  "sources": sources, "route": route, "regenerated": regen > 0}
        if not v["is_valid"]:
            result["warning"] = UNVALIDATED_WARNING
        if transform:
            result["transformed"] = True
            result["note"] = ("Traduction ou reformulation automatique de la réponse précédente : "
                              "vérifiez les doses et les consignes avec la réponse d'origine.")
        yield "result", result
        self.sessions.add_turn(session_id, query, self._summary_for_history(output))
        yield "done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)}


def build_default_pipeline() -> Pipeline:
    """Assemble les vrais agents (importés ici pour garder le module testable sans dépendances lourdes)."""
    from agents.RAG_agent import TextAgent
    from agents.graph_agent import GraphAgent
    from agents.multimodal_agent import ImageAgent
    from agents.validation_agent import ValidationAgent
    from models.llm_loader import load_llm

    return Pipeline(
        text_agent=TextAgent(), graph_agent=GraphAgent(), image_agent=ImageAgent(),
        validator=ValidationAgent(),
        llm=load_llm(settings.GENERATION_TEMPERATURE),
        router_llm=load_llm(0.0),
    )