import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm_loader import load_llm
from utils.prompt_templates import GENERATOR_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

PASSTHROUGH_TAGS = ["[GRAPH]", "[TABLE]", "[IMAGE_B64]"]


class GeneratorAgent:
    def __init__(self):
        self.llm = load_llm()

    def generate(self, state):
        query    = state.get("query",    "")
        context  = state.get("context",  "")
        response = state.get("response", "")

        if response and response.strip().startswith("["):
            logger.info("Passthrough : response deja produite par agent expert")
            return {"response": response}

        for tag in PASSTHROUGH_TAGS:
            closing_tag = tag.replace("[", "[/")
            if tag in context and closing_tag in context:
                start     = context.index(tag)
                end       = context.index(closing_tag) + len(closing_tag)
                formatted = context[start:end].strip()
                logger.info(f"Passthrough : {tag} transmis directement")
                return {"response": formatted}

        correction = (
            state.get("validation_feedback", "") or
            state.get("correction_instruction", "")
        )
        if correction and correction not in ("aucune", "Erreur technique de validation"):
            query = f"{query}\n\n[INSTRUCTION DE CORRECTION] : {correction}"
            logger.info(f"Regeneration avec correction : {correction[:80]}")

        prompt = GENERATOR_PROMPT.format(
            context = context[:4000],
            query   = query
        )

        try:
            resp    = self.llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            logger.info(f"Output type : {content[:15]}")
            return {"response": content}
        except Exception as e:
            logger.error(f"Erreur GeneratorAgent : {e}")
            return {"response": f"[TEXT] Erreur lors de la generation : {e}"}