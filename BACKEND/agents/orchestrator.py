import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict
from langgraph.graph import StateGraph, END

from agents.RAG_agent       import TextAgent
from BACKEND.agents.multimodal_agent      import ImageAgent
from agents.graph_agent      import GraphAgent
from agents.generator_agent  import GeneratorAgent
from agents.validation_agent import ValidationAgent

from memory.short_term_memory import ShortTermMemory
from memory.long_term_memory  import LongTermMemory
from memory.entity_memory     import EntityMemory

from utils.logger import get_logger

logger = get_logger(__name__)

SYMPTOM_KEYWORDS = [
    "symptome", "symptôme", "douleur", "fievre", "fièvre", "toux",
    "mal", "souffre", "ressens", "nausee", "nausée", "vomissement",
    "fatigue", "essoufflement", "vertige", "cephalee", "céphalée",
    "brulure", "brûlure", "saignement", "gonflement", "eruption",
    "j'ai", "je souffre", "depuis hier", "depuis", "j ai"
]

MEDICATION_KEYWORDS = [
    "medicament", "médicament", "traitement", "dose", "posologie",
    "comprime", "comprimé", "antibiotique", "analgesique", "analgésique",
    "anti-inflammatoire", "prescription", "ordonnance", "pharmacie",
    "paracetamol", "ibuprofene", "amoxicilline", "metformine",
    "interaction", "contre-indication", "effets secondaires", "mg", "ml"
]

GRAPH_KEYWORDS = [
    "relation", "graphe", "graph", "lien", "connexion",
    "entre", "associe", "associé", "neo4j", "sous forme"
]

IMAGING_KEYWORDS = [
    "radio", "radiographie", "irm", "scanner", "echographie",
    "echographie", "imagerie", "ecg", "analyse cette image",
    "que montre", "que vois"
]


class AgentState(TypedDict):
    query:               str
    context:             str
    response:            str
    route:               str
    is_valid:            bool
    iterations:          int
    image:               object
    metrics:             dict
    validation_feedback: str


class Orchestrator:
    """
    Orchestrateur medical — 6 agents :
    ┌─────────────────────────────────────────────────────────┐
    │  TextAgent     : RAG FAISS + Web (texte, symptomes,     │
    │                  medicaments, questions medicales)      │
    │  MultimodalAgent    : Gemini Flash+ Groq (images, PDF,  │
    │                  Excel,radios, Word — imagerie medicale)│
    │  GraphAgent    : Neo4j + Web (relations maladie/traite.)│
    │  GeneratorAgent: LLM local + prompt medical             │
    │  ValidationAgent: evaluation qualite + securite         │
    │  Orchestrateur : memoire + routing medical              │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self):
        self.short  = ShortTermMemory()
        self.long   = LongTermMemory()
        self.entity = EntityMemory()

        workflow = StateGraph(AgentState)

        workflow.add_node("memory_and_route", self._memory_and_route)
        workflow.add_node("text",     TextAgent().run)
        workflow.add_node("image",    ImageAgent().run)
        workflow.add_node("graph",    GraphAgent().run)
        workflow.add_node("generate", GeneratorAgent().generate)
        workflow.add_node("validate", ValidationAgent().validate)

        workflow.set_entry_point("memory_and_route")

        workflow.add_conditional_edges(
            "memory_and_route",
            lambda state: state.get("route", "text"),
            {
                "text":  "text",
                "image": "image",
                "graph": "graph",
            }
        )

        workflow.add_edge("text",     "generate")
        workflow.add_edge("image",    "generate")
        workflow.add_edge("graph",    "generate")
        workflow.add_edge("generate", "validate")

        workflow.add_conditional_edges(
            "validate",
            lambda state: "end" if state["is_valid"] or state["iterations"] > 2 else "generate",
            {"end": END, "generate": "generate"}
        )

        self.app = workflow.compile()

    def _memory_and_route(self, state: dict) -> dict:
        query  = state.get("query", "").lower()
        forced = state.get("route")

        if forced in ("text", "image", "graph"):
            route = forced
            logger.info(f"Route forcee : {route}")

        elif state.get("image") is not None:
            route = "image"
            logger.info("Route auto : fichier detecte → ImageAgent")

        elif any(kw in query for kw in IMAGING_KEYWORDS) and state.get("image"):
            route = "image"
            logger.info("Route auto : imagerie medicale → ImageAgent")

        elif any(kw in query for kw in GRAPH_KEYWORDS):
            route = "graph"
            logger.info("Route auto : graphe medical → GraphAgent")

        else:
            route = "text"
            if any(kw in query for kw in SYMPTOM_KEYWORDS):
                logger.info("Route auto : symptomes → TextAgent RAG medical")
            elif any(kw in query for kw in MEDICATION_KEYWORDS):
                logger.info("Route auto : medicament → TextAgent RAG medical")
            else:
                logger.info("Route auto : question medicale → TextAgent RAG")

        try:
            long_results = self.long.search(state.get("query", ""))
            long_context = "\n".join([
                d.page_content for d in long_results
                if hasattr(d, "page_content")
            ]) if long_results else ""
        except Exception:
            long_context = ""

        try:
            short_context = self.short.get_context()
        except Exception:
            short_context = ""

        try:
            entities = self.entity.get_entities()
        except Exception:
            entities = ""

        combined_context = (
            f"Conversation medicale recente :\n{short_context}\n\n"
            f"Documents medicaux (FAISS) :\n{long_context}\n\n"
            f"Entites medicales connues :\n{entities}"
        )

        logger.info(f"Contexte : {len(combined_context)} chars | Route : {route}")
        return {"context": combined_context, "route": route}

    def run(self, state):
        if isinstance(state, str):
            state = {"query": state, "iterations": 0}

        state.setdefault("iterations",          0)
        state.setdefault("image",               None)
        state.setdefault("route",               None)
        state.setdefault("is_valid",            False)
        state.setdefault("metrics",             {})
        state.setdefault("validation_feedback", "")
        state.setdefault("context",             "")
        state.setdefault("response",            "")

        logger.info(f"Run medical → {str(state.get('query',''))[:60]}")
        result = self.app.invoke(state)

        try:
            if result.get("is_valid") and result.get("response"):
                self.short.add(state.get("query", ""), result["response"])
        except Exception:
            pass

        return result