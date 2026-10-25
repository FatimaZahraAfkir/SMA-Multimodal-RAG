from database.graph_db import GraphDB
from models.llm_loader import load_llm
from utils.logger import get_logger
import pdfplumber
import docx
import json
import re
import os

logger = get_logger(__name__)


class GraphRetriever:
    def __init__(self):
        self.db  = GraphDB()
        self.llm = load_llm()
        
    def retrieve(self, entity: str) -> list:
        triplets = self.db.query(entity)
        logger.info(f"Neo4j query '{entity}' -> {len(triplets)} triplets")
        return triplets

    def find_path(self, e1: str, e2: str) -> list:
        return self.db.find_path(e1, e2)

    def stats(self) -> dict:
        return self.db.stats()

    def _parse_triplets(self, text: str) -> list:
        """
        Extrait les triplets depuis la reponse LLM,
        meme si elle est mal formatee.
        """
        if not text:
            return []

        text = re.sub(r"```(?:json)?", "", text)
        text = re.sub(r"```", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception:
            pass

        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return data
            except Exception:
                pass

        triplets = []
        for line in text.splitlines():
            line = line.strip().strip("-*").strip()

            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3 and all(parts):
                    triplets.append({"s": parts[0], "r": parts[1], "o": parts[2]})

            elif "->" in line:
                parts = [p.strip() for p in line.split("->")]
                if len(parts) == 3 and all(parts):
                    triplets.append({"s": parts[0], "r": parts[1], "o": parts[2]})

        if triplets:
            logger.info(f"Triplets extraits par parsing ligne : {len(triplets)}")
            return triplets

        logger.warning(f"Impossible de parser : {text[:200]}")
        return []

    def ingest_from_text(self, text: str, source: str = "") -> int:
        if not text.strip():
            return 0

        prompt = f"""Extrais les relations importantes de ce texte.

FORMAT OBLIGATOIRE - retourne UNIQUEMENT ce JSON, rien d'autre :
[{{"s":"sujet","r":"relation","o":"objet"}},{{"s":"sujet2","r":"relation2","o":"objet2"}}]

Exemples :
[{{"s":"FFN","r":"compose_de","o":"deux_couches"}},{{"s":"Transformer","r":"utilise","o":"Attention"}}]

REGLES :
- Pas de texte avant ou apres le JSON
- Pas de markdown, pas de ```, pas d explication
- Sujet et objet = concepts courts (1-3 mots)
- Relation = verbe court (compose_de, utilise, appartient_a, precede)

Texte :
{text[:3000]}

JSON:"""

        try:
            response = self.llm.invoke(prompt)
            raw      = response.content if hasattr(response, "content") else str(response)
            logger.debug(f"LLM brut : {raw[:300]}")

            triplets = self._parse_triplets(raw)

            if not triplets:
                logger.warning(f"Aucun triplet pour '{source}'")
                return 0

            count = 0
            for t in triplets:
                if isinstance(t, dict) and all(k in t for k in ("s", "r", "o")):
                    s = str(t["s"]).strip()
                    r = str(t["r"]).strip()
                    o = str(t["o"]).strip()
                    if s and r and o:
                        self.db.add_relation(s, r, o, source=source)
                        count += 1

            logger.info(f"Insere {count} triplets depuis '{source}'")
            return count

        except Exception as e:
            logger.error(f"Erreur ingestion : {e}")
            return 0
        
    def ingest_pdf(self, pdf_path: str) -> int:
        total  = 0
        source = os.path.basename(pdf_path)
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    n = self.ingest_from_text(text, source=f"{source}:p{i+1}")
                    total += n
        logger.info(f"PDF '{source}' -> {total} triplets")
        return total

    def ingest_word(self, docx_path: str) -> int:
        source = os.path.basename(docx_path)
        doc    = docx.Document(docx_path)
        text   = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return self.ingest_from_text(text, source=source)