import json
import os
import re
import time

from langchain_community.vectorstores import FAISS

from config.settings import settings
from models.embedding_model import load_embeddings
from utils.logger import get_logger

logger = get_logger(__name__)

META_FILE = "meta.json"


class IndexMismatchError(RuntimeError):
    pass


class VectorStore:
    def __init__(self, path: str = None):
        self.path = path or settings.FAISS_PATH
        self.embeddings = load_embeddings()
        self.db = None

    @staticmethod
    def _signature() -> dict:
        return {"embedding_model": settings.EMBEDDING_MODEL, "dim": settings.EMBEDDING_DIM}

    def _meta_path(self) -> str:
        return os.path.join(self.path, META_FILE)

    def _write_meta(self):
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(self._signature(), f)

    def _meta_problem(self):
        """Retourne un message d'erreur si l'index n'est pas compatible, sinon None."""
        rebuild = "Reconstruis-le avec : python scripts/ingest_data.py"
        if not os.path.exists(self._meta_path()):
            return f"Index sans meta.json (probablement l'ancien index MiniLM). {rebuild}"
        try:
            with open(self._meta_path(), encoding="utf-8") as f:
                saved = json.load(f)
        except Exception:
            return f"meta.json illisible. {rebuild}"
        if saved != self._signature():
            return f"Index construit avec {saved}, config actuelle {self._signature()}. {rebuild}"
        return None

    @staticmethod
    def _retry(fn, attempts: int = 8):
        """Réessaie en respectant le délai que l'API impose en cas de quota (429)."""
        for n in range(attempts):
            try:
                return fn()
            except Exception as e:
                if n == attempts - 1:
                    raise
                msg = str(e)
                m = re.search(r"retry in ([\d.]+)s", msg)
                wait = float(m.group(1)) + 2 if m else min(2 ** n * 2, 30)

                q = re.search(r"'quotaId': '([^']+)'", msg)
                qid = q.group(1) if q else "?"
                if "PerDay" in qid:
                    raise RuntimeError(
                        f"Quota JOURNALIER atteint ({qid}). "
                        "Réessaie demain ou active la facturation."
                    ) from e

                logger.warning(
                    f"Embeddings : quota -> nouvel essai dans {wait:.0f}s "
                    f"({n + 1}/{attempts}) [{qid}]"
                )
                time.sleep(wait)

    def _build(self, docs: list):
        """Construit un index FAISS en envoyant les documents par lots."""
        size = settings.EMBED_BATCH_SIZE
        db = None
        for i in range(0, len(docs), size):
            batch = docs[i:i + size]
            if db is None:
                db = self._retry(lambda b=batch: FAISS.from_documents(
                    b, self.embeddings, normalize_L2=True))
            else:
                time.sleep(settings.EMBED_PAUSE_S)
                self._retry(lambda b=batch: db.add_documents(b))
            logger.info(f"Indexation : {min(i + size, len(docs))}/{len(docs)} documents")
        return db

    def load(self):
        """Charge l'index depuis le disque. Retourne None s'il est absent ou incompatible."""
        self.db = None
        if not os.path.exists(self.path):
            return None
        problem = self._meta_problem()
        if problem:
            logger.error(problem)
            return None
        try:
            self.db = FAISS.load_local(
                self.path, self.embeddings,
                allow_dangerous_deserialization=True,
                normalize_L2=True,
            )
        except Exception as e:
            logger.error(f"Chargement FAISS impossible : {e}")
            self.db = None
        return self.db

    def save(self, docs):
        """Crée un nouvel index (remplace l'ancien) et le sauvegarde."""
        if not docs:
            raise ValueError("Aucun document à indexer")
        self.db = self._build(docs)
        os.makedirs(self.path, exist_ok=True)
        self.db.save_local(self.path)
        self._write_meta()

    def add_documents(self, docs):
        """Ajoute à l'index existant, ou le crée s'il n'existe pas."""
        if not os.path.exists(self.path):
            return self.save(docs)
        problem = self._meta_problem()
        if problem:
            raise IndexMismatchError(problem)
        if self.db is None:
            self.load()
        if self.db is None:
            return self.save(docs)
        size = settings.EMBED_BATCH_SIZE
        for i in range(0, len(docs), size):
            if i:
                time.sleep(settings.EMBED_PAUSE_S)
            self._retry(lambda b=docs[i:i + size]: self.db.add_documents(b))
        self.db.save_local(self.path)

    def similarity_search(self, query, k=3):
        if self.db is None:
            self.load()
        if self.db is None:
            raise RuntimeError(
                "Index FAISS absent ou incompatible. Lance `python scripts/ingest_data.py`."
            )
        return self.db.similarity_search(query, k=k)