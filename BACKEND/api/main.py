"""
API FastAPI. Lancer depuis la racine du projet :
    uvicorn api.main:app --reload --port 8000
"""
import io
import json
import os
import re
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config.settings import settings
from core.pipeline import Pipeline, build_default_pipeline
from utils.logger import get_logger

logger = get_logger(__name__)

SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".pdf",
               ".docx", ".xlsx", ".csv", ".txt", ".md"}
DEFAULT_FILE_QUERY = "Analyse ce fichier médical en détail et résume son contenu."
MAX_HISTORY_TURNS = 6        
MAX_HISTORY_ANSWER_CHARS = 1500

_pipeline: Optional[Pipeline] = None
_lock = threading.Lock()


class UploadedFile(io.BytesIO):
    """Imite l'objet fichier de Streamlit attendu par ImageAgent (.name, .read(), .seek())."""
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def set_pipeline(p: Optional[Pipeline]):
    global _pipeline
    _pipeline = p


def get_pipeline() -> Pipeline:
    global _pipeline
    with _lock:
        if _pipeline is None:
            logger.info("Construction du pipeline...")
            _pipeline = build_default_pipeline()
        return _pipeline


def restore_history(pipeline: Pipeline, sid: str, raw: str):
    """Recharge le contexte d'une conversation rouverte (session expirée ou serveur redémarré).

    Le frontend envoie les tours précédents [{q, a}]. On ne les utilise que si la mémoire de
    session du serveur est vide : une session encore active reste la référence.
    """
    if not raw:
        return
    try:
        items = json.loads(raw)
    except ValueError:
        return
    if not isinstance(items, list):
        return
    try:
        if pipeline.sessions.history_text(sid).strip():
            return
        for it in items[-MAX_HISTORY_TURNS:]:
            if not isinstance(it, dict):
                continue
            q = str(it.get("q", "")).strip()[:settings.MAX_MESSAGE_CHARS]
            a = str(it.get("a", "")).strip()[:MAX_HISTORY_ANSWER_CHARS]
            if q and a:
                pipeline.sessions.add_turn(sid, q, a)
    except Exception:
        logger.exception("Restauration de l'historique impossible")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()          # échoue vite (clé API manquante...) plutôt qu'à la première requête
    yield


app = FastAPI(title="Assistant médical SMA", version="1.1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/health")
def health():
    return {"status": "ok", **get_pipeline().health()}


@app.delete("/api/sessions/{session_id}")
def clear_session(session_id: str):
    return {"cleared": get_pipeline().sessions.clear(session_id)}


@app.post("/api/chat")
async def chat(message: str = Form(""), session_id: str = Form(""), route: str = Form("auto"),
               history: str = Form(""),
               file: Optional[UploadFile] = File(None)):
    message = message.strip()
    if len(message) > settings.MAX_MESSAGE_CHARS:
        raise HTTPException(400, f"Message trop long (max {settings.MAX_MESSAGE_CHARS} caractères).")
    route = route.lower()
    if route not in ("auto", "text", "graph", "image"):
        raise HTTPException(400, "route doit être : auto, text, graph ou image.")
    sid = session_id if SESSION_RE.match(session_id) else uuid.uuid4().hex

    upload = None
    if file is not None and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(415, f"Type de fichier non supporté : {ext or 'inconnu'}.")
        limit = settings.MAX_UPLOAD_MB * 1024 * 1024
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"Fichier trop volumineux (max {settings.MAX_UPLOAD_MB} Mo).")
        if not data:
            raise HTTPException(400, "Fichier vide.")
        upload = UploadedFile(data, os.path.basename(file.filename))

    if route == "image" and upload is None:
        raise HTTPException(400, "Le mode « image » nécessite un fichier.")
    if not message and upload is None:
        raise HTTPException(400, "Écrivez une question ou joignez un fichier.")
    message = message or DEFAULT_FILE_QUERY
    forced = "image" if upload else (None if route == "auto" else route)
    pipeline = get_pipeline()
    restore_history(pipeline, sid, history)

    def stream():
        try:
            for event, data in pipeline.run(message, sid, upload, forced):
                yield sse(event, data)
        except Exception:
            logger.exception("Erreur dans le flux")
            yield sse("error", {"message": "Erreur interne. Réessayez."})
            yield sse("done", {})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})