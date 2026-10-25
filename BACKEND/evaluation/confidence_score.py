WEIGHTS = {"fidelity": 0.35, "relevance": 0.20, "grounding": 0.15, "sources": 0.20, "rules": 0.10}
SOURCE_SCORE = {"documents": 1.0, "graph": 0.9, "web": 0.6, "none": 0.0}


def infer_source_level(context: str) -> str:
    """Devine d'où vient le contexte (sera remplacé par des sources structurées en phase 4)."""
    c = context or ""
    if "Relations du graphe" in c or "[GRAPH]" in c:
        return "graph"
    if len(c.strip()) < 100:
        return "none"
    if "[SOURCE WEB]" in c and "[SOURCE FAISS]" not in c:
        return "web"
    return "documents"


def _ten(value, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value) / 10))
    except (TypeError, ValueError):
        return default


def compute_confidence(*, judge, rouge, rules: float, source_level: str) -> dict:
    parts = {}
    if judge:
        base = judge["score"] / 10
        metrics = judge.get("metrics", {})
        parts["fidelity"] = _ten(metrics.get("fidelite"), base)
        parts["relevance"] = _ten(metrics.get("pertinence"), base)
    parts["grounding"] = None if rouge is None else min(1.0, rouge / 0.5)
    parts["sources"] = SOURCE_SCORE.get(source_level, 0.0)
    parts["rules"] = rules

    avail = {k: v for k, v in parts.items() if v is not None}
    raw = sum(WEIGHTS[k] * v for k, v in avail.items()) / sum(WEIGHTS[k] for k in avail)
    score, reasons = round(raw * 100), []

    verified = judge is not None
    if not verified:
        score = min(score, 50)
        reasons.append("vérification par le juge indisponible")
    if source_level == "none":
        score = min(score, 45)
        reasons.append("réponse non sourcée")

    if not verified:
        label, level = "Non vérifiée", "unverified"
    elif score >= 75:
        label, level = "Élevée", "high"
    elif score >= 50:
        label, level = "Moyenne", "medium"
    else:
        label, level = "Faible", "low"

    return {"score": score, "label": label, "level": level, "source_level": source_level,
            "components": {k: round(v, 2) for k, v in avail.items()}, "reasons": reasons}