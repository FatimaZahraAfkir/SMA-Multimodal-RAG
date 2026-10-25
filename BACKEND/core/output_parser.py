import json
import re

MAX_ROWS, MAX_NODES = 200, 60


def _lenient_json(s: str):
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\}\s*\{", "},{", s)
    return json.loads(re.sub(r"\s+", " ", s).strip())


def _cell(v):
    if v is None:
        return ""
    return v if isinstance(v, (str, int, float, bool)) else str(v)


def _parse_table(body: str):
    try:
        data = _lenient_json(body)
    except Exception:
        return None
    if isinstance(data, dict):
        rows = []
        for k, v in data.items():
            row = {"Concept": k}
            row.update(v if isinstance(v, dict) else {"Valeur": str(v)})
            rows.append(row)
        data = rows
    if not isinstance(data, list) or not data or not all(isinstance(r, dict) for r in data):
        return None
    columns = []
    for r in data:
        for k in r:
            if k not in columns:
                columns.append(k)
    return {"columns": columns,
            "rows": [[_cell(r.get(c)) for c in columns] for r in data[:MAX_ROWS]],
            "total_rows": len(data)}


def _parse_graph(body: str):
    relations = []
    try:
        data = _lenient_json(body)
        relations = data.get("relations", []) if isinstance(data, dict) else data
    except Exception:
        src = re.findall(r'"source"\s*:\s*"([^"]+)"', body)
        dst = re.findall(r'"cible"\s*:\s*"([^"]+)"', body)
        typ = re.findall(r'"type"\s*:\s*"([^"]+)"', body)
        relations = [{"source": s, "cible": d, "type": typ[i] if i < len(typ) else ""}
                     for i, (s, d) in enumerate(zip(src, dst))]
    nodes, edges = {}, []
    for r in relations if isinstance(relations, list) else []:
        if not isinstance(r, dict):
            continue
        s = str(r.get("source", "")).strip()
        d = str(r.get("cible", r.get("target", ""))).strip()
        if not s or not d or s == "?" or d == "?":
            continue
        if len(nodes) >= MAX_NODES and (s not in nodes or d not in nodes):
            continue
        nodes.setdefault(s, {"id": s, "label": s})
        nodes.setdefault(d, {"id": d, "label": d})
        t = str(r.get("type", r.get("label", ""))).strip()
        edges.append({"source": s, "target": d, "type": t, "label": t.replace("_", " ").lower()[:30]})
    return {"nodes": list(nodes.values()), "edges": edges}


def _clean_text(text: str) -> str:
    text = re.sub(r"\[/?(?:TEXT|IMAGE)\]", "", text or "")
    return text.strip()


def _result(type_, text="", table=None, graph=None, image_b64=None):
    return {"type": type_, "text": text, "table": table, "graph": graph, "image_b64": image_b64}


def parse_output(raw: str) -> dict:
    raw = (raw or "").strip()

    if raw.startswith("[TABLEAU]") and "[ANALYSE]" in raw:
        left, analysis = raw.split("[ANALYSE]", 1)
        table = _parse_table(left.replace("[TABLEAU]", "", 1))
        if table:
            return _result("table", _clean_text(analysis), table=table)

    m = re.search(r"\[GRAPH\](.*?)\[/GRAPH\]", raw, re.DOTALL)
    if m:
        graph = _parse_graph(m.group(1))
        if graph["edges"]:
            return _result("graph", _clean_text(raw[:m.start()] + raw[m.end():]), graph=graph)

    m = re.search(r"\[(?:TABLE|TABLEAU)\](.*?)\[/(?:TABLE|TABLEAU)\]", raw, re.DOTALL)
    if m:
        table = _parse_table(m.group(1))
        if table:
            return _result("table", _clean_text(raw[:m.start()] + raw[m.end():]).replace("Note :", "").strip(),
                           table=table)

    if raw.startswith("[IMAGE_B64]"):
        return _result("image", image_b64=raw[len("[IMAGE_B64]"):].strip())

    return _result("text", _clean_text(raw))
