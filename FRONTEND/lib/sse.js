export const API = process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_API || "http://localhost:8000";

export async function streamChat({ message, sessionId, route, file, history, signal, onEvent }) {
  const fd = new FormData();
  fd.append("message", message || "");
  fd.append("session_id", sessionId);
  fd.append("route", route || "auto");
  if (history && history.length) fd.append("history", JSON.stringify(history));
  if (file) fd.append("file", file);

  const res = await fetch(`${API}/api/chat`, { method: "POST", body: fd, signal });
  if (!res.ok) {
    let detail = "";
    try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : ""; } catch {}
    throw new Error(detail || `Erreur du serveur (${res.status}).`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let m;
    while ((m = /\r?\n\r?\n/.exec(buf))) {
      const block = buf.slice(0, m.index);
      buf = buf.slice(m.index + m[0].length);
      let ev = "message";
      const data = [];
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith("event:")) ev = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (!data.length) continue;
      try { onEvent(ev, JSON.parse(data.join("\n"))); } catch { /* bloc illisible : ignoré */ }
    }
  }
}