const KEY = "sma_convs_v1";
const MAX_CONVS = 100;

export const newId = () => crypto.randomUUID().replace(/-/g, "");

export function makeTitle(text) {
  const t = (text || "").replace(/\s+/g, " ").trim();
  if (!t) return "Nouvelle conversation";
  return t.length > 48 ? t.slice(0, 47) + "…" : t;
}

export function loadConvs() {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) || "[]");
    if (!Array.isArray(raw)) return [];
    return raw.map((c) => ({
      ...c,
      messages: (c.messages || []).map((m) => (m.streaming ? { ...m, streaming: false } : m)),
    }));
  } catch {
    return [];
  }
}

function slim(c) {
  return {
    ...c,
    messages: c.messages.map((m) => {
      if (m.role !== "ai") return m;
      const output = m.output ? { ...m.output, image_b64: null } : m.output;
      return { ...m, output, streaming: false };
    }),
  };
}

export function saveConvs(list) {
  const data = [...list].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_CONVS).map(slim);
  try {
    localStorage.setItem(KEY, JSON.stringify(data));
  } catch {

    try { localStorage.setItem(KEY, JSON.stringify(data.slice(0, 30))); } catch {}
  }
}

export function toHistory(messages, maxTurns = 6) {
  const out = [];
  for (let i = 0; i < messages.length - 1; i++) {
    const u = messages[i], a = messages[i + 1];
    if (u.role === "user" && a.role === "ai" && a.text && !a.error) out.push({ q: u.text, a: a.text });
  }
  return out.slice(-maxTurns);
}

export function groupByDate(convs) {
  const now = new Date();
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const DAY = 86400000;
  const defs = [
    ["Aujourd'hui", (t) => t >= start],
    ["Hier", (t) => t >= start - DAY],
    ["7 derniers jours", (t) => t >= start - 7 * DAY],
    ["30 derniers jours", (t) => t >= start - 30 * DAY],
    ["Plus ancien", () => true],
  ];
  const groups = defs.map(([label]) => ({ label, items: [] }));
  [...convs].sort((a, b) => b.updatedAt - a.updatedAt).forEach((c) => {
    groups[defs.findIndex(([, fn]) => fn(c.updatedAt))].items.push(c);
  });
  return groups.filter((g) => g.items.length);
}