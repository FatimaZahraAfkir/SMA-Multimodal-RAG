"use client";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const STEP_ORDER = ["routing", "retrieval", "generation", "regeneration", "validation"];
const LEVEL_COLOR = { high: "var(--ok)", medium: "var(--warn)", low: "var(--bad)", unverified: "var(--mut)" };

function Gauge({ c, hasSources }) {
  if (!c || c.score == null) return c && hasSources ? <div className="mut">Confiance : {c.label.toLowerCase()} — vérifiez les sources ci-dessous</div> : null;
  const col = LEVEL_COLOR[c.level] || "var(--mut)", r = 20, len = 2 * Math.PI * r;
  return (
    <div className="gauge" title={(c.reasons || []).join(" · ")}>
      <svg width="52" height="52" viewBox="0 0 52 52">
        <circle cx="26" cy="26" r={r} fill="none" stroke="var(--line)" strokeWidth="5" />
        <circle cx="26" cy="26" r={r} fill="none" stroke={col} strokeWidth="5" strokeLinecap="round"
          strokeDasharray={`${(c.score / 100) * len} ${len}`} transform="rotate(-90 26 26)" />
        <text x="26" y="30" textAnchor="middle">{c.score}</text>
      </svg>
      <div><b>Confiance {c.label.toLowerCase()}</b><small>{c.reasons?.[0] || "score indicatif"}</small></div>
    </div>
  );
}

function GraphView({ graph }) {
  const W = 760, H = 520, cx = W / 2, cy = H / 2, R = 190;
  const deg = {};
  graph.edges.forEach((e) => {
    deg[e.source] = (deg[e.source] || 0) + 1;
    deg[e.target] = (deg[e.target] || 0) + 1;
  });
  const hub = graph.nodes.reduce((b, n) => ((deg[n.id] || 0) > (deg[b.id] || 0) ? n : b), graph.nodes[0]);
  const others = graph.nodes.filter((n) => n.id !== hub.id);
  const pos = { [hub.id]: { x: cx, y: cy, a: 0 } };
  others.forEach((n, i) => {
    const a = (2 * Math.PI * i) / others.length - Math.PI / 2;
    pos[n.id] = { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a), a };
  });
  const showLabels = graph.edges.length <= 10;   
  const halo = { paintOrder: "stroke", stroke: "var(--card)", strokeWidth: 4, strokeLinejoin: "round" };
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", background: "var(--card)", borderRadius: 12 }}>
      <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M0 0L10 5L0 10z" fill="var(--acc2)" /></marker></defs>
      {graph.edges.map((e, i) => {
        const p1 = pos[e.source], p2 = pos[e.target];
        const dx = p2.x - p1.x, dy = p2.y - p1.y, len = Math.hypot(dx, dy) || 1;
        const ux = dx / len, uy = dy / len;
        return (
          <g key={i}>
            <title>{`${e.source} → ${e.label} → ${e.target}`}</title>
            <line x1={p1.x + ux * 12} y1={p1.y + uy * 12} x2={p2.x - ux * 12} y2={p2.y - uy * 12}
              stroke="var(--acc2)" strokeWidth="1.5" markerEnd="url(#arr)" opacity=".75" />
            {showLabels && (
              <text x={p1.x + dx * 0.6} y={p1.y + dy * 0.6 - 4} textAnchor="middle"
                style={{ ...halo, fill: "var(--acc)", fontSize: 9 }}>{e.label}</text>
            )}
          </g>
        );
      })}
      {graph.nodes.map((nd) => {
        const p = pos[nd.id], isHub = nd.id === hub.id, c = Math.cos(p.a);
        const anchor = isHub ? "middle" : c > 0.25 ? "start" : c < -0.25 ? "end" : "middle";
        const lx = isHub ? p.x : p.x + (c > 0.25 ? 14 : c < -0.25 ? -14 : 0);
        const ly = isHub ? p.y + 28 : p.y + Math.sin(p.a) * 16 + 4;
        return (
          <g key={nd.id}>
            <circle cx={p.x} cy={p.y} r={isHub ? 11 : 7} fill={isHub ? "var(--acc2)" : "var(--acc)"} />
            <text x={lx} y={ly} textAnchor={anchor}
              style={{ ...halo, fill: "var(--txt)", fontSize: isHub ? 13 : 11, fontWeight: isHub ? 700 : 400 }}>
              {nd.label.length > 26 ? nd.label.slice(0, 25) + "…" : nd.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function Table({ t }) {
  return (
    <div className="tablewrap"><table>
      <thead><tr>{t.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
      <tbody>{t.rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j}>{String(v)}</td>)}</tr>)}</tbody>
    </table>{t.total_rows > t.rows.length && <div className="mut">{t.rows.length} / {t.total_rows} lignes affichées</div>}</div>
  );
}

function Sources({ list }) {
  if (!list?.length) return null;
  return (
    <div className="sources">
      <div className="mut">Sources</div>
      {list.map((s, i) => {
        const inner = <>{s.title}{s.page ? ` · p.${s.page}` : ""}{s.snippet && <small>{s.snippet.slice(0, 110)}…</small>}</>;
        return s.url ? <a key={i} className="src" href={s.url} target="_blank" rel="noreferrer">{inner}</a>
                     : <div key={i} className="src">{inner}</div>;
      })}
    </div>
  );
}

function CopyButton({ text, label = "Copier" }) {
  const [done, setDone] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setDone(true);
      setTimeout(() => setDone(false), 1500);
    } catch {}
  }
  return <button className="act" onClick={copy} title={label}>{done ? "✓ Copié" : "⧉ Copier"}</button>;
}

export function UserMessage({ m, canEdit, editing, onEdit, onCancel, onSubmit }) {
  const [draft, setDraft] = useState(m.text);
  const ref = useRef(null);
  useEffect(() => { if (editing) { setDraft(m.text); setTimeout(() => ref.current?.focus(), 0); } }, [editing, m.text]);

  if (editing) {
    return (
      <div className="msg-user-wrap">
        <div className="edit-box">
          <textarea ref={ref} rows={3} value={draft} onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (draft.trim()) onSubmit(draft.trim()); }
              if (e.key === "Escape") onCancel();
            }} />
          <div className="edit-actions">
            <button onClick={onCancel}>Annuler</button>
            <button className="send" disabled={!draft.trim()} onClick={() => onSubmit(draft.trim())}>Envoyer</button>
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="msg-user-wrap">
      <div className="msg-user" dir="auto">{m.text}</div>
      <div className="actions right">
        <CopyButton text={m.text} />
        {canEdit && <button className="act" onClick={onEdit} title="Modifier le message">✎ Modifier</button>}
      </div>
    </div>
  );
}

export function Assistant({ m, canRegen, onRegen }) {
  const o = m.output;
  return (
    <div className="msg-ai">
      {m.alert && <div className="banner alert">🚨 {m.alert}</div>}
      <div className="steps">
        {STEP_ORDER.filter((s) => m.steps?.[s]).map((s) => (
          <span key={s} className={`step ${m.steps[s].state}`}>{m.steps[s].label}{m.steps[s].detail ? ` · ${m.steps[s].detail}` : ""}</span>
        ))}
      </div>
      {o?.type === "graph" && <GraphView graph={o.graph} />}
      {o?.type === "table" && <Table t={o.table} />}
      {m.text && <div dir="auto" className={`md ${m.streaming ? "cursor" : ""}`}><ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown></div>}
      {m.streaming && !m.text && !o && <div className="mut cursor">Réflexion en cours</div>}
      {m.note && <div className="banner warn">{m.note}</div>}
      {m.warning && <div className="banner warn">⚠️ {m.warning}</div>}
      {m.error && <div className="banner alert">{m.error}</div>}
      {((m.confidence && (m.confidence.score != null || m.sources?.length > 0)) || m.sources?.length > 0) && (
        <div className="meta"><Gauge c={m.confidence} hasSources={m.sources?.length > 0} /><Sources list={m.sources} /></div>
      )}
      {o && <div className="foot">Information à but éducatif — elle ne remplace pas l'avis d'un professionnel de santé.</div>}
      {!m.streaming && (
        <div className="actions">
          {m.text && <CopyButton text={m.text} />}
          {canRegen && <button className="act" onClick={onRegen} title="Générer une nouvelle réponse">↻ Régénérer</button>}
        </div>
      )}
    </div>
  );
}