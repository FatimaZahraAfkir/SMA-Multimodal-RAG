"use client";
import { useEffect, useRef, useState } from "react";
import Sidebar from "../components/Sidebar";
import { Assistant, UserMessage } from "../components/Message";
import { streamChat, API } from "../lib/sse";
import { loadConvs, saveConvs, newId, makeTitle, toHistory } from "../lib/store";

const ROUTES = [["auto", "Auto"], ["text", "Documents"], ["graph", "Graphe médical"]];
const EXAMPLES = ["C'est quoi le diabète de type 2 ?", "Posologie du paracétamol chez l'adulte", "Montre le graphe du diabète"];

export default function Page() {
  const [convs, setConvs] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [ready, setReady] = useState(false);
  const [input, setInput] = useState("");
  const [route, setRoute] = useState("auto");
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [search, setSearch] = useState("");
  const [drawer, setDrawer] = useState(false);        
  const [collapsed, setCollapsed] = useState(false);  
  const [editing, setEditing] = useState(null);      
  const [confirmDel, setConfirmDel] = useState(null); 
  const [theme, setTheme] = useState("dark");

  const feedRef = useRef(null), fileRef = useRef(null), taRef = useRef(null);
  const abortRef = useRef(null), stickRef = useRef(true), saveTimer = useRef(null);

  const active = convs.find((c) => c.id === activeId) || null;
  const msgs = active ? active.messages : [];

  useEffect(() => {
    const list = loadConvs();
    setConvs(list);
    const last = [...list].sort((a, b) => b.updatedAt - a.updatedAt)[0];
    if (last) setActiveId(last.id);
    setCollapsed(localStorage.getItem("sma_side_collapsed") === "1");
    setTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
    setReady(true);
  }, []);

  useEffect(() => {
    if (!ready) return;
    clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => saveConvs(convs), 400);  
    return () => clearTimeout(saveTimer.current);
  }, [convs, ready]);

  useEffect(() => {
    const flush = () => saveConvs(convs);
    window.addEventListener("beforeunload", flush);
    return () => window.removeEventListener("beforeunload", flush);
  }, [convs]);

  useEffect(() => {
    if (stickRef.current) feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight });
  }, [msgs]);
  useEffect(() => { stickRef.current = true; }, [activeId]);
  useEffect(() => {
    const el = taRef.current;
    if (el) { el.style.height = "auto"; el.style.height = Math.min(el.scrollHeight, 180) + "px"; }
  }, [input]);

  const onScroll = () => {
    const el = feedRef.current;
    if (el) stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  };

  const updateConv = (id, fn) => setConvs((cs) => cs.map((c) => (c.id === id ? fn(c) : c)));
  const patchLast = (id, fn) => updateConv(id, (c) => {
    const m = [...c.messages];
    m[m.length - 1] = fn(m[m.length - 1]);
    return { ...c, messages: m };
  });

  async function runTurn(id, message, f, prior) {
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setBusy(true);
    stickRef.current = true;

    const onEvent = (ev, d) => {
      if (ev === "status") patchLast(id, (a) => ({ ...a, steps: { ...a.steps, [d.step]: d } }));
      else if (ev === "alert") patchLast(id, (a) => ({ ...a, alert: d.message }));
      else if (ev === "warning") patchLast(id, (a) => ({ ...a, note: d.message }));
      else if (ev === "sources") patchLast(id, (a) => ({ ...a, sources: d.sources }));
      else if (ev === "token") patchLast(id, (a) => ({ ...a, text: a.text + d.text }));
      else if (ev === "reset") patchLast(id, (a) => ({ ...a, text: "" }));
      else if (ev === "result") patchLast(id, (a) => ({ ...a, ...d, text: d.output.text, streaming: false }));
      else if (ev === "error") patchLast(id, (a) => ({ ...a, error: d.message, streaming: false }));
      else if (ev === "done") patchLast(id, (a) => ({ ...a, streaming: false }));
    };

    try {
      await streamChat({
        message, sessionId: id, route, file: f, history: toHistory(prior),
        signal: ctrl.signal, onEvent,
      });
    } catch (e) {
      if (e.name === "AbortError") patchLast(id, (a) => ({ ...a, note: "Génération interrompue.", streaming: false }));
      else patchLast(id, (a) => ({ ...a, error: e.message || "Connexion impossible au serveur.", streaming: false }));
    } finally {
      patchLast(id, (a) => ({ ...a, streaming: false }));
      abortRef.current = null;
      setBusy(false);
    }
  }

  const userMsg = (text, f) => ({ role: "user", text, fileName: f?.name });
  const aiMsg = () => ({ role: "ai", text: "", steps: {}, streaming: true });

  async function send(text = input) {
    const message = text.trim();
    if ((!message && !file) || busy) return;
    const f = file;
    setInput(""); setFile(null);
    if (fileRef.current) fileRef.current.value = "";

    const shown = message || `📎 ${f.name}`;
    let id = active?.id, prior = active ? active.messages : [];
    if (!active) {
      id = newId();
      const conv = { id, title: makeTitle(shown), createdAt: Date.now(), updatedAt: Date.now(), messages: [userMsg(shown, f), aiMsg()] };
      setConvs((cs) => [conv, ...cs]);
      setActiveId(id);
    } else {
      updateConv(id, (c) => ({
        ...c, updatedAt: Date.now(),
        title: c.messages.length ? c.title : makeTitle(shown),
        messages: [...c.messages, userMsg(shown, f), aiMsg()],
      }));
    }
    await runTurn(id, message, f, prior);
  }

  // Rejoue la conversation à partir d'un message (modification ou régénération)
  async function replayFrom(index, text) {
    if (!active || busy) return;
    const id = active.id, prior = active.messages.slice(0, index);
    setEditing(null);
    updateConv(id, (c) => ({
      ...c, updatedAt: Date.now(),
      title: index === 0 ? makeTitle(text) : c.title,
      messages: [...prior, userMsg(text), aiMsg()],
    }));
    try { await fetch(`${API}/api/sessions/${id}`, { method: "DELETE" }); } catch {}   // le backend repart de `prior`
    await runTurn(id, text, null, prior);
  }

  function regenerate() {
    const i = msgs.length - 2;
    if (i >= 0 && msgs[i].role === "user" && !msgs[i].fileName) replayFrom(i, msgs[i].text);
  }

  // ── gestion des conversations ──────────────────────────────────────────────
  function newChat() {
    if (busy) abortRef.current?.abort();
    setActiveId(null); setInput(""); setFile(null); setEditing(null); setDrawer(false);
  }
  function selectConv(id) {
    if (id === activeId) { setDrawer(false); return; }
    setActiveId(id); setEditing(null); setDrawer(false);
  }
  const renameConv = (id, title) => updateConv(id, (c) => ({ ...c, title }));
  async function deleteConv(id) {
    if (busy && id === activeId) abortRef.current?.abort();
    setConvs((cs) => cs.filter((c) => c.id !== id));
    if (id === activeId) setActiveId(null);
    setConfirmDel(null);
    try { await fetch(`${API}/api/sessions/${id}`, { method: "DELETE" }); } catch {}
  }
  function toggleTheme() {
    const next = theme === "light" ? "dark" : "light";
    setTheme(next);
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("sma_theme", next); } catch {}
  }
  function toggleSide() {
    if (window.matchMedia("(max-width: 800px)").matches) { setDrawer((d) => !d); return; }
    setCollapsed((c) => { localStorage.setItem("sma_side_collapsed", c ? "0" : "1"); return !c; });
  }

  const toDelete = convs.find((c) => c.id === confirmDel);
  const lastAi = msgs.length - 1;

  return (
    <div className="shell">
      <Sidebar convs={convs} activeId={activeId} search={search} setSearch={setSearch}
        onSelect={selectConv} onNew={newChat} onRename={renameConv} onDelete={setConfirmDel}
        open={drawer} collapsed={collapsed} onClose={() => setDrawer(false)} />

      <div className="main">
        <div className="topbar">
          <div className="topbar-left">
            <button className="icon" onClick={toggleSide} aria-label="Afficher ou masquer la barre latérale">☰</button>
            <h1>🏥 Assistant médical <span>SMA</span></h1>
          </div>
          <div className="topbar-right">
            <button className="icon" onClick={toggleTheme}
              aria-label={theme === "light" ? "Passer en mode sombre" : "Passer en mode clair"}
              title={theme === "light" ? "Mode sombre" : "Mode clair"}>{theme === "light" ? "🌙" : "☀️"}</button>
            <select value={route} onChange={(e) => setRoute(e.target.value)} disabled={busy}>
              {ROUTES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </div>
        </div>

        <div className="feed" ref={feedRef} onScroll={onScroll}>
          <div className="feed-inner">
            {!msgs.length && (
              <div className="empty">
                <h2>Comment puis-je vous aider ?</h2>
                <div>Symptômes · médicaments · documents · imagerie · graphe médical</div>
                <div className="chips">{EXAMPLES.map((e) => <button key={e} onClick={() => send(e)} disabled={busy}>{e}</button>)}</div>
              </div>
            )}
            {msgs.map((m, i) => m.role === "user" ? (
              <UserMessage key={i} m={m}
                canEdit={!busy && !m.fileName}
                editing={editing === i}
                onEdit={() => setEditing(i)}
                onCancel={() => setEditing(null)}
                onSubmit={(t) => replayFrom(i, t)} />
            ) : (
              <Assistant key={i} m={m}
                canRegen={!busy && i === lastAi && !msgs[i - 1]?.fileName}
                onRegen={regenerate} />
            ))}
          </div>
        </div>

        <div className="composer-wrap">
          <form className="composer" onSubmit={(e) => { e.preventDefault(); send(); }}>
            <input ref={fileRef} type="file" hidden onChange={(e) => setFile(e.target.files?.[0] || null)}
              accept=".png,.jpg,.jpeg,.webp,.gif,.bmp,.pdf,.docx,.xlsx,.csv,.txt,.md" />
            {file && (
              <div className="file">📎 {file.name}
                <button type="button" className="x" onClick={() => { setFile(null); if (fileRef.current) fileRef.current.value = ""; }}>✕</button>
              </div>
            )}
            <div className="composer-row">
              <button type="button" className="icon" onClick={() => fileRef.current?.click()} disabled={busy} aria-label="Joindre un fichier">📎</button>
              <textarea ref={taRef} dir="auto" rows={1} value={input} placeholder="Décrivez votre question médicale…" disabled={busy}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }} />
              {busy
                ? <button type="button" className="stop" onClick={() => abortRef.current?.abort()}>■ Arrêter</button>
                : <button className="send" disabled={!input.trim() && !file}>Envoyer</button>}
            </div>
          </form>
          <div className="mut disclaimer">Informations à but éducatif — elles ne remplacent pas l'avis d'un professionnel de santé.</div>
        </div>
      </div>

      {toDelete && (
        <div className="modal-bg" onClick={() => setConfirmDel(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Supprimer cette conversation ?</h3>
            <p className="mut">« {toDelete.title} » sera définitivement supprimée de ce navigateur.</p>
            <div className="modal-actions">
              <button onClick={() => setConfirmDel(null)}>Annuler</button>
              <button className="danger-btn" onClick={() => deleteConv(toDelete.id)}>Supprimer</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}