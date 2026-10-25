"use client";
import { useEffect, useMemo, useState } from "react";
import { groupByDate } from "../lib/store";

export default function Sidebar({
  convs, activeId, search, setSearch, onSelect, onNew, onRename, onDelete, open, collapsed, onClose,
}) {
  const [menu, setMenu] = useState(null);       
  const [renaming, setRenaming] = useState(null);
  const [draft, setDraft] = useState("");

  useEffect(() => {
    const close = () => setMenu(null);
    window.addEventListener("click", close);
    return () => window.removeEventListener("click", close);
  }, []);

  const q = search.trim().toLowerCase();
  const filtered = useMemo(
    () => (!q ? convs : convs.filter((c) =>
      c.title.toLowerCase().includes(q) || c.messages.some((m) => (m.text || "").toLowerCase().includes(q)))),
    [convs, q],
  );
  const groups = useMemo(() => groupByDate(filtered), [filtered]);

  const commit = () => {
    if (renaming) {
      const t = draft.trim();
      if (t) onRename(renaming, t);
      setRenaming(null);
    }
  };

  return (
    <>
      {open && <div className="overlay" onClick={onClose} />}
      <aside className={`sidebar ${open ? "open" : ""} ${collapsed ? "collapsed" : ""}`}>
        <div className="side-top">
          <button className="new-chat" onClick={onNew}>＋ Nouvelle conversation</button>
          <input className="search" value={search} onChange={(e) => setSearch(e.target.value)}
            placeholder="Rechercher une conversation…" />
        </div>

        <nav className="convs">
          {!groups.length && (
            <div className="mut pad">{q ? "Aucun résultat." : "Aucune conversation pour l'instant."}</div>
          )}
          {groups.map((g) => (
            <div key={g.label} className="group">
              <div className="group-label">{g.label}</div>
              {g.items.map((c) => (
                <div key={c.id} className={`conv ${c.id === activeId ? "active" : ""}`}
                  onClick={() => renaming !== c.id && onSelect(c.id)}>
                  {renaming === c.id ? (
                    <input className="rename" autoFocus value={draft}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => setDraft(e.target.value)}
                      onBlur={commit}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") commit();
                        if (e.key === "Escape") setRenaming(null);
                      }} />
                  ) : (
                    <span className="conv-title" title={c.title}>{c.title}</span>
                  )}
                  {renaming !== c.id && (
                    <button className="dots" aria-label="Options"
                      onClick={(e) => { e.stopPropagation(); setMenu(menu === c.id ? null : c.id); }}>⋯</button>
                  )}
                  {menu === c.id && (
                    <div className="menu" onClick={(e) => e.stopPropagation()}>
                      <button onClick={() => { setDraft(c.title); setRenaming(c.id); setMenu(null); }}>✎ Renommer</button>
                      <button className="danger" onClick={() => { setMenu(null); onDelete(c.id); }}>🗑 Supprimer</button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ))}
        </nav>

        <div className="side-foot mut">Les conversations sont enregistrées dans ce navigateur.</div>
      </aside>
    </>
  );
}