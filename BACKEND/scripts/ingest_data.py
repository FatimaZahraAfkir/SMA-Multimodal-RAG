import argparse
import json
import os
import shutil
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings

DOC_PATH = "data/documents/"
GROUP = 50         
MIN_TEXT_CHARS = 200   


def _manifest_path() -> str:
    return settings.FAISS_PATH.rstrip("/\\") + "_manifest.json"


def _load_manifest() -> dict:
    try:
        with open(_manifest_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_manifest(m: dict):
    with open(_manifest_path(), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)


def _load_pages(path: str):
    low = path.lower()
    if low.endswith(".pdf"):
        return PyPDFLoader(path).load()
    if low.endswith(".docx"):
        return Docx2txtLoader(path).load()
    return None


def main(argv=None, store=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="efface l'index et recommence")
    ap.add_argument("--only", help="n'indexe que les fichiers dont le nom contient ce texte")
    args = ap.parse_args(argv)

    if not os.path.isdir(DOC_PATH):
        print(f"Dossier introuvable : {DOC_PATH} (lance le script depuis le dossier BACKEND)")
        return 1

    if args.rebuild:
        shutil.rmtree(settings.FAISS_PATH, ignore_errors=True)
        if os.path.exists(_manifest_path()):
            os.remove(_manifest_path())
        print("Index précédent effacé.")

    from database.vector_store import VectorStore, IndexMismatchError
    store = store or VectorStore()
    manifest = _load_manifest()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    files = sorted(
        (f for f in os.listdir(DOC_PATH) if f.lower().endswith((".pdf", ".docx"))),
        key=lambda f: os.path.getsize(os.path.join(DOC_PATH, f)),
    )
    if args.only:
        files = [f for f in files if args.only.lower() in f.lower()]
    if not files:
        print("Aucun fichier correspondant dans data/documents/" if args.only
              else "Aucun PDF/DOCX trouvé dans data/documents/")
        return 1
    print(f"{len(files)} fichier(s) à traiter.")

    sent_once = False   

    try:
        for name in files:
            path = os.path.join(DOC_PATH, name)
            size = os.path.getsize(path)
            entry = manifest.get(name)

            if entry and entry.get("size") != size:
                print(f"⚠ {name} : modifié depuis l'indexation -> relance avec --rebuild pour le réindexer")
                continue
            if entry and entry.get("complete"):
                print(f"✓ {name} : déjà indexé ({entry.get('total', 0)} extraits)")
                continue

            pages = _load_pages(path)
            if sum(len(d.page_content.strip()) for d in pages) < MIN_TEXT_CHARS:
                print(f"⚠ {name} : aucun texte extractible (PDF scanné ?) -> ignoré (il faudrait une OCR)")
                manifest[name] = {"size": size, "complete": True, "total": 0, "note": "pas de texte"}
                _save_manifest(manifest)
                continue

            chunks = splitter.split_documents(pages)
            done = entry.get("done", 0) if entry else 0
            print(f"→ {name} : {len(chunks)} extraits" + (f" (reprise à {done})" if done else ""))

            for i in range(done, len(chunks), GROUP):
                if sent_once:
                    time.sleep(settings.EMBED_PAUSE_S)   
                store.add_documents(chunks[i:i + GROUP])
                sent_once = True
                reached = min(i + GROUP, len(chunks))
                manifest[name] = {"size": size, "done": reached, "total": len(chunks),
                                  "complete": reached >= len(chunks)}
                _save_manifest(manifest)
                print(f"   {reached}/{len(chunks)}")

    except IndexMismatchError as e:
        print(f"\nIndex incompatible : {e}\nRelance avec : python scripts/ingest_data.py --rebuild")
        return 1
    except Exception as e:
        print(f"\nInterrompu : {type(e).__name__}: {str(e)[:200]}")
        print("Probablement une limite de l'API gratuite ou le réseau. Ce qui est déjà indexé est conservé :")
        print("relance la même commande (plus tard si c'est un quota), elle reprend là où elle s'est arrêtée.")
        return 1

    total = sum(v.get("total", 0) for v in manifest.values())
    print(f"\nTerminé : {sum(1 for v in manifest.values() if v.get('total'))} fichier(s) indexé(s), {total} extraits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())