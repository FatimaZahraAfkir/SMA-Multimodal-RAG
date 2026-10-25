import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrievers.graph_retriever import GraphRetriever

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="Vider le graphe avant ingestion")
    args = parser.parse_args()

    retriever = GraphRetriever()

    if args.clear:
        retriever.db.clear()
        print("Graphe vide.")

    data_dirs = [
        "data/documents",
        "data/pdfs",
        "data/graph_data",
    ]

    total = 0

    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            continue

        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            ext      = os.path.splitext(filename)[1].lower()

            print(f"Ingestion : {filename}...")

            if ext == ".pdf":
                n = retriever.ingest_pdf(filepath)
            elif ext in (".docx", ".doc"):
                n = retriever.ingest_word(filepath)
            elif ext in (".txt", ".md"):
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                n = retriever.ingest_from_text(text, source=filename)
            else:
                print(f"  Format ignore : {ext}")
                continue

            print(f"  -> {n} triplets inseres")
            total += n

    stats = retriever.stats()
    print(f"\nIngestion terminee !")
    print(f"Total triplets inseres : {total}")
    print(f"Graphe Neo4j : {stats['nodes']} noeuds, {stats['edges']} relations")
    print(f"Visualisation : http://localhost:7474")

if __name__ == "__main__":
    main()