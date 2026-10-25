from database.vector_store import VectorStore

class TextRetriever:
    def __init__(self):
        self.store = VectorStore()
        self.db = self.store.load() 

    def retrieve(self, query, k=4):
        if self.db is None:
            raise RuntimeError(
                "La base vectorielle n'existe pas. Lance d'abord `ingest_data.py` pour l'initialiser."
            )
        return self.db.similarity_search(query, k=k)