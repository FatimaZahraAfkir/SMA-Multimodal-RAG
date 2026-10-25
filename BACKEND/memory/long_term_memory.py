from database.vector_store import VectorStore

class LongTermMemory:
    def __init__(self):
        self.db = VectorStore().load()
        if self.db is None:
            print(" Aucune base vectorielle trouvée. La mémoire longue ne peut pas fonctionner correctement.")

    def search(self, query):
        if not self.db:
            return []  # Retourne une liste vide si db est vide
        return self.db.similarity_search(query, k=3)