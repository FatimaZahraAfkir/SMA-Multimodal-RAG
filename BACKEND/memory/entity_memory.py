from models.llm_loader import load_llm

class EntityMemory:
    def __init__(self):
        self.llm = load_llm()
        self.entities = set()

    def extract(self, text):
        prompt = f"Extrais les entités importantes de:\n{text}\nListe séparée par virgule."
        res = self.llm.invoke(prompt).content
        for ent in res.split(","):
            self.entities.add(ent.strip())

    def get_entities(self):
        return ", ".join(self.entities)