from neo4j import GraphDatabase
from utils.logger import get_logger
from dotenv import load_dotenv
import os

load_dotenv()
logger = get_logger(__name__)


class GraphDB:
    def __init__(self):
        uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
        user     = os.getenv("NEO4J_USER",     "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Connexion Neo4j etablie")

    def close(self):
        self.driver.close()

    def add_relation(self, entity1: str, relation: str, entity2: str, source: str = ""):
        relation_clean = relation.upper().replace(" ", "_").replace("-", "_")
        query = f"""
        MERGE (a:Entity {{name: $e1}})
        MERGE (b:Entity {{name: $e2}})
        MERGE (a)-[r:`{relation_clean}` {{source: $source}}]->(b)
        RETURN a, r, b
        """
        with self.driver.session() as session:
            session.run(query, e1=entity1, e2=entity2, source=source)

    def query(self, entity: str) -> list:
        """Recherche exacte insensible a la casse — cherche dans les 2 sens."""
        query = """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE toLower(a.name) = toLower($entity)
        RETURN a.name AS src, type(r) AS rel, b.name AS dst

        UNION

        MATCH (a:Entity)<-[r]-(b:Entity)
        WHERE toLower(a.name) = toLower($entity)
        RETURN b.name AS src, type(r) AS rel, a.name AS dst
        """
        with self.driver.session() as session:
            results  = session.run(query, entity=entity)
            triplets = []
            for record in results:
                rel = record["rel"].replace("_", " ").lower()
                triplets.append(f"{record['src']} -> {rel} -> {record['dst']}")
            return triplets

    def query_contains(self, keyword: str) -> list:
        """
        Recherche floue : trouve tous les noeuds dont le nom CONTIENT le mot-cle.
        Cherche dans les 2 sens (source ET destination).
        Exemple : 'Generative' trouve 'IA Generative', 'IA Generative & LLMs'...
        """
        query = """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE toLower(a.name) CONTAINS toLower($keyword)
           OR toLower(b.name) CONTAINS toLower($keyword)
        RETURN a.name AS src, type(r) AS rel, b.name AS dst
        LIMIT 20
        """
        with self.driver.session() as session:
            results  = session.run(query, keyword=keyword)
            triplets = []
            for record in results:
                rel = record["rel"].replace("_", " ").lower()
                triplets.append(f"{record['src']} -> {rel} -> {record['dst']}")
            return triplets

    def find_path(self, entity1: str, entity2: str) -> list:
        query = """
        MATCH path = shortestPath(
            (a:Entity {name: $e1})-[*..5]-(b:Entity {name: $e2})
        )
        RETURN [node IN nodes(path) | node.name] AS path_nodes
        """
        with self.driver.session() as session:
            result = session.run(query, e1=entity1, e2=entity2)
            record = result.single()
            return record["path_nodes"] if record else []

    def stats(self) -> dict:
        with self.driver.session() as session:
            nodes = session.run("MATCH (n:Entity) RETURN count(n) AS count").single()["count"]
            edges = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            return {"nodes": nodes, "edges": edges}

    def clear(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Graphe Neo4j vide")