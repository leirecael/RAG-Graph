from neo4j import GraphDatabase
from config.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY
from openai import OpenAI
import openai
client = OpenAI()
openai.api_key = OPENAI_API_KEY
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

NODE_TYPES = [
    "problem", "goal", "requirement",
    "context", "stakeholder", "artifactClass"
]

def remove_redundant_text(text: str) -> str:
    seen = set()
    result = []
    for part in text.split(';'):
        cleaned = part.strip()
        key = cleaned.lower()
        if key and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return '; '.join(result)

def truncate(text):
    parts = text.split(';')
    return parts[0]

def build_text(properties):
    parts = [
        properties.get("name", ""),
        remove_redundant_text(properties.get("hypernym", "")),
        remove_redundant_text(properties.get("alternativeName", "")),
        #truncate(properties.get("description", "")),
    ]
    return " ".join([p for p in parts if p])

def get_embedding(text: str):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

def process_node_type(driver, label):
    query = f"MATCH (n:{label}) RETURN n.name as name, n"
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            node_name = record["name"]
            props = record["n"]
            embedding_input = build_text(props)
            try:
                embedding = get_embedding(embedding_input)
                session.run(
                    "MATCH (n) WHERE n.name = $name SET n.embedding = $embedding",
                    name=node_name,
                    embedding=embedding
                )
                print(f"[{label}] Updated node {node_name}")
            except Exception as e:
                print(f"[{label}] Failed to update node {node_name}: {e}")

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    for label in NODE_TYPES:
       print(f"Processing label: {label}")
       process_node_type(driver, label)
    driver.close()
    # embedding = get_embedding("software developers")
    # query = f"""WITH $embedding AS consulta_vector
    #     MATCH (n:stakeholder {{name: "software developers"}})
    #     WHERE n.embedding IS NOT NULL
    #     RETURN gds.similarity.cosine(consulta_vector, n.embedding) AS similarity, n.name as name"""
    # with driver.session() as session:
    #     result = session.run(query, {
    #         "embedding": embedding,
    #     })
    #     res= [{"node": record["name"], "similarity": record["similarity"]} for record in result]
    #     print(res)


if __name__ == "__main__":
    main()