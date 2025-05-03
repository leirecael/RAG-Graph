from neo4j import GraphDatabase
from config.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY
from openai import OpenAI
import openai
client = OpenAI()
openai.api_key = OPENAI_API_KEY
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

NODE_PROPERTIES = {
    "problem": ["name", "description", "excerpt", "hypernym"],
    "context": ["name", "description", "excerpt", "hypernym"],
    "stakeholder": ["name", "description", "excerpt", "hypernym"],
    "goal": ["name", "description", "excerpt", "hypernym"],
    "requirement": ["name", "description", "excerpt", "hypernym"],
    "artifactClass": ["name", "description", "excerpt", "hypernym"]
}

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def generate_text_for_embedding(record, props):
    values = [record.get(prop) for prop in props if record.get(prop)]
    return " ".join(values)

def update_node_embeddings(tx, node_type, props):
    query = f"MATCH (n:{node_type}) RETURN " + ", ".join([f"n.{prop} as {prop}" for prop in props])
    result = tx.run(query)
    for record in result:
        node_name = record["name"]
        text = generate_text_for_embedding(record, props)
        if text:
            embedding = get_embedding(text)
            tx.run(f"""
                MATCH (n:{node_type})
                WHERE n.name = $name
                SET n.embedding = $embedding
            """, name=node_name, embedding=embedding)

def update_all_embeddings():
    with driver.session() as session:
        for node_type, props in NODE_PROPERTIES.items():
            print(f"Updating embeddings for node type: {node_type}")
            session.execute_write(update_node_embeddings, node_type, props)