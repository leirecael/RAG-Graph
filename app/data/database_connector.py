from neo4j import GraphDatabase
from config.config import NEO4J_URI,NEO4J_PASSWORD,NEO4J_USER
import json

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_similar_nodes_by_entity(entity: str, embedding: list[float], threshold: float = 0.45, top_k: int = 5):
    query = f"""
    WITH $embedding AS embedding
    MATCH (n:{entity})
    WHERE n.embedding IS NOT NULL
    WITH n, gds.similarity.cosine(embedding, n.embedding) AS similarity
    WHERE similarity >= $threshold
    RETURN DISTINCT n.name as name, similarity
    ORDER BY similarity DESC
    LIMIT $top_k
    """
    with driver.session() as session:
        result = session.run(query, {
            "embedding": embedding,
            "threshold": threshold,
            "top_k": top_k
        })
        return [record["name"] for record in result]#return [{"node": record["name"], "similarity": record["similarity"]} for record in result]


def execute_query(cypher_query: str, parameters: dict = None):
    with driver.session() as session:
        result = session.run(cypher_query, parameters or {})
        records = [record.data() for record in result]
        print(json.dumps(records, indent=2))
        return extract_unique_entities(records)


def close_connection():
    driver.close()

def prueba(embedding):
    query = f"""WITH $embedding AS consulta_vector
        MATCH (n:stakeholder {{name: "software developers"}})
        WHERE n.embedding IS NOT NULL
        RETURN gds.similarity.cosine(consulta_vector, n.embedding) AS similarity, n.name as name"""
    with driver.session() as session:
        result = session.run(query, {
            "embedding": embedding,
        })
        return [{"node": record["name"], "similarity": record["similarity"]} for record in result]

def extract_unique_entities(records: list) -> dict:
    CATEGORY_MAP = {
        'problem': 'problems',
        'stakeholder': 'stakeholders',
        'goal': 'goals',
        'context': 'contexts',
        'artifactClass': 'artifactClasses',
        'requirement': 'requirements'
    }

    entities = {v: {} for v in CATEGORY_MAP.values()}
    relationships_set = set()  # <- Usamos un set para evitar duplicados
    relationships = []

    for record in records:
        for key, value in record.items():
            if key.endswith('.name'):
                alias = key.split('.')[0]
                name = value
                desc = record.get(f"{alias}.description", "")
                labels = record.get(f"labels({alias})", [])

                for label in labels:
                    category = CATEGORY_MAP.get(label)
                    if category:
                        if name not in entities[category]:
                            entities[category][name] = {
                                'description': remove_redundant_text(desc),
                                'labels': labels
                            }

        known_rels = {
            ('problem', 'context'): 'arisesAt',
            ('problem', 'stakeholder'): 'concerns',
            ('problem', 'goal'): 'informs',
            ('requirement', 'artifactClass'): 'meetBy',
            ('problem', 'artifactClass'): 'addressedBy',
            ('goal', 'requirement'): 'achievedBy'
        }

        alias_map = {}
        for key in record:
            if key.endswith('.name'):
                alias = key.split('.')[0]
                labels = record.get(f"labels({alias})", [])
                for label in labels:
                    if label in CATEGORY_MAP:
                        alias_map[alias] = label

        for (src_type, tgt_type), rel_type in known_rels.items():
            for src_alias, src_label in alias_map.items():
                if src_label == src_type:
                    for tgt_alias, tgt_label in alias_map.items():
                        if tgt_label == tgt_type and src_alias != tgt_alias:
                            src_name = record.get(f"{src_alias}.name")
                            tgt_name = record.get(f"{tgt_alias}.name")
                            if src_name and tgt_name:
                                rel_key = (src_name, tgt_name, rel_type)
                                if rel_key not in relationships_set:
                                    relationships_set.add(rel_key)
                                    relationships.append({
                                        "from": src_name,
                                        "to": tgt_name,
                                        "type": rel_type,
                                        "from_type": src_type,
                                        "to_type": tgt_type
                                    })

    return {
        "entities": entities,
        "relationships": relationships
    }

def remove_redundant_text(text: str) -> str:
    parts = text.split(';')
    unique_parts = list(dict.fromkeys([p.strip() for p in parts if p.strip()]))
    return '; '.join(unique_parts)

def prob():
    query = """MATCH (p1:problem), (ac:artifactClass), (p2:problem)
    MATCH (p1)-[:addressedBy]->(ac)<-[:addressedBy]-(p2)
    WHERE p1.name IS NOT NULL AND ac.name IS NOT NULL AND p1 <> p2
    WITH DISTINCT p1, ac, p2
    RETURN p1.name, p1.description, labels(p1),
        p2.name, p2.description, labels(p2),
        ac.name, ac.description, labels(ac)"""
    resultados = execute_query(query)
    print(resultados)