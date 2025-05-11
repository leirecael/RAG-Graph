from neo4j import GraphDatabase
from config.config import NEO4J_URI,NEO4J_PASSWORD,NEO4J_USER

driver = None
def get_driver():
    global driver
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return driver

def close_driver():
    global driver
    if driver:
        driver.close()
        driver = None

def generate_similarity_queries(entities_with_value: list, threshold: float = 0.5, top_k: int = 3):
    queries_with_params = []
    for entity in entities_with_value:
        query = f"""
        WITH $embedding AS embedding
        MATCH (n:{entity.type})
        WHERE n.embedding IS NOT NULL
        WITH n, gds.similarity.cosine(embedding, n.embedding) AS similarity
        WHERE similarity >= {threshold}
        RETURN DISTINCT n.name as name, similarity, labels(n) as labels
        ORDER BY similarity DESC
        LIMIT {top_k}
        """
        params = {
            "embedding": entity.embedding
        }
        queries_with_params.append({
            "query": query,
            "params": params
        })
    return queries_with_params

def execute_multiple_queries_with_apoc(queries_with_params: list[dict]):
    query = """
    UNWIND $queriesWithParams AS qp
    CALL apoc.cypher.run(qp.query, qp.params) YIELD value
    RETURN value
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(query, {
            "queriesWithParams": queries_with_params
        })
        records = result.data()
        return group_similarity_results_by_label(records)
    
def group_similarity_results_by_label(results: list[dict]) -> dict[str, list[str]]:
    grouped = {}

    for item in results:
        data = item["value"]  
        labels = data.get("labels", [])
        if not labels:
            continue  
        
        primary_label = labels[0]  
        if primary_label not in grouped:
            grouped[primary_label] = []

        grouped[primary_label].append(data["name"])

    return grouped

# async def get_similar_nodes_by_entity(entity: str, embedding: list[float], threshold: float = 0.5, top_k: int = 3):
#     query = f"""
#     WITH $embedding AS embedding
#     MATCH (n:{entity})
#     WHERE n.embedding IS NOT NULL
#     WITH n, gds.similarity.cosine(embedding, n.embedding) AS similarity
#     WHERE similarity >= $threshold
#     RETURN DISTINCT n.name as name, similarity
#     ORDER BY similarity DESC
#     LIMIT $top_k
#     """
#     driver = await get_driver()
#     async with driver.session() as session:
#         result = await session.run(query, {
#             "embedding": embedding,
#             "threshold": threshold,
#             "top_k": top_k
#         })
#         records = await result.values()
#         return [{"name":name, "similarity": similarity} for name, similarity in records]#return [record["name"] for record in records]


def execute_query(cypher_query: str, parameters: dict = None):
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher_query, parameters or {})
        records = result.data()
        return extract_unique_entities(records)

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
    relationships_set = set() 
    relationships = []

    for record in records:
        for key, value in record.items():
            if key.endswith('.name'):
                alias = key.split('.')[0]
                name = value
                desc = record.get(f"{alias}.description", "")
                labels = record.get(f"labels({alias})", [])
                hyper = record.get(f"{alias}.hypernym", "")

                for label in labels:
                    category = CATEGORY_MAP.get(label)
                    if category:
                        if name not in entities[category]:
                            entities[category][name] = {
                                'description': remove_redundant_text(desc),
                                'labels': labels,
                                'hypernym': remove_redundant_text(hyper)
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

        for (src_type, tgt_type), rel_type in known_rels.items(): #problem, context, arisesAt
            for src_alias, src_label in alias_map.items(): #c, context
                if src_label == src_type: #c == problem
                    for tgt_alias, tgt_label in alias_map.items(): #c, context
                        if tgt_label == tgt_type and src_alias != tgt_alias: #context == context  p1!=c
                            src_name = record.get(f"{src_alias}.name") #Problema 2
                            tgt_name = record.get(f"{tgt_alias}.name") #Contexto A
                            if src_name and tgt_name:
                                rel_key = (src_name, tgt_name, rel_type) #(Problema 2, Contexto A, arisesAt)
                                if rel_key not in relationships_set:
                                    relationships_set.add(rel_key)
                                    relationships.append({
                                        "from": src_name, #Problema A
                                        "to": tgt_name, #Contexto A
                                        "type": rel_type, #arisesAt
                                    })

    return {
        "entities": entities,
        "relationships": relationships
    }

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
# async def prob():
#     query = """MATCH (p1:problem), (ac:artifactClass), (p2:problem)
#     MATCH (p1)-[:addressedBy]->(ac)<-[:addressedBy]-(p2)
#     WHERE p1.name IS NOT NULL AND ac.name IS NOT NULL AND p1 <> p2
#     WITH DISTINCT p1, ac, p2
#     RETURN p1.name, p1.description, labels(p1),
#         p2.name, p2.description, labels(p2),
#         ac.name, ac.description, labels(ac)"""
#     resultados = await execute_query(query)
#     print(resultados)


# async def prueba(embedding):
#     driver = await get_driver()
#     query = f"""WITH $embedding AS consulta_vector
#         MATCH (n:stakeholder {{name: "software developers"}})
#         WHERE n.embedding IS NOT NULL
#         RETURN gds.similarity.cosine(consulta_vector, n.embedding) AS similarity, n.name as name"""
#     async with driver.session() as session:
#         result = await session.run(query, {
#             "embedding": embedding,
#         })
#         records = await result.data()
#         return [{"node": record["name"], "similarity": record["similarity"]} for record in records]