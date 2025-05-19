from neo4j import GraphDatabase, Driver
from config.config import NEO4J_URI,NEO4J_PASSWORD,NEO4J_USER

driver = None
def get_driver() -> Driver:
    global driver
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return driver

def close_driver() -> None:
    global driver
    if driver:
        driver.close()
        driver = None

def execute_multiple_queries(queries_with_params: list[dict])->list[dict]:
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
        return result.data()   

def execute_query(cypher_query: str, parameters: dict = None)->list[dict]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher_query, parameters or {})
        return result.data()


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