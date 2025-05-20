from neo4j import GraphDatabase, Driver
from config.config import NEO4J_URI,NEO4J_PASSWORD,NEO4J_USER

# Global driver instance to manage the Neo4j connection
driver = None

def get_driver() -> Driver:
    """
    Initialize and return a Neo4j driver instance.

    Returns:
        Driver: A Neo4j driver connected to the database.
    """
    global driver
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return driver

def close_driver() -> None:
    """
    Close the global Neo4j driver connection and clean up resources.
    """
    global driver
    if driver:
        driver.close()
        driver = None

def execute_multiple_queries(queries_with_params: list[dict])->list[dict]:
    """
    Execute multiple Cypher queries with parameters using APOC.

    Args:
        queries_with_params (list[dict]): A list of dictionaries, each with 'query' and 'params' keys.

    Returns:
        list[dict]: A list of results from the executed Cypher queries.
    """
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
        return result.data() #Format example: [{"value": {'name': 'software architecture level', 'labels': ['context'], 'similarity': 0.7}}}, {"value": {results2}}, ...]

def execute_query(cypher_query: str, parameters: dict = None)->list[dict]:
    """
    Execute a single Cypher query with optional parameters.

    Args:
        cypher_query (str): The Cypher query to execute.
        parameters (dict, optional): Parameters to use with the query.

    Returns:
        list[dict]: A list of result records.
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher_query, parameters or {})
        return result.data() #Format example: [{'x.prop1': 'text', x.prop2: 'moreText', 'labels(x)': ['entity_type'], 'y.prop1': 'text', 'xCount': 5}]