from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError, AuthConfigurationError
from config.config import NEO4J_URI,NEO4J_PASSWORD,NEO4J_USER

class Neo4jClient:
    """
    Client for managing connections and queries to a Neo4j database.

    Attributes:
        driver (Driver): Neo4j driver instance for database communication.

    Methods:
        test_connection(): Tests if the database can be reached and access credentials are valid. Raises an exception if not.
        close_driver(): Closes the connection with the database.
        execute_multiple_queries(): Executes multiple queries with their respective parameters at the same time. Uses APOC.
        execute_query(): Executes a single query and its parameters.
    """

    def __init__(self):
        """
        Initializes the Neo4j driver with configured URI and credentials,
        and tests the connection to ensure it is valid.
        """
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.test_connection()
    
    def test_connection(self) -> None:
        """
        Tests the connection to the database. If there is any problem, an exception is raised.
        
        """
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
        except (ServiceUnavailable, AuthError, AuthConfigurationError) as e:
            raise RuntimeError("[NEO4J_CONNECTION_ERROR] Failed to connect to Neo4j: " + str(e)) from e

    def close_driver(self) -> None:
        """
        Close the global Neo4j driver connection and clean up resources.
        """
        self.driver.close()


    def execute_multiple_queries(self, queries_with_params: list[dict])->list[dict]:
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
        with self.driver.session() as session:
            result = session.run(query, {
                "queriesWithParams": queries_with_params
            })
            return result.data() #Format example: [{"value": {'name': 'software architecture level', 'labels': ['context'], 'similarity': 0.7}}}, {"value": {results2}}, ...]

    def execute_query(self, cypher_query: str, parameters: dict = None)->list[dict]:
        """
        Execute a single Cypher query with optional parameters.

        Args:
            cypher_query (str): The Cypher query to execute.
            parameters (dict, optional): Parameters to use with the query.

        Returns:
            list[dict]: A list of result records.
        """
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return result.data() #Format example: [{'x.prop1': 'text', x.prop2: 'moreText', 'labels(x)': ['entity_type'], 'y.prop1': 'text', 'xCount': 5}]