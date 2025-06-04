import pytest
from data.neo4j_client import *

#------execute_query------
def test_execute_query_basic_return():
    """
    Test whether the database returns a valid 'total_nodes'.
    Ensures that the database connection and query execution work.

    Verifies:
        - The query returns exactly one result.
        - The result contains the 'total_nodes' key.
        - The value of 'total_nodes' is an integer.
    """
    result = execute_query("MATCH (n) RETURN COUNT(n) AS total_nodes")

    assert len(result) == 1
    assert "total_nodes" in result[0]
    assert isinstance(result[0]["total_nodes"], int)

def test_execute_query_specific_nodes():
    """
    Test querying a limited number of nodes with the 'problem' label.
    Ensures that the returned names are valid strings.

    Verifies:
        - Result is a list.
        - Each record contains a 'name' key.
        - Each 'name' value is a string.
    """
    query = """
    MATCH (p:problem)
    WHERE p.name IS NOT NULL
    RETURN p.name AS name
    LIMIT 5
    """
    result = execute_query(query)

    assert isinstance(result, list)
    for record in result:
        assert "name" in record
        assert isinstance(record["name"], str)

#-------execute_multiple_queries----------
def test_execute_multiple_queries_no_params():
    """
    Test executing multiple Cypher queries using APOC without parameters.
    Checks that the results include correct fields and expected types.

    Verifies:
        - The result is a list with expected number of elements.
        - Each element has a 'value' key containing expected data.
    """
    queries = [
        {
            "query": "MATCH (n) RETURN COUNT(n) AS total_nodes",
            "params": {}
        },
        {
            "query": "MATCH (n:problem) RETURN n.name AS name LIMIT 3",
            "params": {}
        }
    ]

    result = execute_multiple_queries(queries)

    # Should return a list of dicts: [{'value': {...}}, ...]
    print(result)
    assert isinstance(result, list)
    assert len(result) == 4

    # First query: total_nodes
    assert "value" in result[0]
    assert "total_nodes" in result[0]["value"]
    assert isinstance(result[0]["value"]["total_nodes"], int)

    # Second query: name
    names_output = result[1]["value"]
    assert isinstance(names_output, dict)
    assert "name" in names_output
    assert isinstance(names_output["name"], str)

def test_execute_multiple_queries_with_params():
    """
    Test executing multiple parameterized queries using APOC.
    Verifies correct result mapping and type for known node names.

    Verifies:
        - Result list length matches query count.
        - Each result contains expected keys and values.
        - Returned values match the parameterized input queries.
    """
    queries = [
        {
            "query": "MATCH (c:context {name: $name}) RETURN c.name AS name",
            "params": {"name": "EMF-based model variants"}
        },
        {
            "query": "MATCH (s:stakeholder {name: $name}) RETURN s.name AS name",
            "params": {"name": "developers"}
        }
    ]

    result = execute_multiple_queries(queries)

    assert isinstance(result, list)
    assert len(result) == 2

    # First query result
    context_result = result[0]["value"]
    assert isinstance(context_result, dict)
    assert "name" in context_result
    assert isinstance(context_result["name"], str)
    assert context_result["name"] == "EMF-based model variants"

    # Second query result
    stakeholder_result = result[1]["value"]
    assert isinstance(stakeholder_result, dict)
    assert "name" in stakeholder_result
    assert isinstance(stakeholder_result["name"], str)
    assert stakeholder_result["name"] == "developers"

@pytest.fixture(scope="session", autouse=True)
def teardown_driver():
    """
    Automatically invoked once after all tests in the session are run.
    Ensures the Neo4j driver is closed and cleaned.
    """
    yield  # Run tests first
    close_driver()  # Then clean up