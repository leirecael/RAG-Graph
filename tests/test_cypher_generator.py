import pytest
from app.logic.cypher_generator import create_cypher_query

def validate_cypher_format(query: str):
    assert "MATCH" in query.upper(), "Must contain MATCH"
    assert "RETURN" in query.upper(), "Must contain RETURN"
    assert query.count("(") == query.count(")"), "Parenthesis open-close quantity should be the same"
    assert "cypher" not in query.upper(), "Query should contain only the cypher query"
    assert "CREATE" not in query.upper(), "CREATE is forbidden"
    assert "DELETE" not in query.upper(), "DELETE is forbidden"
    assert "SET" not in query.upper(), "SET is forbidden"
    assert "MERGE" not in query.upper(), "MERGE is forbidden"
    

@pytest.mark.asyncio
async def test_create_cypher_query_real_call():
    question = "What problems are addressed by the same artifactClass?"
    nodes = {
        "problem": ["Problem A", "Problem B"],
        "artifactClass": ["System X"]
    }

    query, cost = await create_cypher_query(question, nodes)

    validate_cypher_format(query)
    assert isinstance(query, str), "Query is not a string"
    assert cost > 0, "Cypher generation had no cost"

@pytest.mark.asyncio
async def test_create_cypher_query_empty_question():
    question = ""
    nodes = {"problem": ["Problem A"], "artifactClass": ["System X"]}
    query, _ = await create_cypher_query(question, nodes)
    validate_cypher_format(query)

@pytest.mark.asyncio
async def test_create_cypher_query_empty_nodes():
    question = "What problems are addressed by the same artifactClass?"
    nodes = {}
    query, _ = await create_cypher_query(question, nodes)
    validate_cypher_format(query)

@pytest.mark.asyncio
async def test_create_cypher_query_nodes_with_none_values():
    question = "Which goals are achieved?"
    nodes = {"goal": None, "requirement": None}
    query, _ = await create_cypher_query(question, nodes)
    validate_cypher_format(query)
    assert "IS NOT NULL" in query, "None entities must appear with IS NOT NULL"

@pytest.mark.asyncio
async def test_create_cypher_query_special_characters():
    question = "How does problem @#$%^ relate to others?"
    nodes = {"problem": ["@#$%^", "Problem (X)"]}
    query, _ = await create_cypher_query(question, nodes)
    validate_cypher_format(query)

@pytest.mark.asyncio
async def test_create_cypher_query_no_list():
    question = "What are the problems?"
    nodes = {"problem": "valueA"}
    query, _ = await create_cypher_query(question, nodes)
    validate_cypher_format(query)