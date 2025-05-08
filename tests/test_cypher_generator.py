import pytest
from app.logic.cypher_generator import create_cypher_query

@pytest.mark.asyncio
async def test_create_cypher_query_real_call():
    question = "What problems are addressed by the same artifactClass?"
    nodes = {
        "problem": ["Problem A", "Problem B"],
        "artifactClass": ["System X"]
    }

    query, cost = await create_cypher_query(question, nodes)

    assert isinstance(query, str)
    assert "MATCH" in query
    assert "RETURN" in query
    assert cost > 0