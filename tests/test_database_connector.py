import pytest
from app.data.database_connector import *

def test_extract_unique_entities_basic_case():
    fake_records = [
        {
            "p1.name": "Problem A",
            "p1.description": "A description of Problem A",
            "labels(p1)": ["problem"],
            "p2.name": "Problem B",
            "p2.description": "Another problem",
            "labels(p2)": ["problem"],
            "c.name": "Context X",
            "c.description": "The environment",
            "labels(c)": ["context"]
        }
    ]

    result = extract_unique_entities(fake_records)
    
    assert "problems" in result["entities"]
    assert "contexts" in result["entities"]
    assert "Problem A" in result["entities"]["problems"]
    assert "Context X" in result["entities"]["contexts"]

    rel = result["relationships"]
    assert any(r["type"] == "arisesAt" for r in rel)
    assert any(r["from"] == "Problem A" and r["to"] == "Context X" for r in rel)
    assert any(r["from"] == "Problem B" and r["to"] == "Context X" for r in rel)


def test_extract_unique_entities_ignores_unknown_labels():
    fake_records = [
        {
            "x.name": "Mystery Node",
            "x.description": "Unknown entity",
            "labels(x)": ["unknown"]
        }
    ]
    result = extract_unique_entities(fake_records)
    assert all(len(v) == 0 for v in result["entities"].values())
    assert result["relationships"] == []

def test_extract_unique_entities_no_duplicates_in_relationships():
    fake_records = [
        {
            "p.name": "Problem A",
            "p.description": "Some issue",
            "labels(p)": ["problem"],
            "c.name": "Context Z",
            "c.description": "Where it happens",
            "labels(c)": ["context"]
        },
        {
            "p.name": "Problem A",
            "p.description": "Some issue",
            "labels(p)": ["problem"],
            "c.name": "Context Z",
            "c.description": "Where it happens",
            "labels(c)": ["context"]
        }
    ]
    result = extract_unique_entities(fake_records)
    rels = result["relationships"]
    assert len(rels) == 1  # Should not duplicate relationship


# -------- TESTS FUNCIONALES REALES -------- #

@pytest.mark.asyncio
async def test_get_similar_nodes_by_entity():
    dummy_embedding = [0.01] * 1536  # Suponiendo un embedding válido de 1536 dims
    result = await get_similar_nodes_by_entity("problem", dummy_embedding, threshold=0.0, top_k=3)

    assert isinstance(result, list)
    assert len(result) <= 3
    for name in result:
        assert isinstance(name, str)


@pytest.mark.asyncio
async def test_execute_query_structure():
    query = """
    MATCH (p1:problem), (c:context), (p2:problem)
    MATCH (p1)-[:arisesAt]->(c)<-[:arisesAt]-(p2)
    WHERE p1.name IS NOT NULL AND c.name IS NOT NULL AND p1 <> p2
    WITH DISTINCT p1, c, p2
    RETURN p1.name, p1.description, labels(p1),
           p2.name, p2.description, labels(p2),
           c.name, c.description, labels(c)
    """
    result = await execute_query(query)

    assert "entities" in result
    assert "relationships" in result
    assert isinstance(result["entities"], dict)
    assert isinstance(result["relationships"], list)