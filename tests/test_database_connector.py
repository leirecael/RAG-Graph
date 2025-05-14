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
    assert len(rels) == 1 


def test_execute_query_structure():
    query = """
    MATCH (p1:problem), (c:context), (p2:problem)
    MATCH (p1)-[:arisesAt]->(c)<-[:arisesAt]-(p2)
    WHERE p1.name IS NOT NULL AND c.name IS NOT NULL AND p1 <> p2
    WITH DISTINCT p1, c, p2
    RETURN p1.name, p1.description, labels(p1),
           p2.name, p2.description, labels(p2),
           c.name, c.description, labels(c)
    """
    result = execute_query(query)

    assert "entities" in result
    assert "relationships" in result
    assert isinstance(result["entities"], dict)
    assert isinstance(result["relationships"], list)

def test_extract_entities_with_missing_fields():
    fake_records = [
        {
            "p.name": "Problem A",
            "labels(p)": ["problem"],
        }
    ]
    result = extract_unique_entities(fake_records)

    assert "Problem A" in result["entities"]["problems"]
    entity = result["entities"]["problems"]["Problem A"]
    assert entity["description"] == ""
    assert entity["hypernym"] == ""

def test_extract_entities_with_multiple_labels():
    fake_records = [
        {
            "x.name": "Hybrid Node",
            "x.description": "Multiple meanings",
            "labels(x)": ["problem", "goal"]
        }
    ]
    result = extract_unique_entities(fake_records)

    assert "Hybrid Node" in result["entities"]["problems"] or "Hybrid Node" in result["entities"]["goals"]

def test_remove_redundant_text_normalization():
    text = "Improves efficiency; improves efficiency ;   Security ; security ;"
    cleaned = remove_redundant_text(text)
    assert cleaned == "Improves efficiency; Security"

def test_execute_query_invalid_cypher():
    with pytest.raises(Exception):
        execute_query("THIS IS NOT CYPHER")

def test_execute_multiple_queries_with_apoc_real_case():
    queries = [{
        "query": "MATCH (n:problem) RETURN n.name as name, labels(n) as labels",
        "params": {}
    }]
    result = execute_multiple_queries_with_apoc(queries)
    assert isinstance(result, dict)

def test_extract_unique_entities_handles_others():
    fake_records = [
        {
            "source": "User input",
            "stakeholderList": [
                "developers",
                "software engineers",
                "developers",
                "Software Engineers"
            ],
            "labels(x)": ["unknown"]
        }
    ]
    result = extract_unique_entities(fake_records)

    assert "others" in result
    assert isinstance(result["others"], dict)
    assert "stakeholderList" in result["others"]
    deduped = result["others"]["stakeholderList"]

    # Expect deduped list, case-insensitive
    assert sorted(deduped) == sorted(["developers", "software engineers"])

def test_remove_redundant_text_in_list_deduplicates_case_insensitive():
    input_data = ["Security", "security", "SECURITY", "Efficiency", "efficiency "]
    cleaned = remove_redundant_text_in_list(input_data)
    assert cleaned == ["Security", "Efficiency"]

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
            "labels(c)": ["context"],
            "source": "User Feedback"
        }
    ]

    result = extract_unique_entities(fake_records)

    assert "problems" in result["entities"]
    assert "contexts" in result["entities"]
    assert "others" in result
    assert result["others"].get("source") == "User Feedback"

def test_extract_unique_entities_others_with_semicolon_lists():
    fake_records = [
        {
            "note": ["Security ; performance ; security", " Usability ; usability "]
        }
    ]
    result = extract_unique_entities(fake_records)

    assert "others" in result
    values = result["others"]["note"]

    assert values == ["Security; performance", "Usability"]