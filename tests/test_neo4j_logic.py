from app.logic.neo4j_logic import generate_similarity_queries, parse_similarity_results, parse_related_nodes_results, remove_duplicate_text, remove_duplicate_text_in_list
from app.models.entity import Entity

#-----generate_similarity_queries---------
def test_generate_similarity_queries_single_entity():
    """
    Test that the function creates a valid Cypher query and parameter dictionary for a single Entity input.
    """
    entity = Entity(value="Problem A", type="problem", embedding=[0.1, 0.2, 0.3])
    result = generate_similarity_queries([entity])

    assert isinstance(result, list)
    assert len(result) == 1
    query_entry = result[0]
    assert "query" in query_entry and "params" in query_entry
    assert "$embedding" in query_entry["query"]
    assert "problem" in query_entry["query"]
    assert query_entry["params"]["embedding"] == [0.1, 0.2, 0.3]

def test_generate_similarity_queries_multiple_entities():
    """
    Test that the function generates a correct query for each entity in a list of multiple Entity inputs.
    Checks the entity type, similarity threshold, and limit per query.
    """
    entities = [
        Entity(value="Goal A", type="goal", embedding=[0.2, 0.4, 0.6]),
        Entity(value="Context B", type="context", embedding=[0.9, 0.1, 0.3])
    ]
    result = generate_similarity_queries(entities, threshold=0.7, top_k=5)

    assert len(result) == 2
    for query_entry, ent in zip(result, entities):
        assert ent.type in query_entry["query"]
        assert query_entry["params"]["embedding"] == ent.embedding
        assert "LIMIT 5" in query_entry["query"]
        assert f"WHERE similarity >= 0.7" in query_entry["query"]

def test_generate_similarity_queries_empty_input():
    """
    Test that passing an empty list of entities returns an empty list of queries.
    """
    result = generate_similarity_queries([])
    assert result == []

#------parse_similarity_results---------
def test_parse_similarity_results_groups_by_first_label():
    """
    Verify that similarity results are grouped correctly by the first label in the node's label list.
    """
    fake_results = [
        {"value": {"name": "Item A", "labels": ["goal", "secondary"]}},
        {"value": {"name": "Item B", "labels": ["goal"]}},
        {"value": {"name": "Item C", "labels": ["problem"]}}
    ]
    parsed = parse_similarity_results(fake_results)

    assert parsed == {
        "goal": ["Item A", "Item B"],
        "problem": ["Item C"]
    }


#------parse_related_nodes_results---------
def test_parse_related_nodes_results_others():
    """
    Test that 'others' values not mapped to entities are captured in the 'others' section.
    Also verifies deduplication in lists.
    """
    fake_records = [
        {
            "s.name": "dev",
            "stakeholderList": [
                "developers",
                "software engineers",
                "developers",
                "Software Engineers"
            ],
            "labels(s)": ["stakeholder"]
        }
    ]
    result = parse_related_nodes_results(fake_records)

    assert "others" in result
    assert isinstance(result["others"], dict)
    assert "stakeholderList" in result["others"]
    non_dup = result["others"]["stakeholderList"]

    # Expected list without duplicates, case-insensitive
    assert sorted(non_dup) == sorted(["developers", "software engineers"])

def test_parse_related_nodes_results_others_with_semicolon_lists():
    """
    Test deduplication and normalization of semicolon-separated items in string lists in the 'others' section.
    """
    fake_records = [
        {
            "note": ["Security ; performance ; security", " Usability ; usability "]
        }
    ]
    result = parse_related_nodes_results(fake_records)

    assert "others" in result
    values = result["others"]["note"]

    assert values == ["Security; performance", "Usability"]

def test_parse_related_nodes_results_with_missing_fields():
    """
    Ensure the parser handles missing optional fields like description and hypernym.
    """
    fake_records = [
        {
            "p.name": "Problem A",
            "labels(p)": ["problem"],
        }
    ]
    result = parse_related_nodes_results(fake_records)

    assert "Problem A" in result["entities"]["problems"]
    entity = result["entities"]["problems"]["Problem A"]
    assert entity["description"] == ""
    assert entity["hypernym"] == ""

def test_parse_related_nodes_results_with_multiple_labels():
    """
    Test that a node with multiple valid labels is added to at least one appropriate entity category.
    """
    fake_records = [
        {
            "x.name": "Hybrid Node",
            "x.description": "Multiple meanings",
            "labels(x)": ["problem", "goal"]
        }
    ]
    result = parse_related_nodes_results(fake_records)

    assert "Hybrid Node" in result["entities"]["problems"] or "Hybrid Node" in result["entities"]["goals"]

def test_parse_related_nodes_results_basic_case():
    """
    Full-flow test parsing problems and contexts and verifying relationship types like 'arisesAt'.
    """
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

    result = parse_related_nodes_results(fake_records)
    
    assert "problems" in result["entities"]
    assert "contexts" in result["entities"]
    assert "Problem A" in result["entities"]["problems"]
    assert "Context X" in result["entities"]["contexts"]

    rel = result["relationships"]
    assert any(r["type"] == "arisesAt" for r in rel)
    assert any(r["from"] == "Problem A" and r["to"] == "Context X" for r in rel)
    assert any(r["from"] == "Problem B" and r["to"] == "Context X" for r in rel)

def test_parse_related_nodes_results_ignores_unknown_labels():
    """
    Ensure that nodes with unknown labels are ignored and not added to entity maps.
    """
    fake_records = [
        {
            "x.name": "Mystery Node",
            "x.description": "Unknown entity",
            "labels(x)": ["unknown"]
        }
    ]
    result = parse_related_nodes_results(fake_records)
    assert all(len(v) == 0 for v in result["entities"].values())
    assert result["relationships"] == []

def test_parse_related_nodes_results_no_duplicates_in_relationships():
    """
    Ensure that duplicate node pairs do not result in duplicate relationships in the output.
    """
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
    result = parse_related_nodes_results(fake_records)
    rels = result["relationships"]
    assert len(rels) == 1 

def test_parse_related_nodes_results_includes_alternative_name():
    """
    Test that alternativeName is included when available on a node.
    """
    fake_records = [
        {
            "p.name": "Problem A",
            "p.description": "Some desc",
            "p.alternativeName": "Alt A",
            "labels(p)": ["problem"]
        }
    ]
    result = parse_related_nodes_results(fake_records)
    entity = result["entities"]["problems"]["Problem A"]
    assert entity["alternativeName"] == "Alt A"

#------remove_duplicate_text---------
def test_remove_duplicate_text_normalization():
    """
    Verify semicolon-separated text is deduplicated and normalized for spacing and case.
    """
    text = "Improves efficiency; improves efficiency ;   Security ; security ;"
    cleaned = remove_duplicate_text(text)
    assert cleaned == "Improves efficiency; Security"


#------remove_duplicate_text_in_list--------
def test_remove_duplicate_text_in_list_deduplicates_case_insensitive():
    """
    Test that the list deduplication function removes case-insensitive duplicates.
    """
    input_data = ["Security", "security", "SECURITY", "Efficiency", "efficiency "]
    cleaned = remove_duplicate_text_in_list(input_data)
    assert cleaned == ["Security", "Efficiency"]


















