from app.logic.neo4j_logic import Neo4jLogic
from app.models.entity import Entity
import pytest

test_logic = Neo4jLogic()

#-----generate_similarity_queries---------
@pytest.mark.parametrize("entities,expected_len", [
    ([Entity(value="X", type="problem", embedding=[0.1, 0.2])], 1),
    ([Entity(value="X", type="goal", embedding=[0.1, 0.2]),
      Entity(value="Y", type="context", embedding=[0.3, 0.4])], 2),
])
def test_generate_similarity_queries_variants(entities, expected_len):
    """
    Test that generate_similarity_queries creates a valid query and params for a single and multiple entities.

    Verifies:
        - The returned list contains the expected query dictionaries.
        - The query references the entity type.
        - The parameters of embedding, top_k and threshold matches the input ones.
    """
    result = test_logic.generate_similarity_queries(entities, threshold=0.7, top_k=5)
    assert isinstance(result, list)
    assert len(result) == expected_len
    for entry, ent in zip(result, entities):
        assert ent.type in entry["query"]
        assert entry["params"]["embedding"] == ent.embedding
        assert entry["params"]["top_k"] == 5
        assert entry["params"]["threshold"] == 0.7

def test_generate_similarity_queries_empty_input():
    """
    Test that generate_similarity_queries returns an empty list when given no entities.

    Verifies:
        - The function safely returns an empty list with no errors.
    """
    result = test_logic.generate_similarity_queries([])
    assert result == []

def test_generate_similarity_queries_unknown_label():
    """
    Test that only the allowed labels will get into the query.
    
    Verifies:
        - ValueError is raised with an unknown label.
    """
    #Create a fake Entity class, as the real one will not allow other type of labels.
    class EntityFake:
        def __init__(self, value, type, embedding):
            self.value = value
            self.type = type
            self.embedding = embedding

    entities = [
        EntityFake(value="Goal A", type="FAKE", embedding=[0.2, 0.4, 0.6]),
        EntityFake(value="Context B", type="context", embedding=[0.9, 0.1, 0.3])
    ]

    with pytest.raises(ValueError):
        test_logic.generate_similarity_queries(entities, threshold=0.7, top_k=5)

#-----generate_similarity_queries_no_label---------
@pytest.mark.parametrize("entities,expected_len", [
    ([Entity(value="X", type="problem", embedding=[0.1, 0.2])], 1),
    ([Entity(value="X", type="goal", embedding=[0.1, 0.2]),
      Entity(value="Y", type="context", embedding=[0.3, 0.4])], 2),
])
def test_generate_similarity_queries_no_label_variants(entities, expected_len):
    """
    Test that generate_similarity_queries_no_label creates a valid query and params for a single and multiple entities.

    Verifies:
        - The returned list contains the expected query dictionaries.
        - The query does not care about the entity type.
        - The parameters of embedding, top_k and threshold matches the input ones.
    """
    result = test_logic.generate_similarity_queries_no_label(entities, threshold=0.7, top_k=5)
    assert isinstance(result, list)
    assert len(result) == expected_len
    for entry, ent in zip(result, entities):
        assert ent.type not in entry["query"]
        assert entry["params"]["embedding"] == ent.embedding
        assert entry["params"]["top_k"] == 5
        assert entry["params"]["threshold"] == 0.7

def test_generate_similarity_queries_no_label_empty_input():
    """
    Test that generate_similarity_queries_no_label returns an empty list when given no entities.

    Verifies:
        - The function safely returns an empty list with no errors.
    """
    result = test_logic.generate_similarity_queries_no_label([])
    assert result == []

#------parse_similarity_results---------
def test_parse_similarity_results_groups_by_first_label():
    """
    Test that parse_similarity_results groups nodes by their first label.

    Verifies:
        - Nodes are grouped correctly by the first label in their label list.
        - Different labels produce separate groups.
    """
    fake_results = [
        {"value": {"name": "Item A", "labels": ["goal", "secondary"]}},
        {"value": {"name": "Item B", "labels": ["goal"]}},
        {"value": {"name": "Item C", "labels": ["problem"]}}
    ]
    parsed = test_logic.parse_similarity_results(fake_results)

    assert parsed == {
        "goal": ["Item A", "Item B"],
        "problem": ["Item C"]
    }

def test_parse_similarity_results_empty_results():
    """
    Test that parse_similarity_results works with empty results.

    Verifies:
        - Empty results do not return error.
    """
    parsed = test_logic.parse_similarity_results([])

    assert parsed == {}

#------parse_related_nodes_results---------
def test_parse_related_nodes_results_others_and_deduplication():
    """
    Test that parse_related_nodes_results captures unmapped values under 'others' and deduplicates them.

    Verifies:
        - 'others' dictionary contains expected keys and values.
        - String with ; are deduplicated.
        - Lists are deduplicated.
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
            "labels(s)": ["stakeholder"],
            "stakeholderCount": 4,
            "stakeholderDesc": "description;description",
            "note": ["Security ; performance ; security", " Usability ; usability "]
        }
    ]
    result = test_logic.parse_related_nodes_results(fake_records)

    assert "others" in result
    assert isinstance(result["others"], dict)
    assert "stakeholderList" in result["others"]
    assert "stakeholderCount" in result["others"]
    assert result["others"]["stakeholderCount"] == 4
    assert "stakeholderDesc" in result["others"]
    assert result["others"]["stakeholderDesc"] == "description"

    values = result["others"]["note"]

    assert values == ["Security; performance", "Usability"]

    non_dup = result["others"]["stakeholderList"]

    # Expected list without duplicates
    assert sorted(non_dup) == sorted(["developers", "software engineers"])

def test_parse_related_nodes_results_with_multiple_labels():
    """
    Test that nodes with multiple valid labels are included in appropriate entity groups.

    Verifies:
        - The node appears under all valid categories.
    """
    fake_records = [
        {
            "x.name": "Hybrid Node",
            "x.description": "Multiple meanings",
            "labels(x)": ["problem", "goal"]
        }
    ]
    result = test_logic.parse_related_nodes_results(fake_records)

    assert "Hybrid Node" in result["entities"]["problems"] and "Hybrid Node" in result["entities"]["goals"]

def test_parse_related_nodes_results_basic_case():
    """
    Test that parse_related_nodes_results parses problems, contexts, and their relationships correctly.

    Verifies:
        - Problems and contexts are included in entities.
        - Relationships like 'arisesAt' exist linking the correct nodes.
        - Missing 'hypernym' field defaults to empty string.
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

    result = test_logic.parse_related_nodes_results(fake_records)
    
    assert "problems" in result["entities"]
    assert "contexts" in result["entities"]
    assert "Problem A" in result["entities"]["problems"]
    assert "Context X" in result["entities"]["contexts"]

    entity = result["entities"]["problems"]["Problem A"]
    assert entity["hypernym"] == ""

    rel = result["relationships"]
    assert any(r["type"] == "arisesAt" for r in rel)
    assert any(r["from"] == "Problem A" and r["to"] == "Context X" for r in rel)
    assert any(r["from"] == "Problem B" and r["to"] == "Context X" for r in rel)

def test_parse_related_nodes_results_ignores_unknown_labels():
    """
    Test that nodes with unknown labels are ignored.

    Verifies:
        - Entities dictionaries are empty.
        - No relationships are returned.
    """
    fake_records = [
        {
            "x.name": "Mystery Node",
            "x.description": "Unknown entity",
            "labels(x)": ["unknown"]
        }
    ]
    result = test_logic.parse_related_nodes_results(fake_records)
    assert all(not values for values in result["entities"].values())
    assert result["relationships"] == []

def test_parse_related_nodes_results_no_duplicates_in_relationships():
    """
    Test that duplicate node pairs do not cause duplicate relationships.

    Verifies:
        - Relationships list contains no duplicates for the same node pairs.
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
    result = test_logic.parse_related_nodes_results(fake_records)
    rels = result["relationships"]
    assert len(rels) == 1 

def test_parse_related_nodes_results_includes_alternative_name():
    """
    Test that alternativeName is included in entities when present.

    Verifies:
        - The alternativeName field is present and correctly set.
    """
    fake_records = [
        {
            "p.name": "Problem A",
            "p.description": "Some desc",
            "p.alternativeName": "Alt A",
            "labels(p)": ["problem"]
        }
    ]
    result = test_logic.parse_related_nodes_results(fake_records)
    entity = result["entities"]["problems"]["Problem A"]
    assert entity["alternativeName"] == "Alt A"

#------remove_duplicate_text---------
def test_remove_duplicate_text_normalization():
    """
    Test that remove_duplicate_text normalizes and deduplicates semicolon-separated strings.

    Verifies:
        - Duplicate segments are removed case-insensitively.
        - The output string is trimmed and normalized.
    """
    text = "Improves efficiency; improves efficiency ;   Security ; security ;"
    cleaned = test_logic.remove_duplicate_text(text)
    assert cleaned == "Improves efficiency; Security"


#------remove_duplicate_text_in_list--------
def test_remove_duplicate_text_in_list_deduplicates_case_insensitive():
    """
    Test that remove_duplicate_text_in_list removes duplicates from a list case-insensitively.

    Verifies:
        - Case-insensitive duplicates are removed.
        - The original order of first occurrences is preserved.
    """
    input_data = ["Security", "security", "SECURITY", "Efficiency", "efficiency "]
    cleaned = test_logic.remove_duplicate_text_in_list(input_data)
    assert cleaned == ["Security", "Efficiency"]


















