import pytest
import json
from logic.orchestrator import process_question, contains_pii, sanitize_input
from app.models.entity import Entity

#------contains_pii---------
def test_contains_pii_positive():
    """
    Test that PII is correctly identified in the text.
    """
    text_with_pii = "My name is Pedro and my email is pedro@gmail.com"
    assert contains_pii(text_with_pii) is True


def test_contains_pii_negative():
    """
    Test that non-PII text does not trigger false positives.
    """
    safe_text = "What problems do software developers face?"
    assert contains_pii(safe_text) is False

#------sanitize_input---------
def test_sanitize_input_removes_special_characters():
    """
    Test that sanitize_input removes punctuation and special characters while preserving spaces and alphanumerics.
    """
    raw_text = "I have a question~~&---"
    cleaned = sanitize_input(raw_text)
    assert cleaned == "I have a question"


#------process_question---------
@pytest.mark.asyncio
async def test_process_question_basic_path(mocker):
    """
    Full end-to-end basic path:
    - Valid question
    - Entities found
    - Database returns related nodes
    - Final natural language response is generated
    """
    mocker.patch("logic.orchestrator.contains_pii", return_value = False)
    
    mocker.patch("logic.orchestrator.sanitize_input", return_value = "What problems do developers face?")

    mocker.patch("logic.orchestrator.validate_question", return_value=(json.dumps({
        "value": "What problems do developers face?",
        "is_valid": True,
        "reasoning": None
    }), 0.001))

    mocker.patch("logic.orchestrator.extract_entities", return_value=(json.dumps({
        "entities": [
            {"value": "developers", "type": "stakeholder", "embedding": None},
            {"value": None, "type": "problem", "embedding": None}
        ]
    }), 0.002))

    mocker.patch("logic.orchestrator.generate_entity_embeddings", return_value=([Entity(value='developers', type='stakeholder', embedding=[-0.013434951193630695, 0.013434951193630695]), Entity(value=None, type='problem', embedding=None)], 0.1))

    mocker.patch("logic.orchestrator.generate_similarity_queries", return_value=["SIMILARITY_QUERY"])
    mocker.patch("logic.orchestrator.execute_multiple_queries", return_value=[{"value":{"name": "developers", "similarity": 0.7, "labels":["stakeholder"]}}])
    mocker.patch("logic.orchestrator.parse_similarity_results", return_value={
        "stakeholder": ["developers"]
    })

    mocker.patch("logic.orchestrator.create_cypher_query", return_value=("MATCH ...", 0.001))

    mocker.patch("logic.orchestrator.execute_query", return_value={
        "records": {"record": "Info"},
    })
    mocker.patch("logic.orchestrator.parse_related_nodes_results", return_value={
        "entities": {
            "problems": {
                "Latency Issue": {
                    "description": "High response time",
                    "labels": ["problem"],
                    "hypernym": "performance"
                }
            },
            "stakeholders": {
                "developers": {
                    "description": "People writing code.",
                    "labels": ["stakeholder"],
                    "hypernym": "team members"
                }
            }
        },
        "relationships": [
            {"from": "Latency Issue", "to": "developers", "type": "concerns"}
        ],
        "others": []
    })

    # Mock enrichment and LLM
    mocker.patch("logic.orchestrator.generate_final_answer", return_value=("They face latency issues.", 0.01))

    # Call the function
    response = await process_question("What problems do developers face?")
    assert isinstance(response, str)
    assert "latency" in response.lower()


@pytest.mark.asyncio
async def test_process_question_invalid_question(mocker):
    """
    Simulates a user asking a question that is valid structurally but outside the expected domain.
    """
    mocker.patch("logic.orchestrator.validate_question", return_value=(json.dumps({
        "value": "What's the weather?",
        "is_valid": False,
        "reasoning": "Non-technical domain."
    }), 0.001))

    response = await process_question("What's the weather?")
    assert "not valid" in response.lower()
    assert "non-technical" in response.lower()


@pytest.mark.asyncio
async def test_process_question_no_entities_found(mocker):
    """
    Tests the scenario where a question is valid but no useful entities are found in the graph.
    """
    mocker.patch("logic.orchestrator.validate_question", return_value=(json.dumps({
        "value": "What problems do doctors face?",
        "is_valid": True,
        "reasoning": None
    }), 0.001))

    mocker.patch("logic.orchestrator.extract_entities", return_value=(json.dumps({
        "entities": [
            {"value": "doctors", "type": "stakeholder", "embedding": None}
        ]
    }), 0.002))

    mocker.patch("logic.orchestrator.generate_entity_embeddings", return_value=([Entity(value='doctors', type='stakeholder', embedding=[-0.013434951193630695, 0.013434951193630695]), Entity(value=None, type='problem', embedding=None)], 0.1))
    mocker.patch("logic.orchestrator.parse_similarity_results", return_value={
        "stakeholder": ["doctors"]
    })

    mocker.patch("logic.orchestrator.generate_similarity_queries", return_value=["SIM_QUERY"])
    mocker.patch("logic.orchestrator.execute_multiple_queries", return_value=[{"value":{"name": "doctors", "similarity": 0.7, "labels":["stakeholder"]}}])

    mocker.patch("logic.orchestrator.create_cypher_query", return_value=("MATCH ...", 0.001))

    mocker.patch("logic.orchestrator.execute_query", return_value={
        "records": {},
    })
    mocker.patch("logic.orchestrator.parse_related_nodes_results", return_value={
        "entities": {},
        "relationships": [],
        "others": []
    })

    response = await process_question("What problems do doctors face?")
    assert "no available information" in response.lower()