import pytest
import json
from logic.orchestrator import process_question

@pytest.mark.asyncio
async def test_process_question_happy_path(mocker):
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

    mocker.patch("logic.orchestrator.get_embedding", return_value=(["0.1", "0.2"], 0.003))

    mocker.patch("logic.orchestrator.generate_similarity_queries", return_value=["SIMILARITY_QUERY"])
    mocker.patch("logic.orchestrator.execute_multiple_queries_with_apoc", return_value={
        "stakeholder": ["developers"]
    })

    mocker.patch("logic.orchestrator.create_cypher_query", return_value=("MATCH ...", 0.001))

    mocker.patch("logic.orchestrator.execute_query", return_value={
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
        ]
    })

    # Mock enrichment and LLM
    mocker.patch("logic.orchestrator.enrich_prompt", return_value=("Prompt", "System"))
    mocker.patch("logic.orchestrator.call_llm", return_value=("They face latency issues.", 0.01))

    # Call the function
    response = await process_question("What problems do developers face?")
    assert isinstance(response, str)
    assert "latency" in response.lower()


@pytest.mark.asyncio
async def test_process_question_invalid_question(mocker):
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

    mocker.patch("logic.orchestrator.get_embedding", return_value=(["0.1", "0.2"], 0.003))

    mocker.patch("logic.orchestrator.generate_similarity_queries", return_value=["SIM_QUERY"])
    mocker.patch("logic.orchestrator.execute_multiple_queries_with_apoc", return_value={
        "stakeholder": ["doctors"]
    })

    mocker.patch("logic.orchestrator.create_cypher_query", return_value=("MATCH ...", 0.001))

    mocker.patch("logic.orchestrator.execute_query", return_value={
        "entities": {},
        "relationships": []
    })

    response = await process_question("What problems do doctors face?")
    assert "no available information" in response.lower()