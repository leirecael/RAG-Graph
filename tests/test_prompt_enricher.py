import pytest
from app.logic.prompt_enricher import enrich_prompt

def test_enrich_prompt_basic_structure():
    question = "What problems affect users?"
    context = {
        "entities": {
            "problems": {
                "Latency Issue": {
                    "description": "The response time is too high.",
                    "labels": ["problem"]
                }
            },
            "stakeholders": {}
        },
        "relationships": [
            {
                "from": "Latency Issue",
                "to": "Developers",
                "type": "concerns"
            }
        ]
    }

    prompt = enrich_prompt(question, context)

    assert "### QUESTION" in prompt
    assert question in prompt
    assert "### ENTITIES" in prompt
    assert "### PROBLEMS" in prompt
    assert "**Latency Issue**" in prompt
    assert "The response time is too high." in prompt
    assert "[problem]" in prompt
    assert "### RELATIONSHIPS" in prompt
    assert "Latency Issue --[concerns]--> Developers" in prompt


def test_enrich_prompt_empty_entities():
    question = "Is there a problem?"
    context = {
        "entities": {
            "problems": {},
            "contexts": {}
        },
        "relationships": []
    }

    prompt = enrich_prompt(question, context)

    assert "### ENTITIES" in prompt
    assert "### RELATIONSHIPS" in prompt
    assert "Is there a problem?" in prompt
    # Should not contain any node lines
    assert "- **" not in prompt


def test_enrich_prompt_multiple_entities_and_relationships():
    question = "How do problems relate to goals in context A?"
    context = {
        "entities": {
            "problems": {
                "Data Loss": {
                    "description": "Loss of critical information.",
                    "labels": ["problem"]
                }
            },
            "goals": {
                "Data Integrity": {
                    "description": "Ensure no data is lost.",
                    "labels": ["goal"]
                }
            },
            "contexts": {
                "Context A": {
                    "description": "Specific operational setting.",
                    "labels": ["context"]
                }
            }
        },
        "relationships": [
            {"from": "Data Loss", "to": "Context A", "type": "arisesAt"},
            {"from": "Data Loss", "to": "Data Integrity", "type": "informs"}
        ]
    }

    prompt = enrich_prompt(question, context)

    assert "### GOALS" in prompt
    assert "Data Integrity" in prompt
    assert "### CONTEXTS" in prompt
    assert "Context A" in prompt
    assert "Data Loss --[arisesAt]--> Context A" in prompt
    assert "Data Loss --[informs]--> Data Integrity" in prompt