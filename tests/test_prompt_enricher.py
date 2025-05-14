import pytest
from app.logic.prompt_enricher import enrich_prompt

def test_enrich_prompt_basic_structure():
    question = "What problems affect users?"
    context = {
        "entities": {
            "problems": {
                "Latency Issue": {
                    "description": "The response time is too high.",
                    "labels": ["problem"],
                    "hypernym": "connection issues"
                }
            },
            "stakeholders": {
                "Developers": {
                    "description": "People",
                    "labels": ["stakeholder"],
                    "hypernym": "people"
                }
            }
        },
        "relationships": [
            {
                "from": "Latency Issue",
                "to": "Developers",
                "type": "concerns"
            }
        ],
        "others": {
            "note": "Some general note"
        }
    }

    prompt, _ = enrich_prompt(question, context)

    assert "### QUESTION" in prompt
    assert question in prompt
    assert "### PROBLEMS" in prompt
    assert "**Latency Issue(;connection issues)**" in prompt
    assert "The response time is too high." in prompt
    assert "[problem]" in prompt
    assert "### RELATIONSHIPS" in prompt
    assert "Latency Issue --[concerns]--> Developers" in prompt
    assert "### OTHERS" in prompt
    assert "-note: Some general note" in prompt

def test_enrich_prompt_with_alternative_name():
    question = "Explain problem aliases"
    context = {
        "entities": {
            "problems": {
                "Crash Error": {
                    "description": "Unexpected application shutdown.",
                    "labels": ["problem"],
                    "hypernym": "software failure",
                    "alternativeName": "Fatal Crash"
                }
            }
        },
        "relationships": [],
        "others": {}
    }

    prompt, _ = enrich_prompt(question, context)
    assert "-**Crash Error(Fatal Crash;software failure)**" in prompt
    assert "Unexpected application shutdown." in prompt

def test_enrich_prompt_handles_missing_fields():
    question = "Test missing fields"
    context = {
        "entities": {
            "goals": {
                "Uptime": {
                    "description": "",
                    "labels": [],
                    "hypernym": "",
                    "alternativeName": ""
                }
            }
        },
        "relationships": [],
        "others": {}
    }

    prompt, _ = enrich_prompt(question, context)
    assert "**Uptime(;)**" in prompt  
    assert "[]\n" in prompt 

def test_enrich_prompt_ignores_empty_entity_categories():
    question = "What is missing?"
    context = {
        "entities": {
            "problems": {},
            "goals": {},
        },
        "relationships": [],
        "others": {}
    }

    prompt, _ = enrich_prompt(question, context)
    assert "### PROBLEMS" not in prompt
    assert "### GOALS" not in prompt
    assert "### RELATIONSHIPS" in prompt
    assert "- **" not in prompt  

def test_enrich_prompt_multiple_entities_and_relationships():
    question = "How do problems relate to goals in context A?"
    context = {
        "entities": {
            "problems": {
                "Data Loss": {
                    "description": "Loss of critical information.",
                    "labels": ["problem"],
                    "hypernym": "problem hyper"
                }
            },
            "goals": {
                "Data Integrity": {
                    "description": "Ensure no data is lost.",
                    "labels": ["goal"],
                    "hypernym": "goal hyper"
                }
            },
            "contexts": {
                "Context A": {
                    "description": "Specific operational setting.",
                    "labels": ["context"],
                    "hypernym": "context hyper"
                }
            }
        },
        "relationships": [
            {"from": "Data Loss", "to": "Context A", "type": "arisesAt"},
            {"from": "Data Loss", "to": "Data Integrity", "type": "informs"}
        ],
        "others": {}
    }

    prompt, _ = enrich_prompt(question, context)

    assert "### GOALS" in prompt
    assert "### PROBLEMS" in prompt
    assert "context hyper" in prompt
    assert "Data Integrity" in prompt
    assert "### CONTEXTS" in prompt
    assert "Context A" in prompt
    assert "Data Loss --[arisesAt]--> Context A" in prompt
    assert "Data Loss --[informs]--> Data Integrity" in prompt