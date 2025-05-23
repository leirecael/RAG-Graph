import pytest
from app.logic.llm_tasks import validate_question, extract_entities, create_cypher_query, enrich_prompt, generate_entity_embeddings
from app.models.question import Question
from app.models.entity import *
import json

#----------validate_question---------
@pytest.mark.asyncio
async def test_validate_question_invalid():
    """
    Test that a question outside the domain (e.g. about weather) is marked as invalid.
    """
    question = "What is the weather in Paris?"
    response, cost = await validate_question(question)

    response_dict = json.loads(response)
    response_obj = Question.model_validate(response_dict)

    assert isinstance(response_obj, Question)
    assert response_obj.is_valid is False
    assert isinstance(response_obj.reasoning, str)
    assert isinstance(cost, float)

@pytest.mark.asyncio
async def test_validate_question_valid():
    """
    Test that a question inside the domain is marked as valid.
    """
    question = "How can we address climate change?"
    response, cost = await validate_question(question)

    response_dict = json.loads(response)
    response_obj = Question.model_validate(response_dict)

    assert isinstance(response_obj, Question)
    assert response_obj.is_valid is True
    assert response_obj.reasoning is None
    assert isinstance(cost, float)

@pytest.mark.asyncio
async def test_validate_question_spelling_correction():
    """
    Test that the validator corrects poorly spelled questions.
    """
    question = "Hw cn featre modelz help?"
    response, _ = await validate_question(question)

    response_obj = Question.model_validate(json.loads(response))
    assert response_obj.is_valid is True
    assert response_obj.value != question 

#----------extract_entities---------
@pytest.mark.asyncio
async def test_extract_entities_basic():
    """
    Test that basic entities are correctly extracted.
    """
    question = "What problems do developers face?"
    response, cost = await extract_entities(question)

    response_dict = json.loads(response)
    response_obj = EntityList.model_validate(response_dict)

    assert isinstance(response_obj, EntityList)
    entity_types = [e.type for e in response_obj.entities]
    assert "stakeholder" in entity_types
    assert "problem" in entity_types
    assert isinstance(cost, float)

@pytest.mark.asyncio
async def test_extract_entities_none():
    """
    Test that irrelevant questions return no entities.
    """
    question = "What's the weather today?"
    response, _ = await extract_entities(question)

    response_dict = json.loads(response)
    response_obj = EntityList.model_validate(response_dict)

    assert response_obj.entities == []

@pytest.mark.asyncio
async def test_extract_entities_with_null_values():
    """
    Test that entity values can be None when they are abstract or implicit.
    """
    question = "What problems are solved by the same artifact?"
    response, _ = await extract_entities(question)

    response_dict = json.loads(response)
    response_obj = EntityList.model_validate(response_dict)

    problem = [e for e in response_obj.entities if e.type == "problem"]
    artifactClass = [e for e in response_obj.entities if e.type == "problem"]
    assert (p.value is None for p in problem)
    assert (a.value is None for a in artifactClass)

#--------generate_entity_embedding----------
@pytest.mark.asyncio
async def test_generate_entity_embeddings_with_valid_entities():
    """
    Test function with a list of valid entities.
    Embeddings should be assigned and total cost > 0.
    """
    entities = [Entity(value="AI", type=EntityEnum.context, embedding=None), Entity(value="developers", type=EntityEnum.stakeholder, embedding=None)]

    updated_entities, total_cost = await generate_entity_embeddings(entities)

    for entity in updated_entities:
        if entity.value:
            assert isinstance(entity.embedding, list)
            assert all(isinstance(x, float) for x in entity.embedding)
            assert len(entity.embedding) > 0

    assert total_cost > 0.0

@pytest.mark.asyncio
async def test_generate_entity_embeddings_with_none_values():
    """
    Test function when some entities have None values.
    Only entities with actual values should get embeddings.
    """
    entities = [Entity(value="AI", type=EntityEnum.context, embedding=None), Entity(value=None, type=EntityEnum.problem, embedding=None)]

    updated_entities, total_cost = await generate_entity_embeddings(entities)

    assert updated_entities[0].embedding is not None
    assert updated_entities[1].embedding is None
    assert total_cost > 0.0

@pytest.mark.asyncio
async def test_generate_entity_embeddings_empty_list():
    """
    Test function with an empty list.
    Should return empty list and zero cost.
    """
    updated_entities, total_cost = await generate_entity_embeddings([])

    assert updated_entities == []
    assert total_cost == 0.0

#--------create_cypher_query----------
def validate_cypher_format(query: str):
    """
    Validates that a Cypher query follow the rules and is formated correctly.

    Args:
        query (str): The cypher query that need validation.
    """
    assert isinstance(query,str), "Query must be a string"
    assert "MATCH" in query.upper(), "Must contain MATCH"
    assert "RETURN" in query.upper(), "Must contain RETURN"
    assert query.count("(") == query.count(")"), "Parenthesis open-close quantity should be the same"
    assert "cypher" not in query.upper(), "Query should contain only the cypher query"
    assert "CREATE" not in query.upper(), "CREATE is forbidden"
    assert "DELETE" not in query.upper(), "DELETE is forbidden"
    assert "SET" not in query.upper(), "SET is forbidden"
    assert "MERGE" not in query.upper(), "MERGE is forbidden"
    

@pytest.mark.asyncio
async def test_create_cypher_query_basic_call():
    """
    Test Cypher generation for a valid question and nodes.
    """
    question = "Can problems A and B be addressed by X?"
    nodes = {
        "problem": ["Problem A", "Problem B"],
        "artifactClass": ["System X"]
    }

    query, cost = await create_cypher_query(question, nodes)

    validate_cypher_format(query)
    assert cost > 0, "Cypher generation must have a cost"

@pytest.mark.asyncio
async def test_create_cypher_query_empty_nodes():
    """
    Test Cypher generation with no available nodes.
    """
    question = "What problems are addressed by the same artifactClass?"
    nodes = {}
    query, _ = await create_cypher_query(question, nodes)
    assert len(query) == 0, "Cypher generation must return nothing"

@pytest.mark.asyncio
async def test_create_cypher_query_nodes_with_none_values():
    """
    Test Cypher generation when node values are None.
    """
    question = "Which goals are achieved?"
    nodes = {"goal": None, "requirement": None}
    query, _ = await create_cypher_query(question, nodes)
    validate_cypher_format(query)
    assert "IS NOT NULL" in query, "None entities must appear with IS NOT NULL"

@pytest.mark.asyncio
async def test_create_cypher_query_special_characters():
    """
    Test Cypher generation with special characters in node names.
    """
    question = "How does problem @#$%^ relate to others?"
    nodes = {"problem": ["@#$%^", "Problem (X)"]}
    query, _ = await create_cypher_query(question, nodes)
    validate_cypher_format(query)

@pytest.mark.asyncio
async def test_create_cypher_query_no_list():
    """
    Test Cypher generation when node value is a string instead of a list.
    """
    question = "What are the problems?"
    nodes = {"problem": "valueA"}
    query, _ = await create_cypher_query(question, nodes)
    validate_cypher_format(query)

#--------enrich_prompt----------
def test_enrich_prompt_basic_structure():
    """
    Test prompt enrichment for a standard context with entities, relationships and others.
    """
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
    """
    Test that prompt includes alternative names for entities if present.
    """
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

def test_enrich_prompt_with_missing_fields():
    """
    Test that missing or empty fields do not break prompt generation.
    """
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
    assert "[]" in prompt 

def test_enrich_prompt_ignores_empty_entity_categories():
    """
    Test that empty entity types are not included in the prompt.
    """
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
    """
    Test full prompt generation with multiple entities and relationships.
    """
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

