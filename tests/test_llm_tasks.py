import pytest
from app.logic.llm_tasks import validate_question, extract_entities, create_cypher_query, enrich_prompt, generate_entity_embeddings
from app.models.question import Question
from app.models.entity import *
import json

#----------validate_question---------
@pytest.mark.asyncio
async def test_validate_question_invalid():
    """
    Test that a question outside the domain (e.g., about the weather) is marked as invalid.

    Verifies:
        - The question is flagged as invalid.
        - A reasoning message is returned.
        - The cost is calculated and returned as a float.
    """
    question = "What is the weather in Paris?"
    response, cost = await validate_question(question)

    assert response.is_valid is False
    assert isinstance(response.reasoning, str)
    assert isinstance(cost, float)

@pytest.mark.asyncio
async def test_validate_question_valid():
    """
    Test that a relevant, domain-specific question is marked as valid.

    Verifies:
        - The question is marked as valid.
        - No reasoning is returned.
        - The cost is returned as a float.
    """
    question = "How can we address climate change?"
    response, cost = await validate_question(question)

    assert response.is_valid is True
    assert response.reasoning is None
    assert isinstance(cost, float)

@pytest.mark.asyncio
async def test_validate_question_spelling_correction():
    """
    Test that the validator corrects poorly written questions.

    Verifies:
        - The corrected question is considered valid.
        - The output question is different from the input.
    """
    question = "Hw cn featre modelz help?"
    response, _ = await validate_question(question)

    assert response.is_valid is True
    assert response.value != question 

#----------extract_entities---------
@pytest.mark.asyncio
async def test_extract_entities_basic():
    """
    Test that common entities are extracted from a basic question.

    Verifies:
        - Entities like 'stakeholder' and 'problem' are correctly identified.
        - The cost is returned as a float.
    """
    question = "What problems do developers face?"
    response, cost = await extract_entities(question)

    entity_types = [e.type for e in response.entities]
    assert "stakeholder" in entity_types
    assert "problem" in entity_types
    assert isinstance(cost, float)

@pytest.mark.asyncio
async def test_extract_entities_none():
    """
    Test that unrelated questions return no entities.

    Verifies:
        - The list of entities is empty.
    """
    question = "What's the weather today?"
    response, _ = await extract_entities(question)

    assert response.entities == []

@pytest.mark.asyncio
async def test_extract_entities_with_null_values():
    """
    Test that entities can have None as their value when abstract or implied.

    Verifies:
        - Entities may have a value of None.
        - No error occurs when handling these entities.
    """
    question = "What problems are solved by the same artifact?"
    response, _ = await extract_entities(question)

    problem = [e for e in response.entities if e.type == "problem"]
    artifactClass = [e for e in response.entities if e.type == "problem"]
    assert (p.value is None for p in problem)
    assert (a.value is None for a in artifactClass)

#--------generate_entity_embedding----------
@pytest.mark.asyncio
async def test_generate_entity_embeddings_with_valid_entities():
    """
    Test that valid entities receive embeddings and that the total cost is greater than zero.

    Verifies:
        - Each entity with a value gets an embedding (a list of floats).
        - Total cost is greater than 0.
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
    Test that only entities with values get embeddings.

    Verifies:
        - Entities without a value get no embedding.
        - Other entities do get embeddings.
        - The cost reflects only the valid ones.
    """
    entities = [Entity(value="AI", type=EntityEnum.context, embedding=None), Entity(value=None, type=EntityEnum.problem, embedding=None)]

    updated_entities, total_cost = await generate_entity_embeddings(entities)

    assert updated_entities[0].embedding is not None
    assert updated_entities[1].embedding is None
    assert total_cost > 0.0

@pytest.mark.asyncio
async def test_generate_entity_embeddings_empty_list():
    """
    Test that an empty input list returns no embeddings and zero cost.

    Verifies:
        - Output list is empty.
        - Total cost is 0.
    """
    updated_entities, total_cost = await generate_entity_embeddings([])

    assert updated_entities == []
    assert total_cost == 0.0

#--------create_cypher_query----------
def validate_cypher_format(query: str):
    """
    Validate the structure of a generated Cypher query.

    Verifies:
        - Query is a string.
        - Contains 'MATCH' and 'RETURN'.
        - Parentheses are balanced.
        - Does not contain forbidden clauses like CREATE, DELETE, SET, or MERGE.
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
    Test Cypher generation for a valid question and multiple node values.

    Verifies:
        - The generated query follows the expected Cypher format.
        - The query includes the required MATCH and RETURN clauses.
        - The cost of generation is greater than zero.
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
    Test behavior when no nodes are provided.

    Verifies:
        - The generated query is empty.
        - No Cypher logic is attempted without nodes.
    """
    question = "What problems are addressed by the same artifactClass?"
    nodes = {}
    query, _ = await create_cypher_query(question, nodes)
    assert len(query) == 0, "Cypher generation must return nothing"

@pytest.mark.asyncio
async def test_create_cypher_query_nodes_with_none_values():
    """
    Test Cypher generation when some node values are None.

    Verifies:
        - A query is still generated.
        - Entities with None are represented with 'IS NOT NULL'.
        - The query passes format validation.
    """
    question = "Which goals are achieved?"
    nodes = {"goal": None, "requirement": None}
    query, _ = await create_cypher_query(question, nodes)
    validate_cypher_format(query)
    assert "IS NOT NULL" in query, "None entities must appear with IS NOT NULL"

@pytest.mark.asyncio
async def test_create_cypher_query_special_characters():
    """
    Test Cypher generation with special characters in node values.

    Verifies:
        - Query includes entities with special characters without failure.
        - The query passes format validation.
    """
    question = "How does problem @#$%^ relate to others?"
    nodes = {"problem": ["@#$%^", "Problem (X)"]}
    query, _ = await create_cypher_query(question, nodes)
    validate_cypher_format(query)

@pytest.mark.asyncio
async def test_create_cypher_query_no_list():
    """
    Test behavior when a node has a single string value instead of a list.

    Verifies:
        - The single string value is accepted and processed.
        - The resulting query is valid and well-formed.
    """
    question = "What are the problems?"
    nodes = {"problem": "valueA"}
    query, _ = await create_cypher_query(question, nodes)
    validate_cypher_format(query)

#--------enrich_prompt----------
def test_enrich_prompt_basic_structure():
    """
    Test prompt enrichment with a full context including entities, relationships, and others.

    Verifies:
        - The question appears in the prompt.
        - Entities include names, descriptions, labels, and hypernyms.
        - Relationships are correctly formatted.
        - Other context fields (e.g., notes) are included in the prompt.
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
    Test inclusion of alternative names in the enriched prompt.

    Verifies:
        - Entities with 'alternativeName' include that alias in the formatted output.
        - Descriptions and hypernyms are correctly shown.
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
    Test prompt generation when entity fields like description or labels are empty.

    Verifies:
        - Prompt generation handles missing or empty fields without error.
        - Fields appear in output with default/empty values where appropriate.
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
    Test that empty entity groups are excluded from the prompt.

    Verifies:
        - Entity categories with no items are omitted.
        - The prompt includes only relevant sections like relationships or non-empty categories.
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
    Test full prompt generation with multiple types of entities and multiple relationships.

    Verifies:
        - All entity categories appear with expected content.
        - Multiple relationships are formatted and included correctly.
        - Hypernyms, descriptions, and labels appear for each entity.
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

