import pytest
from app.models.entity import *
from app.logic.question_processor import extract_entities

@pytest.mark.asyncio
async def test_extract_entities_basic():
    question = "What problems do developers face?"
    response, cost = await extract_entities(question)

    assert isinstance(response, EntityList)
    entity_types = [e.type for e in response.entities]
    assert "stakeholder" in entity_types
    assert "problem" in entity_types
    assert any(e.primary is True for e in response.entities)
    assert isinstance(cost, float)

@pytest.mark.asyncio
async def test_extract_entities_complex():
    question = "How to address problem X in context Y?"
    response, cost = await extract_entities(question)

    assert isinstance(response, EntityList)
    types = [e.type for e in response.entities]
    assert "problem" in types
    assert "context" in types
    assert "artifactClass" in types
    assert sum(e.primary for e in response.entities) >= 1