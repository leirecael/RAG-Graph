import pytest
import json
from app.models.entity import *
from app.logic.question_processor import extract_entities

@pytest.mark.asyncio
async def test_extract_entities_basic():
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
async def test_extract_entities_complex():
    question = "How to address problem X in context Y?"
    response, cost = await extract_entities(question)

    response_dict = json.loads(response)
    response_obj = EntityList.model_validate(response_dict)

    assert isinstance(response_obj, EntityList)
    types = [e.type for e in response_obj.entities]
    assert "problem" in types
    assert "context" in types
    assert "artifactClass" in types