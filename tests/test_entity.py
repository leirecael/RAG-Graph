import pytest
from pydantic import ValidationError
from app.models.entity import *

def test_entity_model_valid():
    e = Entity(
        value="performance",
        type=EntityEnum.problem,
        primary=True,
        embedding=[0.1, 0.2, 0.3]
    )
    assert e.type == "problem"
    assert e.primary is True
    assert isinstance(e.embedding, list)

def test_entity_model_invalid_type():
    with pytest.raises(ValidationError):
        Entity(value="X", type="invalid", primary=False)

def test_entity_list_valid():
    entities = [
        Entity(value="devs", type=EntityEnum.stakeholder, primary=False, embedding=None),
        Entity(value=None, type=EntityEnum.problem, primary=True, embedding=None)
    ]
    el = EntityList(entities=entities)
    assert len(el.entities) == 2
    assert el.entities[0].type == "stakeholder"