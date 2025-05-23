import pytest
from pydantic import ValidationError
from app.models.entity import *

def test_entity_model_valid():
    """
    Test that a valid Entity instance is created correctly.

    Verifies:
        - The 'value' is correctly pared as a string.
        - The 'type' is stored as its string value due to 'use_enum_values = True'.
        - The 'embedding' is correctly parsed as a list.
    """
    e = Entity(
        value="high latency",
        type=EntityEnum.problem,
        embedding=[0.1, 0.2, 0.3]
    )
    assert isinstance(e.value, str)
    assert e.type == "problem"
    assert isinstance(e.embedding, list)

def test_entity_model_invalid_type():
    """
    Test that using an invalid enum value for 'type' raises a ValidationError.

    Verifies:
        - Pydantic correctly rejects values not in the EntityEnum.
    """
    with pytest.raises(ValidationError):
        Entity(value="X", type="invalid")

def test_entity_list_valid():
    """
    Test that a valid EntityList can be created with multiple Entity instances.

    Verifies:
        - EntityList accepts a list of valid Entity objects.
        - The objects maintain their structure and enum types.
        - Optional fields (like value or embedding) can be None.
    """
    entities = [
        Entity(value="developers", type=EntityEnum.stakeholder, embedding=None),
        Entity(value=None, type=EntityEnum.problem, embedding=None)
    ]
    el = EntityList(entities=entities)
    assert len(el.entities) == 2
    assert el.entities[0].type == "stakeholder"
    assert el.entities[1].value == None