import pytest
from pydantic import ValidationError
from app.models.question import Question

def test_question_model_valid():
    """
    Test that a Question object is created successfully with valid required fields.

    Verifies:
        - The Question instance has the expected field values.
        - Optional fields like 'reasoning' can be None without error.
    """
    q = Question(
        value="What problems do developers face?",
        is_valid=True,
        reasoning=None,
    )
    assert q.is_valid is True
    assert q.reasoning is None

def test_question_model_requires_fields():
    """
    Test that omitting required fields when creating a Question raises ValidationError.

    Verifies:
        - The model enforces presence of all required fields.
        - ValidationError is raised if required fields are missing.
    """
    with pytest.raises(ValidationError):
        Question(value="Me and no more")