import pytest
from pydantic import ValidationError
from app.models.question import Question

def test_question_model_valid():
    """
    Test that a Question object is correctly created when all required fields are valid.
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
    Test that creating a Question without all required fields raises a ValidationError.
    """
    with pytest.raises(ValidationError):
        Question(value="Me and no more")