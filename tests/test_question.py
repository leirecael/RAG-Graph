import pytest
from pydantic import ValidationError
from app.models.question import Question

def test_question_model_valid():
    q = Question(
        value="What problems do developers face?",
        is_valid=True,
        reasoning=None,
        is_simple=None
    )
    assert q.is_valid is True
    assert q.reasoning is None

def test_question_model_requires_fields():
    with pytest.raises(ValidationError):
        Question(value="Only value provided")