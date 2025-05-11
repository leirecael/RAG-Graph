import pytest
from app.logic.validator import validate_question
from app.models.question import Question
import json

@pytest.mark.asyncio
async def test_validate_question_negative():
    question = "What is the weather in Paris?"
    response, cost = await validate_question(question)

    response_dict = json.loads(response)
    response_obj = Question.model_validate(response_dict)

    assert isinstance(response_obj, Question)
    assert response_obj.is_valid is False
    assert isinstance(response_obj.reasoning, str)
    assert isinstance(cost, float)

@pytest.mark.asyncio
async def test_validate_question_positive():
    question = "How can we address climate change?"
    response, cost = await validate_question(question)

    response_dict = json.loads(response)
    response_obj = Question.model_validate(response_dict)

    assert isinstance(response_obj, Question)
    assert response_obj.is_valid is True
    assert response_obj.reasoning is None
    assert isinstance(cost, float)