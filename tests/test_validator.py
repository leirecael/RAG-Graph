import pytest
from app.logic.validator import validate_question
from app.models.question import Question

@pytest.mark.asyncio
async def test_validate_question_negative():
    question = "What is the weather in Paris?"
    response, cost = await validate_question(question)

    assert isinstance(response, Question)
    assert response.is_valid is False
    assert isinstance(response.reasoning, str)
    assert isinstance(cost, float)

@pytest.mark.asyncio
async def test_validate_question_positive():
    question = "How can we address climate change?"
    response, cost = await validate_question(question)

    assert isinstance(response, Question)
    assert response.is_valid is True
    assert isinstance(response.reasoning, str)
    assert isinstance(cost, float)