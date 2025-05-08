import pytest
import pytest_asyncio
from unittest.mock import AsyncMock
from app.llm.llm_client import *

@pytest.mark.parametrize("model,input_tokens,output_tokens,expected_cost", [
    ("gpt-4.1", 1000, 500, 0.002 * 1 + 0.008 * 0.5),
    ("gpt-3.5-turbo", 2000, 1000, 0.0005 * 2 + 0.0015 * 1),
])
def test_calculate_token_cost_by_parts(model, input_tokens, output_tokens, expected_cost):
    cost = calculate_token_cost(model, input_tokens=input_tokens, output_tokens=output_tokens)
    assert round(cost, 6) == round(expected_cost, 6)


def test_calculate_token_cost_total_tokens():
    cost = calculate_token_cost("text-embedding-3-small", total_tokens=1000)
    assert round(cost, 8) == 0.00002


def test_calculate_token_cost_unknown_model():
    with pytest.raises(ValueError):
        calculate_token_cost("unknown-model", input_tokens=100)


def test_truncate_prompt_within_limit():
    short_prompt = "Hello world!"
    encoding = MODEL_INFO["gpt-3.5-turbo"]["encoding"]
    result = truncate_prompt(short_prompt, encoding, max_tokens=100)
    assert result == short_prompt


def test_truncate_prompt_exceeds_limit():
    long_prompt = "token " * 2000
    encoding = MODEL_INFO["gpt-3.5-turbo"]["encoding"]
    truncated = truncate_prompt(long_prompt, encoding, max_tokens=10)
    assert isinstance(truncated, str)
    assert len(truncated.split()) <= 10 


@pytest_asyncio.fixture
def mock_client(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr("your_module.client", mock)
    return mock


@pytest.mark.asyncio
async def test_call_llm(mock_client):
    mock_response = AsyncMock()
    mock_response.usage.input_tokens = 1000
    mock_response.usage.output_tokens = 500
    mock_response.output_text = "Generated text"
    mock_client.responses.create.return_value = mock_response

    result, cost = await call_llm("Explain the context.", model="gpt-4.1")
    assert "Generated text" in result
    assert cost > 0


@pytest.mark.asyncio
async def test_call_llm_structured(mock_client):
    mock_response = AsyncMock()
    mock_response.usage.prompt_tokens = 800
    mock_response.usage.completion_tokens = 300
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "Structured content"
    mock_client.beta.chat.completions.parse.return_value = mock_response

    result, cost = await call_llm_structured("List problems.", model="gpt-4.1", text_format="question")
    assert "Structured content" in result
    assert cost > 0


@pytest.mark.asyncio
async def test_get_embedding(mock_client):
    mock_response = AsyncMock()
    mock_response.usage.total_tokens = 256
    mock_response.data = [AsyncMock()]
    mock_response.data[0].embedding = [0.1] * 10
    mock_client.embeddings.create.return_value = mock_response

    embedding, cost = await get_embedding("What is context?")
    assert isinstance(embedding, list)
    assert len(embedding) == 10
    assert cost > 0