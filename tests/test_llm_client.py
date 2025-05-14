import pytest
import pytest_asyncio
from app.llm.llm_client import *

@pytest.mark.parametrize("model,input_tokens,output_tokens,expected_cost", [
    ("gpt-4.1", 1000, 500, 0.002 * 1 + 0.008 * 0.5),
    ("gpt-3.5-turbo", 2000, 1000, 0.0005 * 2 + 0.0015 * 1),
])
def test_calculate_token_cost_by_parts(model, input_tokens, output_tokens, expected_cost):
    cost = calculate_token_cost(model, input_tokens=input_tokens, output_tokens=output_tokens)
    assert round(cost, 6) == round(expected_cost, 6)


def test_calculate_token_cost_total_tokens():
    cost = calculate_token_cost("text-embedding-3-large", total_tokens=1000)
    assert round(cost, 8) == 0.00013


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
    assert len(truncated.strip()) > 0

#FALTAN LOS CALL Y EMBEDDING