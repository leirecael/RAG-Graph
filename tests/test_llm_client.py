import pytest
from app.llm.llm_client import *

#-------calculate_token_cost------
def test_calculate_token_cost_llm():
    """
    Test cost calculation for a llm model (gpt-4.1).

    Verifies:
        - Cost is calculated based on input and output tokens.
        - Used the correct costs from MODEL_INFO.
    """
    cost = calculate_token_cost("gpt-4.1", input_tokens=1000, output_tokens=500)
    expected_cost = (1000/1000) * 0.002 + (500/1000) * 0.008 #Pricing per 1k tokens

    assert round(cost, 6) == round(expected_cost, 6)


def test_calculate_token_cost_embedding():
    """
    Test cost calculation for an embedding model.

    Verifies:
        - Cost is calculated using total_tokens.
    """
    cost = calculate_token_cost("text-embedding-3-large", total_tokens=1000)

    assert round(cost, 6) == 0.00013 #1k tokens cost 0.00013$


def test_calculate_token_cost_unknown_model():
    """
    Test that using an unrecognized model raises a ValueError.

    Verifies:
        - Only models in MODEL_INFO are accepted.
    """
    with pytest.raises(ValueError):
        calculate_token_cost("unknown-model", input_tokens=100, output_tokens=100)

def test_calculate_token_cost_wrong_tokens():
    """
    Test that incorrect combinations of token arguments raise a ValueError.

    Verifies:
        - Must provide both input and output tokens for LLMs.
        - Must not pass total_tokens to a non-embedding model.
    """
    with pytest.raises(ValueError):
        calculate_token_cost("gpt-4.1-nano", input_tokens=100) #Must have output_token
    with pytest.raises(ValueError):
        calculate_token_cost("gpt-4.1", total_tokens=100) #Cannot use total_tokens

#-------truncate_prompt------
def test_truncate_prompt_within_limit():
    """
    Test prompt that does not exceed the token limit.

    Verifies:
        - Prompt does not change.
        - 'truncated' flag is False.
    """
    short_prompt = "Short text"
    encoding = MODEL_INFO["gpt-4.1-nano"]["encoding"]
    result, truncated = truncate_prompt(short_prompt, encoding, max_tokens=100)

    assert result == short_prompt
    assert truncated is False


def test_truncate_prompt_exceeds_limit():
    """
    Test that a prompt exceeding the token limit is truncated.

    Verifies:
        - Output prompt length is reduced to the max tokens length allowed.
        - 'truncated' flag is True.
        - Output is a string.
    """
    long_prompt = "long text " * 2000
    encoding_name = MODEL_INFO["gpt-4.1-nano"]["encoding"]
    encoding = tiktoken.get_encoding(encoding_name)

    result, truncated = truncate_prompt(long_prompt, encoding_name, max_tokens=10)
    token_count = len(encoding.encode(result))

    assert isinstance(result, str)
    assert token_count == 10 
    assert truncated is True

#-----call_llm------
@pytest.mark.asyncio
async def test_call_llm_raises_value_errors(): 
    """
    Test that the expected errors are raised.

    Verifies:
        - Unknown model raises ValueError
        - Temperature above 2 raises ValueError
    """
    prompt = "Hello"
    system_prompt="Greetings"
    
    with pytest.raises(ValueError):
        await call_llm(prompt, system_prompt, "unknown model")
    with pytest.raises(ValueError):
        await call_llm(prompt, system_prompt, temperature=3.0)

#-----call_llm_structured------
@pytest.mark.asyncio
async def test_call_llm_structured_raises_value_errors(): 
    """
    Test that the expected errors are raised.

    Verifies:
        - Unknown model raises ValueError
        - Temperature below 0 raises ValueError
        - Unknown text format raises ValueError
    """
    prompt = "Hello"
    system_prompt="Greetings"
    
    with pytest.raises(ValueError):
        await call_llm_structured(prompt, system_prompt, text_format="question", model="unknown model 2")
    with pytest.raises(ValueError):
        await call_llm_structured(prompt, system_prompt, text_format="question", temperature=-3.0)
    with pytest.raises(ValueError):
        await call_llm_structured(prompt, system_prompt, text_format="unknown")

#-----get_embedding------
@pytest.mark.asyncio
async def test_get_embedding_raises_value_errors():
    """
    Test that the expected errors are raised.

    Verifies:
        - Unknown model raises ValueError
    """ 
    prompt="Text"

    with pytest.raises(ValueError):
        await get_embedding(prompt, model="unknown")