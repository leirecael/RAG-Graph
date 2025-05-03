from openai import AsyncOpenAI
from config.config import OPENAI_API_KEY
import openai

openai.api_key = OPENAI_API_KEY
client = AsyncOpenAI()

#Costes abril 2025 por 1k tokens
MODEL_PRICING = {
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4.1": {"input": 0.002, "output": 0.008},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "text-embedding-3-small": 0.00002
}

async def call_llm(prompt, model="gpt-4.1", temperature=0.7):
    response = await client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = calculate_token_cost(model,input_tokens, output_tokens)
    return response.output_text, cost

async def get_embedding(text):
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    total_tokens = response.usage.total_tokens
    cost = calculate_token_cost("text-embedding-3-small", total_tokens=total_tokens)
    return response.data[0].embedding, cost

def calculate_token_cost(model, input_tokens=0, output_tokens=0, total_tokens=0) -> float:
    if model not in MODEL_PRICING:
        raise ValueError(f"Unknown model pricing for: {model}")
    pricing = MODEL_PRICING[model]
    if total_tokens > 0:
        return (total_tokens / 1000) * pricing
    return (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]