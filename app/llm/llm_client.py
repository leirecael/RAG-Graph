from openai import AsyncOpenAI
from config.config import OPENAI_API_KEY
import openai
import tiktoken
from models.entity import EntityList
from models.question import Question

openai.api_key = OPENAI_API_KEY
client = AsyncOpenAI()

#Costes abril 2025 por 1k tokens
MODEL_INFO = {
    "gpt-4.1-mini": {"input_price": 0.0004, "output_price": 0.0016, "max_context": 1047576, "max_output":32768, "encoding":"o200k_base"},
    "gpt-4.1": {"input_price": 0.002, "output_price": 0.008, "max_context": 1047576, "max_output":32768, "encoding":"o200k_base"},
    "gpt-3.5-turbo": {"input_price": 0.0005, "output_price": 0.0015, "max_context": 16385, "max_output":4096, "encoding":"cl100k_base"},
    "text-embedding-3-small": 0.00002
}

RESPONSE_FORMAT = {
    "entitylist": EntityList,
    "question": Question
}

async def call_llm(prompt, model="gpt-4.1", temperature=0.7):
    max_tokens= MODEL_INFO[model]["max_context"] - MODEL_INFO[model]["max_output"]
    prompt = truncate_prompt(prompt, MODEL_INFO[model]["encoding"], max_tokens)
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

async def call_llm_structured(prompt, model="gpt-4.1", temperature=0.7, text_format=None):
    max_tokens= MODEL_INFO[model]["max_context"] - MODEL_INFO[model]["max_output"]
    prompt = truncate_prompt(prompt, MODEL_INFO[model]["encoding"], max_tokens)
    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        response_format=RESPONSE_FORMAT[text_format]
    )
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = calculate_token_cost(model,input_tokens, output_tokens)
    return response.choices[0].message.content, cost

async def get_embedding(text):
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    total_tokens = response.usage.total_tokens
    cost = calculate_token_cost("text-embedding-3-small", total_tokens=total_tokens)
    return response.data[0].embedding, cost

def calculate_token_cost(model, input_tokens=0, output_tokens=0, total_tokens=0) -> float:
    if model not in MODEL_INFO:
        raise ValueError(f"Unknown model pricing for: {model}")
    pricing = MODEL_INFO[model]
    if total_tokens > 0:
        return (total_tokens / 1000) * pricing
    return (input_tokens / 1000) * pricing["input_price"] + (output_tokens / 1000) * pricing["output_price"]

def truncate_prompt(prompt,model, max_tokens):
    encodig = tiktoken.get_encoding(model)
    tokens = encodig.encode(prompt)
    if len(tokens)<max_tokens:
        return prompt
    truncated = tokens[:max_tokens]
    print("TRUNCATED PROMPT")
    return encodig.decode(truncated)
