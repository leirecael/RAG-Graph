from openai import AsyncOpenAI
from config.config import OPENAI_API_KEY
import openai
import tiktoken
from datetime import datetime
import time
from models.entity import EntityList
from models.question import Question
from data.logger import log_data

openai.api_key = OPENAI_API_KEY
client = AsyncOpenAI()

#Costes abril 2025 por 1k tokens
MODEL_INFO = {
    "gpt-4.1-mini": {"input_price": 0.0004, "output_price": 0.0016, "max_context": 1047576, "max_output":32768, "encoding":"o200k_base"},
    "gpt-4.1-nano": {"input_price": 0.0001, "output_price": 0.0004, "max_context": 1047576, "max_output":32768, "encoding":"o200k_base"},
    "gpt-4.1": {"input_price": 0.002, "output_price": 0.008, "max_context": 1047576, "max_output":32768, "encoding":"o200k_base"},
    "gpt-3.5-turbo": {"input_price": 0.0005, "output_price": 0.0015, "max_context": 16385, "max_output":4096, "encoding":"cl100k_base"},
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013
}

RESPONSE_FORMAT = {
    "entitylist": EntityList,
    "question": Question
}

async def call_llm(user_prompt, system_prompt, model="gpt-4.1", temperature=0.7, task_name = None):
    start_time = time.time()
    max_tokens= MODEL_INFO[model]["max_context"] - MODEL_INFO[model]["max_output"]
    truncated_prompt = truncate_prompt(user_prompt, MODEL_INFO[model]["encoding"], max_tokens)
    response = await client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": truncated_prompt}
        ],
        temperature=temperature,
    )
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = calculate_token_cost(model,input_tokens, output_tokens)
    duration_sec = time.time() - start_time
    log_data({
        "timestamp": datetime.now().isoformat(),
        "log_type": "llm_call",
        "task_name": task_name,
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": truncated_prompt,
        "response": response.output_text,
        "temperature": temperature,
        "tokens": {
            "input": input_tokens,
            "output": output_tokens
        },
        "cost": cost,
        "log_duration_sec": duration_sec
    })
    return response.output_text, cost

async def call_llm_structured(user_prompt, system_prompt, model="gpt-4.1", temperature=0.7, text_format=None, task_name = None):
    start_time = time.time()
    max_tokens= MODEL_INFO[model]["max_context"] - MODEL_INFO[model]["max_output"]
    truncated_prompt = truncate_prompt(user_prompt, MODEL_INFO[model]["encoding"], max_tokens)
    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": truncated_prompt}
        ],
        temperature=temperature,
        response_format=RESPONSE_FORMAT[text_format]
    )
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = calculate_token_cost(model,input_tokens, output_tokens)
    duration_sec = time.time() - start_time   

    log_data({
        "timestamp": datetime.now().isoformat(),
        "log_type": "llm_call",
        "task_name": task_name,
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": truncated_prompt,
        "response": response.choices[0].message.content,
        "temperature": temperature,
        "tokens": {
            "input": input_tokens,
            "output": output_tokens
        },
        "cost": cost,
        "log_duration_sec": duration_sec
    })

    return response.choices[0].message.content, cost

async def get_embedding(text, model="text-embedding-3-large", task_name = None):
    start_time = time.time()
    response = await client.embeddings.create(
        model=model,
        input=text
    )
    total_tokens = response.usage.total_tokens
    cost = calculate_token_cost(model, total_tokens=total_tokens)
    duration_sec = time.time() - start_time
    log_data({
        "timestamp": datetime.now().isoformat(),
        "log_type": "embedding",
        "task_name": task_name,
        "model": model,
        "input": text,
        "tokens": total_tokens,
        "cost": cost,
        "log_duration_sec": duration_sec
    })
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
