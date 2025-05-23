from openai import AsyncOpenAI
from config.config import OPENAI_API_KEY
import openai
import tiktoken
from datetime import datetime
import time
from models.entity import EntityList
from models.question import Question
from logs.logger import log_data

#Set API key and create OpenAI client
openai.api_key = OPENAI_API_KEY
client = AsyncOpenAI()

#April 2025 pricing(US$) per 1,000 tokens for each model, encoding type and token usage limits. Max_content is the maximun tokens the model can use per call(input/output). Max_output is for the maximum token output allowed per call.
MODEL_INFO = {
    "gpt-4.1-mini": {"input_price": 0.0004, "output_price": 0.0016, "max_context": 1047576, "max_output":32768, "encoding":"o200k_base"},
    "gpt-4.1-nano": {"input_price": 0.0001, "output_price": 0.0004, "max_context": 1047576, "max_output":32768, "encoding":"o200k_base"},
    "gpt-4.1": {"input_price": 0.002, "output_price": 0.008, "max_context": 1047576, "max_output":32768, "encoding":"o200k_base"},
    "gpt-3.5-turbo": {"input_price": 0.0005, "output_price": 0.0015, "max_context": 16385, "max_output":4096, "encoding":"cl100k_base"},
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013
}

#Mapping from response format name to data model class
RESPONSE_FORMAT = {
    "entitylist": EntityList,
    "question": Question
}

async def call_llm(user_prompt:str, system_prompt:str, model:str="gpt-4.1", temperature:float=0.7, task_name:str = None)->tuple[str, float]:
    """
    Calls an OpenAI LLM.

    Args:
        user_prompt (str): The user's prompt text.
        system_prompt (str): System-level prompt instructions.
        model (str): OpenAI model name.
        temperature (float): Sampling temperature.
        task_name (str, optional): Logging task name.

    Returns:
        tuple[str, float]: The generated response and the cost.
    """
    start_time = time.time()
    if model not in MODEL_INFO:
        raise ValueError(f"Unknown model: {model}")
    
    #Calculate max allowed input tokens
    max_tokens= MODEL_INFO[model]["max_context"] - MODEL_INFO[model]["max_output"]
    truncated_prompt, truncated = truncate_prompt(user_prompt, MODEL_INFO[model]["encoding"], max_tokens)

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

    #Log LLM call data
    log_data({
        "timestamp": datetime.now().isoformat(),
        "log_type": "llm_call",
        "task_name": task_name,
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": truncated_prompt,
        "response": response.output_text,
        "truncated": truncated,
        "temperature": temperature,
        "tokens": {
            "input": input_tokens,
            "output": output_tokens
        },
        "cost": cost,
        "log_duration_sec": duration_sec
    })
    return response.output_text, cost

async def call_llm_structured(user_prompt: str, system_prompt:str, text_format:str, model:str ="gpt-4.1", temperature:float=0.7, task_name:str = None)->tuple[str, float]:
    """
    Calls an OpenAI LLM with structured output parsing.

    Args:
        user_prompt (str): The user's input text.
        system_prompt (str): System-level prompt instructions.
        text_format (str): Key in RESPONSE_FORMAT dict for expected structured format.
        model (str): OpenAI model name.
        temperature (float): Sampling temperature.
        task_name (str, optional): Logging task name.

    Returns:
        tuple[str, float]: Structured model response content and its cost.
    """
    start_time = time.time()
    if model not in MODEL_INFO:
        raise ValueError(f"Unknown model: {model}")
    
    #Calculate max allowed input tokens
    max_tokens= MODEL_INFO[model]["max_context"] - MODEL_INFO[model]["max_output"]
    truncated_prompt, truncated = truncate_prompt(user_prompt, MODEL_INFO[model]["encoding"], max_tokens)

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

    #Log LLM call data
    log_data({
        "timestamp": datetime.now().isoformat(),
        "log_type": "llm_call",
        "task_name": task_name,
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": truncated_prompt,
        "response": response.choices[0].message.content,
        "truncated": truncated,
        "temperature": temperature,
        "tokens": {
            "input": input_tokens,
            "output": output_tokens
        },
        "cost": cost,
        "log_duration_sec": duration_sec
    })
    return response.choices[0].message.content, cost

async def get_embedding(text:str, model:str="text-embedding-3-large", task_name:str = None)->tuple[list[float],float]:
    """
    Calls OpenAI's embedding endpoint to calculate the vector of the input text.

    Args:
        text (str): The input text to embed.
        model (str): The embedding model to use.
        task_name (str, optional): Logging task name.

    Returns:
        tuple[list[float], float]: The embedding vector and its cost.
    """
    start_time = time.time()
    if model not in MODEL_INFO:
        raise ValueError(f"Unknown model: {model}")
    
    response = await client.embeddings.create(
        model=model,
        input=text
    )

    total_tokens = response.usage.total_tokens
    cost = calculate_token_cost(model, total_tokens=total_tokens)
    duration_sec = time.time() - start_time

    #Log embedding endpoint call data
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

def calculate_token_cost(model:str, input_tokens:int=0, output_tokens:int=0, total_tokens:int=0) -> float:
    """
    Calculates the cost in US$ for a model call based on tokens used and model pricing.

    Args:
        model (str): Model name.
        input_tokens (int): Number of input tokens.
        output_tokens (int): Number of output tokens.
        total_tokens (int): Used for embedding models.

    Returns:
        float: Cost in USD.
    """
    if model not in MODEL_INFO:
        raise ValueError(f"Unknown model pricing for: {model}")
    
    pricing = MODEL_INFO[model]

    #Embedding model price
    if isinstance(pricing, float) and total_tokens > 0 and input_tokens == 0 and output_tokens == 0:
        return (total_tokens / 1000) * pricing
    #LLM model price
    elif "input_price" in pricing and "output_price" in pricing and total_tokens == 0 and input_tokens > 0 and output_tokens > 0:
        return (input_tokens / 1000) * pricing["input_price"] + (output_tokens / 1000) * pricing["output_price"]
    else:
        raise ValueError("Token values are incorrect.")

def truncate_prompt(prompt:str,model:str, max_tokens:int) -> tuple[str,bool]:
    """
    Truncates a prompt to fit within token limits for a model.

    Args:
        prompt (str): Full text prompt.
        model (str): Encoding model for tiktoken.
        max_tokens (int): Maximum number of tokens allowed.

    Returns:
        str: Truncated prompt if necessary.
    """
    encodig = tiktoken.get_encoding(model)
    tokens = encodig.encode(prompt)

    if len(tokens)<max_tokens:
        return prompt, False
    
    truncated = tokens[:max_tokens]
    return encodig.decode(truncated), True
