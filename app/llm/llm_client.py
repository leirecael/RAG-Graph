from openai import AsyncOpenAI, AuthenticationError
from config.config import OPENAI_API_KEY
import openai
import tiktoken
from datetime import datetime
import time
from models.entity import EntityList
from models.question import Question
from logs.logger import Logger


class LlmClient:
    """
    Client to interact asynchronously with OpenAI's language models and embeddings endpoint.

    Contains pricing info, model limits, and methods to call LLMs with plain or structured output,
    generate embeddings, calculate costs, and truncate prompts.

    Attributes:
        client (AsyncOpenAI): Asynchronous OpenAI API client instance.
        logger (Logger): Logger instance for logging API usage and errors.
        MODEL_INFO (dict): Pricing and limits for each model.
        RESPONSE_FORMAT (dict): Mapping of response format keys to data model classes.

    Methods:
        call_llm(): Calls an OpenAI model with an user and system prompt, logs the interaction and does not expect a structured output.
        call_llm_structured(): Calls an OpenAI model with an user and system prompt, logs the interaction and expects a structured output.
        get_embedding(): Calls the embedding endpoint to embed a text.
        calculate_token_cost(): Calculates the total cost for the used tokens based on the model's price.
        truncate_prompt(): Shortens the user prompt if it exceeds the limit of the model it is going to be used on.
    """

    #Set API key
    openai.api_key = OPENAI_API_KEY

    #April 2025 pricing(US$) per 1,000 tokens for each model, encoding type and token usage limits. Max_content is the maximun tokens the model can use per call(input/output). Max_output is for the maximum token output allowed per call.
    MODEL_INFO = {
        "gpt-4.1-mini": {"input_price": 0.0004, "output_price": 0.0016, "max_context": 1047576, "max_output":32768, "encoding":"o200k_base"},
        "gpt-4.1-nano": {"input_price": 0.0001, "output_price": 0.0004, "max_context": 1047576, "max_output":32768, "encoding":"o200k_base"},
        "gpt-4.1": {"input_price": 0.002, "output_price": 0.008, "max_context": 1047576, "max_output":32768, "encoding":"o200k_base"},
        "text-embedding-3-small": 0.00002,
        "text-embedding-3-large": 0.00013
    }

    #Mapping from response format name to data model class
    RESPONSE_FORMAT = {
        "entitylist": EntityList,
        "question": Question
    }

    def __init__(self):
        """
        Initializes the LlmClient with a AsyncOpenAI instance for communicating with the OpenAI client and a Logger instance for logging data.
        """
        self.client = AsyncOpenAI()
        self.logger = Logger()

    async def call_llm(self, user_prompt:str, system_prompt:str, model:str="gpt-4.1", temperature:float=0.7, task_name:str = None)->tuple[str, float]:
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
        if model not in self.MODEL_INFO:
            raise ValueError(f"Unknown model: {model}")
        
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError(f"Temperatura is out of bounds (0-2): {temperature}")
        
        #Calculate max allowed input tokens
        max_tokens= self.MODEL_INFO[model]["max_context"] - self.MODEL_INFO[model]["max_output"]
        truncated_prompt, truncated = self.truncate_prompt(user_prompt, self.MODEL_INFO[model]["encoding"], max_tokens)

        try:
            response = await self.client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": truncated_prompt}
                ],
                temperature=temperature,
            )
        except AuthenticationError as e:
            raise RuntimeError("The API key is invalid or it was not configured.") from e

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        cost = self.calculate_token_cost(model,input_tokens, output_tokens)
        duration_sec = time.time() - start_time

        #Log LLM call data
        self.logger.log_data({
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

    async def call_llm_structured(self, user_prompt: str, system_prompt:str, text_format:str, model:str ="gpt-4.1", temperature:float=0.7, task_name:str = None)->tuple[Question|EntityList, float]:
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
            tuple[Question|EntityList, float]: Structured model response and its cost.
        """
        start_time = time.time()
        if model not in self.MODEL_INFO:
            raise ValueError(f"Unknown model: {model}")
        
        if temperature < 0.0 or temperature> 2.0:
            raise ValueError(f"Temperatura is out of bounds (0-2): {temperature}")
        
        if text_format not in self.RESPONSE_FORMAT:
            raise ValueError(f"Unknown output format: {text_format}")

        #Calculate max allowed input tokens
        max_tokens= self.MODEL_INFO[model]["max_context"] - self.MODEL_INFO[model]["max_output"]
        truncated_prompt, truncated = self.truncate_prompt(user_prompt, self.MODEL_INFO[model]["encoding"], max_tokens)

        try:
            response = await self.client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": truncated_prompt}
                ],
                temperature=temperature,
                text_format=self.RESPONSE_FORMAT[text_format]
            )
        except AuthenticationError as e:
            raise RuntimeError("The API key is invalid or it was not configured.") from e

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = self.calculate_token_cost(model,input_tokens, output_tokens)
        duration_sec = time.time() - start_time   

        #Log LLM call data
        self.logger.log_data({
            "timestamp": datetime.now().isoformat(),
            "log_type": "llm_call",
            "task_name": task_name,
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": truncated_prompt,
            "response": response.output_parsed.json(),
            "truncated": truncated,
            "temperature": temperature,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens
            },
            "cost": cost,
            "log_duration_sec": duration_sec
        })
        
        return response.output_parsed, cost

    async def get_embedding(self, text:str, model:str="text-embedding-3-large", task_name:str = None)->tuple[list[float],float]:
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
        if model not in self.MODEL_INFO:
            raise ValueError(f"Unknown model: {model}")
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=text
            )
        except AuthenticationError as e:
            raise RuntimeError("The API key is invalid or it was not configured.") from e
        total_tokens = response.usage.total_tokens
        cost = self.calculate_token_cost(model, total_tokens=total_tokens)
        duration_sec = time.time() - start_time

        #Log embedding endpoint call data
        self.logger.log_data({
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

    def calculate_token_cost(self, model:str, input_tokens:int=0, output_tokens:int=0, total_tokens:int=0) -> float:
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
        if model not in self.MODEL_INFO:
            raise ValueError(f"Unknown model pricing for: {model}")
        
        pricing = self.MODEL_INFO[model]

        #Embedding model price
        if isinstance(pricing, float) and total_tokens > 0 and input_tokens == 0 and output_tokens == 0:
            return (total_tokens / 1000) * pricing
        #LLM model price
        elif "input_price" in pricing and "output_price" in pricing and total_tokens == 0 and input_tokens > 0 and output_tokens > 0:
            return (input_tokens / 1000) * pricing["input_price"] + (output_tokens / 1000) * pricing["output_price"]
        else:
            raise ValueError("Token values are incorrect.")

    def truncate_prompt(self, prompt:str,model:str, max_tokens:int) -> tuple[str,bool]:
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
