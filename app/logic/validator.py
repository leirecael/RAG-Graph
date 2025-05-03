from models.question import ValidQuestionOutput
from agents import Runner
from llm.agents import valid_question_agent
from llm.hooks import MyHooks
from llm.llm_client import calculate_token_cost

myhooks = MyHooks()

async def validate_question(question) -> ValidQuestionOutput:
    myhooks.reset()
    result = await Runner.run(valid_question_agent, hooks=myhooks, input=question)
    cost = calculate_token_cost(
        "gpt-4.1",
        myhooks.total_input_tokens,
        myhooks.total_output_tokens,
    )
    return result.final_output,cost


