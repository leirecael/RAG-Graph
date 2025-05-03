from pydantic import BaseModel

class ValidQuestionOutput(BaseModel):
    is_valid_question: bool
    reasoning: str