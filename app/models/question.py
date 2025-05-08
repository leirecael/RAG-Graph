from pydantic import BaseModel, Field
from typing import Optional

class Question(BaseModel):
    value: str = Field(...,description="The question's text")
    is_valid: bool = Field(...,description="If the question is valid or not")
    reasoning: Optional[str] = Field(...,description="The reason the queestion is not valid")
    is_simple: Optional[bool] = Field(...,description="If the question is considered simple or not")