from pydantic import BaseModel, Field
from typing import Optional

class Question(BaseModel):
    """
    A model representing a user question and its validation result.

    Attributes:
        value (str): The raw text of the user's question.
        is_valid (bool): Indicates whether the question passed validation.
        reasoning (Optional[str]): Explanation for why the question is invalid.
    """
    value: str = Field(...,description="The question's text")
    is_valid: bool = Field(...,description="If the question is valid or not")
    reasoning: Optional[str] = Field(...,description="The reason the question is not valid")