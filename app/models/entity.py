from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class EntityEnum(str, Enum):
    """
    Enum representing valid entity types.
    
    Values:
        - problem: A challenge or issue.
        - context: Domain or situation.
        - goal: An objective or desired outcome.
        - requirement: A specific need or condition.
        - artifactClass: A type of technical solution to a problem.
        - stakeholder: Person or group affected or interested.
    """
    problem = 'problem'
    context = 'context'
    goal = 'goal'
    requirement = 'requirement'
    artifactClass = 'artifactClass'
    stakeholder ='stakeholder'

class Entity(BaseModel):
    """
    A single entity object that includes type, value, and optional embedding.

    Attributes:
        value (Optional[str]): The text associated with the entity found explicitly in the question.
        type (EntityEnum): The type of entity.
        embedding (Optional[list[float]]): Vector representation (embedding) of value.
    """
    value: Optional[str] = Field(...,description="The textual content of the entity (e.g., 'optimize performance'). None if not found explicitly in the question")   
    type: EntityEnum = Field(...,description="The type of the entity (e.g., 'goal', 'context').")
    embedding: Optional[list[float]] = Field(...,description="Optional embedding vector for the entity's value.")

    class Config:  
        use_enum_values = True #Serialize enums as raw values instead of Enum.name

class EntityList(BaseModel):
    """
    A list of entities.

    Attributes:
        entities (list[Entity]): List of entities.
    """
    entities: list[Entity] = Field(...,description="List of entities extracted from the question.")