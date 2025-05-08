from pydantic import BaseModel
from enum import Enum
from typing import Optional

class EntityEnum(str, Enum):
    problem = 'problem'
    context = 'context'
    goal = 'goal'
    requirement = 'requirement'
    artifactClass = 'artifactClass'
    stakeholder ='stakeholder'

class Entity(BaseModel):
    value: Optional[str]
    type: EntityEnum
    primary: bool
    embedding: Optional[list[float]]

    class Config:  
        use_enum_values = True

class EntityList(BaseModel):
    entities: list[Entity]