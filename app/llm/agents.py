from agents import Agent
from models.question import ValidQuestionOutput

valid_question_agent = Agent(
    name="Check Question Validity",
    instructions=(
        """You are an assistant that checks if a question is appropriate for a database of scientific and technological papers.
        Return true only if the question relates to research, technical topics, or similar content. Here is some information about the database schema:
        - (problem)-[:arisesAt]->(context)
        - (problem)-[:concerns]->(stakeholder)
        - (problem)-[:informs]->(goal)
        - (requirement)-[:meetBy]->(artifactClass)
        - (problem)-[:addressedBy]->(artifactClass)
        - (goal)-[:achievedBy]->(requirement)"""
    ),
    output_type=ValidQuestionOutput,
)