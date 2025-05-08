from llm.llm_client import call_llm_structured, call_llm
import json

async def determine_simplicity(question: str) -> bool:
    prompt = f"""
        Your task is to classify the following question as **simple** or **complex** in the context of generating a Cypher query.

        ### Criteria:
        - **Simple**: only one instance per entity type is required (multiple types allowed).
        - **Complex**: the question requires multiple instances of the same entity type.

        Respond **only** in JSON format: {{"is_simple": true}}

        Examples:
        - What problems do developers face? → true  
        - What problems affect the same context? → false  
        - What goals are related to requirement R1? → true  
        - What requirements are linked to different problems in the same context? → false

        Question: {question}
    """

    response, cost = await call_llm(prompt)
    simplicity = json.loads(response)   
    return simplicity["is_simple"], cost

async def extract_entities(question: str) -> dict:
    prompt = f"""
        Extract relevant entities from the question. Mark with `primary=true` those that represent what the user wants to know (only one primary per entity type). If an entity is not clearly stated or inferred, do not include it.

        Entity types:
        - problem: a deficiency or issue
        - stakeholder: a person or group affected or interested
        - goal: desired high-level outcome (e.g. improvements)
        - context: environment or situation
        - requirement: functionality or condition the system must meet
        - artifactClass: general category of solution

        Relationships (Cypher):
        - (problem)-[:arisesAt]->(context)
        - (problem)-[:concerns]->(stakeholder)
        - (problem)-[:informs]->(goal)
        - (requirement)-[:meetBy]->(artifactClass)
        - (problem)-[:addressedBy]->(artifactClass)
        - (goal)-[:achievedBy]->(requirement)

        Format: list of objects with `value`, `type`, `primary`, and `embedding=null`.

        Example:
        Question: What problems do developers face?
        →
        [
            {{ "value": "developers", "type": "stakeholder", "primary": false, "embedding": null }},
            {{ "value": null, "type": "problem", "primary": true, "embedding": null }}
        ]

        Question: How to address X problem in Y?
        →
        [
            {{ "value": "X", "type": "problem", "primary": false, "embedding": null }},
            {{ "value": Y, "type": "context", "primary": false, "embedding": null }},
            {{ "value": null, "type": "artifactClass", "primary": true, "embedding": null }}
        ]

        Question: What is goal X and its requirements?
        →
        [
            {{ "value": "X", "type": "goal", "primary": true, "embedding": null }},
            {{ "value": null, "type": "requirement", "primary": true, "embedding": null }}
        ]

        Question: {question}
    """

    response, cost = await call_llm_structured(prompt, text_format="entitylist")
    

    return response, cost