from llm.llm_client import call_llm_structured

async def extract_entities(question: str) -> dict:

    system_prompt = "You are an expert entity extractor for scientific knowledge graphs. Extract only entities that are clearly present or directly implied. Do not invent entities."

    prompt = f"""
        # TASK
        Extract relevant entities from the question. You must understand the user's intention. Understand which entities will give the user the answer they want. 

        # ENTITY TYPES
        - problem: a deficiency or issue (e.g. lack of traceability)
        - stakeholder: person or group affected (e.g. developers)
        - goal: desired high-level outcome (e.g. improve maintainability)
        - context: domain or situation (e.g. safety-critical systems)
        - requirement: system-level functionality or need
        - artifactClass: type of technical solution to a problem (e.g. feature model)
        
        # RELATIONSHIPS
        - (problem)-[:arisesAt]->(context)
        - (problem)-[:concerns]->(stakeholder)
        - (problem)-[:informs]->(goal)
        - (requirement)-[:meetBy]->(artifactClass)
        - (problem)-[:addressedBy]->(artifactClass)
        - (goal)-[:achievedBy]->(requirement)

        # FORMAT
        List of JSON objects with: value, type, embedding (always null). value and type can never be the same (e.g. {{"value": "stakeholders", "type": "stakeholder", "embedding": null}})

        # POSITIVE EXAMPLES
        Question: What problems do developers face?
        [{{"value": "developers", "type": "stakeholder", "embedding": null}}, {{"value": null, "type": "problem", "embedding": null}}]
        Question: How can we fix the problem of climate change?
        [{{"value": "climate change", "type": "problem", "embedding": null}}, {{"value": null, "type": "artifactClass", "embedding": null}}]
        Question: What problems are solved by the same artifact?
        [{{"value": null, "type": "problem", "embedding": null}}, {{"value": null, "type": "artifactClass", "embedding": null}}]
        Question: How many stakeholders are affected by the lack of software evolution history?
        [{{"value": "lack of software evolution history", "type": "problem", "embedding": null}}, {{"value": null, "type": "stakeholder", "embedding": null}}]

        # NEGATIVE EXAMPLE
        Question: What's the weather today?
        []

        # QUESTION
        {question}
    """

    response, cost = await call_llm_structured(prompt, system_prompt, text_format="entitylist", task_name="entity_extraction") 
    return response, cost