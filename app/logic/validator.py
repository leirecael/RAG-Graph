from llm.llm_client import call_llm_structured

async def validate_question(question: str):
    prompt = f"""You are an assistant that checks if a question is appropriate for a database of scientific and technological papers.
        Return true only if the question relates to research, technical topics, or similar content. Here is some information about the database schema:
        - (problem)-[:arisesAt]->(context)
        - (problem)-[:concerns]->(stakeholder)
        - (problem)-[:informs]->(goal)
        - (requirement)-[:meetBy]->(artifactClass)
        - (problem)-[:addressedBy]->(artifactClass)
        - (goal)-[:achievedBy]->(requirement)
        
        Leave the value of is_simple empty.

        Question: {question}
        """
    
    response, cost = await call_llm_structured(prompt,text_format="question")

    return response, cost


