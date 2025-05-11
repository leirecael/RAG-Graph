from llm.llm_client import call_llm_structured

async def validate_question(question: str):
    system_prompt = "You are a research domain classifier. Only return true if the question relates to technical or scientific issues."

    prompt = f"""
        # TASK
        Validate whether a question fits within a research or technical knowledge graph.
        
        # GRAPH SCHEMA
        (:problem)-[:arisesAt]->(:context)
        (:problem)-[:concerns]->(:stakeholder)
        (:problem)-[:informs]->(:goal)
        (:requirement)-[:meetBy]->(:artifactClass)
        (:problem)-[:addressedBy]->(:artifactClass)
        (:goal)-[:achievedBy]->(:requirement)

        # VALID
        - Research problems
        - Technical improvements
        - Structured scientific inquiries
        - Questions related to graph nodes.

        # INVALID
        - News, opinions, non-technical questions

        # EXAMPLES
        Q: How can feature models improve reuse? -> true
        Q: Who won the match? -> false
        Q: What problems are there? -> true

        #FORMAT
        A JSON objects with: value(the question, fixed if orthographically incorrct), is_valid(true or false), reasoning (why it is not valid) 

        # QUESTION
        {question}
    """
    
    response, cost = await call_llm_structured(prompt, system_prompt, text_format="question", task_name="question_validation")

    return response, cost


