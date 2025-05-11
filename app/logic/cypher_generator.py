from llm.llm_client import call_llm

async def create_cypher_query(question, all_relevant_nodes):
    system_prompt = "You are a Cypher query generator for a scientific knowledge graph. Only return the query."

    prompt = f"""
        # TASK
        You are given a question and a set of relevant node types with optional filters.
        Generate a syntactically and semantically correct Cypher query using the schema, following the rules and examples below.

        # GRAPH SCHEMA
        (:problem)-[:arisesAt]->(:context)
        (:problem)-[:concerns]->(:stakeholder)
        (:problem)-[:informs]->(:goal)
        (:requirement)-[:meetBy]->(:artifactClass)
        (:problem)-[:addressedBy]->(:artifactClass)
        (:goal)-[:achievedBy]->(:requirement)

        # RULES
        1. Always use entity types as labels, e.g. (p:problem).
        2. For each entry in AVAILABLE NODES:
        - If the value is a list, use `name IN [...]`
        - If the value is None, filter with `name IS NOT NULL`
        - If multiple nodes of the same type are needed, use aliases like p1, p2.
        3. Use `WITH DISTINCT` to eliminate duplicates before RETURN.
        4. Use `LIMIT` only when relevant.
        5. Always return: `name`, `description`, `hypernym`, and `labels(...)` for all nodes involved.
        6. Do not rename output fields. Maintain standard Cypher return format.
        7. Only generate the Cypher query. Do not add comments or explanations.
        8. You can traverse the graph to look for related ideas. Use all schema relationships that apply.(e.g. artifacts related by problem and requirement, goals related by requirement and problem)

        # EXAMPLES
        Q: What problems are solved by the same artifact?
        AVAILABLE NODES: {{'problem': None, 'artifactClass': None}}

        Cypher:
        MATCH (p1:problem)-[:addressedBy]->(a:artifactClass)<-[:addressedBy]-(p2:problem)
        WHERE p1.name IS NOT NULL AND a.name IS NOT NULL AND p2.name IS NOT NULL AND p1 <> p2
        WITH DISTINCT p1, a, p2
        RETURN p1.name, p1.description, p1.hypernym, labels(p1),
            p2.name, p2.description, p2.hypernym, labels(p2),
            a.name, a.description, a.hypernym, labels(a)

        Q: What problems are related?
        AVAILABLE NODES: {{'problem': None}}
        Cypher:
        MATCH (p1:problem)-[:arisesAt|concerns|informs]->(x)<-[:arisesAt|concerns|informs]-(p2:problem)
        WHERE p1 <> p2
        WITH DISTINCT p1, p2, x
        RETURN p1.name, p1.description, p1.hypernym, labels(p1),
            p2.name, p2.description, p2.hypernym, labels(p2),
            x.name, x.description, x.hypernym, labels(x)

        # QUESTION
        {question}

        # AVAILABLE NODES
        {all_relevant_nodes}
    """

    query, cost = await call_llm(prompt, system_prompt, model= "gpt-4.1",task_name="cypher_generation")
    return query, cost