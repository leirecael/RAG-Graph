from llm.llm_client import call_llm

async def create_cypher_query(queston, all_relevant_nodes):
    prompt = f"""
        Your task is to generate a valid Cypher query that connects the following nodes using only the predefined relationships.

        ### Allowed Relationships:
        - (problem)-[:arisesAt]->(context)
        - (problem)-[:concerns]->(stakeholder)
        - (problem)-[:informs]->(goal)
        - (requirement)-[:meetBy]->(artifactClass)
        - (problem)-[:addressedBy]->(artifactClass)
        - (goal)-[:achievedBy]->(requirement)

        ### Construction Rules:
        1. **Node labels**: Use type as label (e.g. `(p:problem)`) and a short alias.
        2. **PrimaryQuestion**: For entities marked `"PrimaryQuestion"`, use `name IS NOT NULL`, no filtering by specific value.
        3. **Filters**: Use `WHERE n.name = '...'` or `IN [...]` for non-primary entities.
        4. **Multiple instances**: For repeated entity types (e.g. two `problem` nodes), use different aliases (`p1`, `p2`) and ensure `p1 <> p2`.
        5. **MATCH**: Include only the necessary nodes and relationships based on what's present.
        6. **WITH DISTINCT**: Always use it to avoid duplicates.
        7. **RETURN**: Return `.name`, `.description`, and `labels(...)` for all matched nodes.
        8. **Single entity**: If only one node type is provided, just filter and return it, no relationships needed.

        ### Example:
        Question: What problems affect the same context?
        Cypher:
        MATCH (p1:problem), (c:context), (p2:problem)
        MATCH (p1)-[:arisesAt]->(c)<-[:arisesAt]-(p2)
        WHERE p1.name IS NOT NULL AND c.name IS NOT NULL AND p1 <> p2
        WITH DISTINCT p1, c, p2
        RETURN p1.name, p1.description, labels(p1),
            p2.name, p2.description, labels(p2),
            c.name, c.description, labels(c)

        Return **only the Cypher query**. Nothing else.

        Question: {queston}
        Available nodes: {all_relevant_nodes}
    """


    query,cost = await call_llm(prompt)
    return query, cost

def build_cypher_query(entities: dict) -> str:
    
    relationships = {
        ('problem', 'context'): 'arisesAt',
        ('problem', 'stakeholder'): 'concerns',
        ('problem', 'goal'): 'informs',
        ('requirement', 'artifactClass'): 'meetBy',
        ('problem', 'artifactClass'): 'addressedBy',
        ('goal', 'requirement'): 'achievedBy'
    }

    match_clauses = []
    where_clauses = []
    paths = []
    return_props = set()
    alias_set = set()

    for entity_type, value in entities.items():
        alias = entity_type[0]  
        alias_set.update(alias)
        match_clauses.append(f"({alias}:{entity_type})")
        
        if not value["result"] is None:
            
            if isinstance(value["result"], list):
                names = "', '".join(value["result"])
                where_clauses.append(f"{alias}.name IN ['{names}']")
            else:
                where_clauses.append(f"{alias}.name = '{value}'")
        else:
            where_clauses.append(f"{alias}.name IS NOT NULL")


        return_props.update([f"{alias}.name", f"{alias}.description", f"labels({alias})"])


    entity_types = list(entities.keys())
    if len(entity_types) > 1:
        for i in range(len(entity_types)):
            for j in range(i + 1, len(entity_types)):
                pair = (entity_types[i], entity_types[j])
                inverse_pair = (entity_types[j], entity_types[i])

                if pair in relationships:
                    rel = relationships[pair]
                    path = f"({entity_types[i][0]})-[:{rel}]->({entity_types[j][0]})"
                    paths.append(path)
                elif inverse_pair in relationships:
                    rel = relationships[inverse_pair]
                    path = f"({entity_types[j][0]})-[:{rel}]->({entity_types[i][0]})"
                    paths.append(path)


    match_string = "MATCH " + ", ".join(match_clauses)
    

    if paths:
        match_string += "\nMATCH " + ", ".join(paths)


    where_string = ""
    if where_clauses:
        where_string = "WHERE " + " AND ".join(where_clauses)

    with_string = "WITH DISTINCT " + ", ".join(alias_set)

    return_string = "RETURN " + ", ".join(return_props)


    query = f"""
    {match_string}
    {where_string}
    {with_string}
    {return_string}
    """.strip()

    return query