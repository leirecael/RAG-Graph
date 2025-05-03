from llm.llm_client import call_llm

async def create_cypher_query(queston, all_relevant_nodes):
    prompt = f"""
        Tu tarea es generar una consulta Cypher válida que conecte los siguientes nodos entre sí a través de relaciones predefinidas.

        ### Relaciones válidas:
        - (problem)-[:arisesAt]->(context)
        - (problem)-[:concerns]->(stakeholder)
        - (problem)-[:informs]->(goal)
        - (requirement)-[:meetBy]->(artifactClass)
        - (problem)-[:addressedBy]->(artifactClass)
        - (goal)-[:achievedBy]->(requirement)

        ### Reglas de construcción:
        1. **Nombres de nodos**: Usa el nombre del tipo como etiqueta (`(p:problem)`, `(c:context)`) y una letra como alias.
        2. **PrimaryQuestion**: Si una entidad tiene valor `"PrimaryQuestion"`, es el nodo principal. No filtres por `name = ...`, solo asegúrate que `name IS NOT NULL`.
        3. **Filtros**: Para entidades no "PrimaryQuestion", usa `WHERE n.name = '...'` o `IN ['a', 'b']` según el caso.
        4. **Relaciones múltiples**: Si se necesitan múltiples instancias de una entidad (por ejemplo, dos `problem`), usa alias diferentes (`p1`, `p2`) y asegúrate de usar `p1 <> p2` para que no sean el mismo nodo.
        5. **MATCH**: Incluye todos los nodos mencionados y las relaciones relevantes entre ellos, según las relaciones permitidas.
        6. **WITH DISTINCT**: Siempre úsalo para evitar duplicados.
        7. **RETURN**: Devuelve los nodos involucrados con `n.name`, `n.description`, y `labels(n)`.
        8. **No relaciones innecesarias**: Si solo hay una entidad, no construyas relaciones. Solo filtra y devuelve.

        ### Ejemplo:
        Pregunta: ¿Qué problemas afectan al mismo contexto?
        Consulta esperada:
        MATCH (p1:problem), (c:context), (p2:problem)
        MATCH (p1)-[:arisesAt]->(c)<-[:arisesAt]-(p2)
        WHERE p1.name IS NOT NULL AND c.name IS NOT NULL AND p1 <> p2
        WITH DISTINCT p1, c, p2
        RETURN p1.name, p1.description, labels(p1),
            p2.name, p2.description, labels(p2),
            c.name, c.description, labels(c)

        Devuelve **solo la consulta Cypher**. Nada más.

        Pregunta: {queston}
        Nodos disponibles: {all_relevant_nodes}
        
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
        
        if value != "PrimaryQuestion":
            
            if isinstance(value, list):
                names = "', '".join(value)
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