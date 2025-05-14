def enrich_prompt(question, context):
    node_blocks = []
    for category, nodes in context['entities'].items():
        if nodes:
            node_blocks.append(f"### {category.upper()}")
            for name, data in nodes.items():
                alt_name = f"{data['alternativeName']}" if 'alternativeName' in data else ""
                node_blocks.append(f"-**{name}({alt_name};{data['hypernym']})**: {data['description']} [{', '.join(data['labels'])}]")

    rel_lines = []
    for rel in context['relationships']:
        rel_lines.append(f"- {rel['from']} --[{rel['type']}]--> {rel['to']}")

    oth_lines = []
    for key, value in context["others"].items():
        oth_lines.append(f"-{key}: {value}")
    system_prompt = """You are an expert assistant that answers questions based strictly on structured graph data. Use only the information provided. Do not make assumptions or fabricate details. 
                        If the graph does not provide enough information, say so clearly. Provide answers that are concise, technically accurate, and well-organized."""
    prompt = f"""Use the following information to answer the question. 

    

        ### QUESTION
        {question}

        ### ENTITIES
        {"\n".join(node_blocks)}

        ### RELATIONSHIPS
        {"\n".join(rel_lines)}

        ### OTHERS
        {"\n".join(oth_lines)}
    """
    return prompt, system_prompt
