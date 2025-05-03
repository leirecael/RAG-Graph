def enrich_prompt2(question, related_nodes):
    context_text = ""

    for section_name, nodes in related_nodes.items():
        context_text += f"## {section_name.capitalize()}\n"
        for node_name, node_info in nodes.items():
            context_text += f"- **{node_name}**: {node_info['description']}\n"

    prompt = f"""You are an expert system. 
You must answer the user's question using only the following context information, without inventing anything. 
Answer concisely and clearly.

Question: {question}

Context:
{context_text}
"""
    return prompt

def enrich_prompt(question, context):
    node_blocks = []
    for category, nodes in context['entities'].items():
        if nodes:
            node_blocks.append(f"### {category.upper()}")
            for name, data in nodes.items():
                node_blocks.append(f"- **{name}**: {data['description']} [{', '.join(data['labels'])}]")

    rel_lines = []
    for rel in context['relationships']:
        rel_lines.append(f"- {rel['from']} --[{rel['type']}]--> {rel['to']}")

    prompt = f"""You must answer the following question based strictly on the following structured graph context. The answer must be clear and well structured.

### QUESTION
{question}

### ENTITIES
{chr(10).join(node_blocks)}

### RELATIONSHIPS
{chr(10).join(rel_lines)}
"""
    return prompt
