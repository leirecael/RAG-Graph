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

    prompt = f"""You must answer the following question based strictly on the following structured graph context. The answer must be clear,well structured and concise. 

### QUESTION
{question}

### ENTITIES
{"\n".join(node_blocks)}

### RELATIONSHIPS
{"\n".join(rel_lines)}
"""
    return prompt
