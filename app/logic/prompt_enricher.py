MODEL_INFO = {
    "gpt-4.1-mini": {"input_price": 0.0004, "output_price": 0.0016, "max_context": 1047576, "max_output":32768, "encoding":"o200k_base"},
    "gpt-4.1": {"input_price": 0.002, "output_price": 0.008, "max_context": 1047576, "max_output":32768, "encoding":"o200k_base"},
    "gpt-3.5-turbo": {"input_price": 0.0005, "output_price": 0.0015, "max_context": 16385, "max_output":4096, "encoding":"cl100k_base"},
    "text-embedding-3-small": 0.00002
}

def enrich_prompt(question, context):
    node_blocks = []
    for category, nodes in context['entities'].items():
        if nodes:
            node_blocks.append(f"### {category.upper()}")
            for name, data in nodes.items():
                node_blocks.append(f"- **{name}({data['hypernym']})**: {data['description']} [{', '.join(data['labels'])}]")

    rel_lines = []
    for rel in context['relationships']:
        rel_lines.append(f"- {rel['from']} --[{rel['type']}]--> {rel['to']}")
    system_prompt = """You are an expert assistant that answers questions based strictly on structured graph data. Use only the information provided. Do not make assumptions or fabricate details. 
                        If the graph does not provide enough information, say so clearly. Provide answers that are concise, technically accurate, and well-organized."""
    prompt = f"""Use the following information to answer the question. 

        ### QUESTION
        {question}

        ### ENTITIES
        {"\n".join(node_blocks)}

        ### RELATIONSHIPS
        {"\n".join(rel_lines)}
    """
    return prompt, system_prompt
