from models.entity import Entity


def generate_similarity_queries(entities_with_value: list[Entity], threshold: float = 0.6, top_k: int = 3)->list[dict]:
    queries_with_params = []
    for entity in entities_with_value:
        query = f"""
        WITH $embedding AS embedding
        MATCH (n:{entity.type})
        WHERE n.embedding IS NOT NULL
        WITH n, gds.similarity.cosine(embedding, n.embedding) AS similarity
        WHERE similarity >= {threshold}
        RETURN DISTINCT n.name as name, similarity, labels(n) as labels
        ORDER BY similarity DESC
        LIMIT {top_k}
        """
        params = {
            "embedding": entity.embedding
        }
        queries_with_params.append({
            "query": query,
            "params": params
        })
    return queries_with_params

def parse_similarity_results(results: list[dict]) -> dict[str, list[str]]:
    parsed = {}

    for item in results:
        data = item["value"]  
        labels = data.get("labels", [])
        if not labels:
            continue  
        
        primary_label = labels[0]  
        if primary_label not in parsed:
            parsed[primary_label] = []

        parsed[primary_label].append(data["name"])

    return parsed

def parse_related_nodes_results(records: list[dict]) -> dict:
    CATEGORY_MAP = {
        'problem': 'problems',
        'stakeholder': 'stakeholders',
        'goal': 'goals',
        'context': 'contexts',
        'artifactClass': 'artifactClasses',
        'requirement': 'requirements'
    }

    entities = {v: {} for v in CATEGORY_MAP.values()}
    relationships_set = set()
    relationships = []
    others = {}

    for record in records:
        for key, value in record.items():
            if '.' not in key and not key.startswith('labels'):
                if isinstance(value, list):
                    others[key] = remove_redundant_text_in_list(value)
                elif isinstance(value, str):
                    others[key] = remove_redundant_text(value)
                else:
                    others[key] = value
                continue

            if key.endswith('.name'):
                alias = key.split('.')[0]
                name = value
                desc = record.get(f"{alias}.description", "")
                labels = record.get(f"labels({alias})", [])
                hyper = record.get(f"{alias}.hypernym", "")
                alt_name = record.get(f"{alias}.alternativeName", "")

                for label in labels:
                    category = CATEGORY_MAP.get(label)
                    if category:
                        if name not in entities[category]:
                            entities[category][name] = {
                                'description': remove_redundant_text(desc),
                                'labels': labels,
                                'hypernym': remove_redundant_text(hyper)
                            }
                            if alt_name:
                                entities[category][name]['alternativeName'] = remove_redundant_text(alt_name)

        # Relationships
        known_rels = {
            ('problem', 'context'): 'arisesAt',
            ('problem', 'stakeholder'): 'concerns',
            ('problem', 'goal'): 'informs',
            ('requirement', 'artifactClass'): 'meetBy',
            ('problem', 'artifactClass'): 'addressedBy',
            ('goal', 'requirement'): 'achievedBy'
        }

        alias_map = {}
        for key in record:
            if key.endswith('.name'):
                alias = key.split('.')[0]
                labels = record.get(f"labels({alias})", [])
                for label in labels:
                    if label in CATEGORY_MAP:
                        alias_map[alias] = label

        for (src_type, tgt_type), rel_type in known_rels.items():
            for src_alias, src_label in alias_map.items():
                if src_label == src_type:
                    for tgt_alias, tgt_label in alias_map.items():
                        if tgt_label == tgt_type and src_alias != tgt_alias:
                            src_name = record.get(f"{src_alias}.name")
                            tgt_name = record.get(f"{tgt_alias}.name")
                            if src_name and tgt_name:
                                rel_key = (src_name, tgt_name, rel_type)
                                if rel_key not in relationships_set:
                                    relationships_set.add(rel_key)
                                    relationships.append({
                                        "from": src_name,
                                        "to": tgt_name,
                                        "type": rel_type,
                                    })

    return {
        "entities": entities,
        "relationships": relationships,
        "others": others
    }


def remove_redundant_text(text: str) -> str:
    seen = set()
    result = []

    for part in text.split(';'):
        cleaned = part.strip()
        key = cleaned.lower()
        if key and key not in seen:
            seen.add(key)
            result.append(cleaned)

    return '; '.join(result)


def remove_redundant_text_in_list(items: list) -> list:
    seen = set()
    result = []

    for item in items:
        cleaned_text = remove_redundant_text(str(item)).strip()
        key = cleaned_text.lower()
        if key and key not in seen:
            seen.add(key)
            result.append(cleaned_text)

    return result