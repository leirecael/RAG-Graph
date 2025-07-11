from models.entity import Entity

class Neo4jLogic:
    """
    Contains logic for constructing Cypher queries and parsing results for graph queries.

    This class supports generating similarity search queries with or without label constraints,
    parsing results from Neo4j, and removing duplicate textual data from query outputs.

    Methods:
        generate_similarity_queries(): Generate label-based similarity queries.
        generate_similarity_queries_no_label(): Generate similarity queries ignoring node labels.
        parse_similarity_results(): Parse similarity search results grouped by entity types.
        parse_related_nodes_results(): Parse related node records into structured data.
        remove_duplicate_text(): Remove duplicate semicolon-separated segments in a string.
        remove_duplicate_text_in_list(): Clean and deduplicate a list of strings.
    """

    def __init__(self):
        pass

    def generate_similarity_queries(self, entities_with_value: list[Entity], threshold: float = 0.7, top_k: int = 3)->list[dict]:
        """
        Generate Cypher queries to find nodes similar to given entities based on cosine similarity of embeddings.

        Args:
            entities_with_value (list[Entity]): List of Entity objects with calculated embeddings.
            threshold (float): Minimum cosine similarity threshold to consider as valid.
            top_k (int): Maximum number of similar nodes to return per entity.

        Returns:
            list[dict]: List of dicts with 'query' and 'params' keys for batch execution.
        """
        ALLOWED_LABELS = {"problem", "goal", "requirement", "context", "stakeholder", "artifactClass"}

        queries_with_params = []

        for entity in entities_with_value:
            #Make sure the label is correct
            if entity.type not in ALLOWED_LABELS:
                raise ValueError(f"Invalid label: {entity.type}")
            
            query = f"""
            WITH $embedding AS embedding
            MATCH (n:{entity.type})
            WHERE n.embedding IS NOT NULL
            WITH n, gds.similarity.cosine(embedding, n.embedding) AS similarity
            WHERE similarity >= $threshold
            RETURN DISTINCT n.name as name, similarity, labels(n) as labels
            ORDER BY similarity DESC
            LIMIT $top_k
            """
            params = {
                "embedding": entity.embedding,
                "threshold": threshold,           
                "top_k": top_k

            }
            queries_with_params.append({
                "query": query,
                "params": params
            })
        return queries_with_params

    def generate_similarity_queries_no_label(self, entities_with_value: list[Entity], threshold: float = 0.6, top_k: int = 3) -> list[dict]:
        """
        Generate Cypher queries to find nodes similar to entity values, no matter the label, based on cosine similarity of embeddings.

        Args:
            entities_with_value (list[Entity]): List of Entity objects with calculated embeddings.
            threshold (float): Minimum cosine similarity threshold to consider as valid.
            top_k (int): Maximum number of similar nodes to return per entity.

        Returns:
            list[dict]: List of dicts with 'query' and 'params' keys for batch execution.
        """
        queries_with_params = []
        for entity in entities_with_value:
            query = f"""
                WITH $embedding AS embedding
                MATCH (n)
                WHERE n.embedding IS NOT NULL
                WITH n, gds.similarity.cosine(embedding, n.embedding) AS similarity
                WHERE similarity >= $threshold
                RETURN DISTINCT n.name as name, similarity, labels(n) as labels
                ORDER BY similarity DESC
                LIMIT $top_k
                """
            params = {
                    "embedding": entity.embedding,
                    "threshold": threshold,           
                    "top_k": top_k

                }
            queries_with_params.append({
                    "query": query,
                    "params": params
            })

        return queries_with_params

    def parse_similarity_results(self, results: list[dict]) -> dict:
        """
        Parse the results from a batch of similarity queries, grouping similar node names by their entity type (label).

        Args:
            results (list[dict]): List of result dicts returned from Neo4j APOC cypher calls.

        Returns:
            dict: Mapping from entity type to list of similar node names. (e.g. {"problem": ["world hunger"], "goal": ["enhance food production", "improve agriculture technology"]})
        """
        parsed = {}
        for item in results:
            data = item["value"]  #Contains 'name', 'similarity', 'labels'
            labels = data.get("labels", [])
            
            primary_label = labels[0]  #Use the first label
            if primary_label not in parsed:
                parsed[primary_label] = []

            parsed[primary_label].append(data["name"])

        return parsed

    def parse_related_nodes_results(self, records: list[dict]) -> dict:
        """
        Parse related node records from Neo4j query results into structured entities, relationships, and other info. Removes duplicates.

        Args:
            records (list[dict]): Records from a Neo4j query.

        Returns:
            dict: Dictionary containing:
                - "entities": categorized entities with their details,
                - "relationships": list of unique relationships between entities,
                - "others": any other information that is not an entity or relationship.
        """
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
            for key, value in record.items(): #Example: c.hypernym, software development context

                #Process other information, not x.name or labels(x) type of information (e.g. problemsCount)
                if '.' not in key and not key.startswith('labels'):
                    if isinstance(value, list):
                        others[key] = self.remove_duplicate_text_in_list(value)
                    elif isinstance(value, str):
                        others[key] = self.remove_duplicate_text(value)
                    else:
                        others[key] = value
                    continue
                
                #Process nodes and their attributes from each node name
                if key.endswith('.name'):
                    alias = key.split('.')[0] #Extract node alias
                    name = value
                    desc = record.get(f"{alias}.description", "")
                    labels = record.get(f"labels({alias})", [])
                    hyper = record.get(f"{alias}.hypernym", "")
                    alt_name = record.get(f"{alias}.alternativeName", "")

                    for label in labels:
                        category = CATEGORY_MAP.get(label)
                        if category:
                            #Add the node information to it's entity type dictionary. Can be added only once
                            if name not in entities[category]:
                                entities[category][name] = {
                                    'description': self.remove_duplicate_text(desc),
                                    'labels': labels,
                                    'hypernym': self.remove_duplicate_text(hyper)
                                }
                                #AlternativeName is a property that not all nodes have
                                if alt_name:
                                    entities[category][name]['alternativeName'] = self.remove_duplicate_text(alt_name)

            # Relationships: (from, to) : type
            valid_rels = {
                ('problem', 'context'): 'arisesAt',
                ('problem', 'stakeholder'): 'concerns',
                ('problem', 'goal'): 'informs',
                ('requirement', 'artifactClass'): 'meetBy',
                ('problem', 'artifactClass'): 'addressedBy',
                ('goal', 'requirement'): 'achievedBy'
            }

            #Map aliases to their labels for relationship identification
            alias_map = {}
            for key in record: #Example: p.name 
                if key.endswith('.name'):
                    alias = key.split('.')[0]
                    labels = record.get(f"labels({alias})", [])
                    for label in labels:
                        if label in CATEGORY_MAP:
                            alias_map[alias] = label #Example: {'p': 'problem', 'c1': 'context', 'c2': 'context'}

            #Generate unique relationships between related entities
            for (src_type, tgt_type), rel_type in valid_rels.items(): #Get a relationship(source, target)
                for src_alias, src_label in alias_map.items(): #Get the source alias and label
                    if src_type == src_label: #Match source labels
                        for tgt_alias, tgt_label in alias_map.items(): #Get the target alias and label
                            if tgt_type == tgt_label: #Match target labels
                                src_name = record.get(f"{src_alias}.name")
                                tgt_name = record.get(f"{tgt_alias}.name")
                                if src_name and tgt_name:
                                    #Create the relationship
                                    rel_key = (src_name, tgt_name, rel_type)
                                    #Save the relationship if it is not duplicate
                                    if rel_key not in relationships_set:
                                        relationships_set.add(rel_key) #No duplicates
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


    def remove_duplicate_text(self, text: str) -> str:
        """
        Remove duplicate semicolon-separated segments from a string.

        Args:
            text (str): Input string with potentially redundant semicolon-separated parts.

        Returns:
            str: Cleaned string with duplicates removed.
        """
        seen = set()
        result = []

        for part in text.split(';'):
            cleaned = part.strip()
            key = cleaned.lower()
            if key and key not in seen:
                seen.add(key)
                result.append(cleaned)

        return '; '.join(result)


    def remove_duplicate_text_in_list(self, items: list) -> list:
        """
        Remove duplicates from a list of strings, cleaning each with remove_redundant_text.

        Args:
            items (list): List of strings.

        Returns:
            list: List of unique, cleaned strings.
        """
        seen = set()
        result = []

        for item in items:
            cleaned_text = self.remove_duplicate_text(str(item)).strip()
            key = cleaned_text.lower()
            if key and key not in seen:
                seen.add(key)
                result.append(cleaned_text)

        return result

