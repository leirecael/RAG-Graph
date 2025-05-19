from data.neo4j_client import execute_query, execute_multiple_queries
from logic.llm_tasks import validate_question, extract_entities, create_cypher_query, generate_final_answer, generate_entity_embeddings
from logs.logger import log_data, log_error
from logic. neo4j_logic import generate_similarity_queries, parse_similarity_results, parse_related_nodes_results
import json
import time
from datetime import datetime
from models.entity import EntityList, Entity
from models.question import Question

async def process_question(userQuestion: str) -> str:
    start = time.time()
    total_cost = 0


    try:
        validation, cost =  await validate_question(userQuestion)
        question = Question.model_validate(json.loads(validation))
        if not question.is_valid:
            log_error("InvalidQuestion", {
                "question": question.value,
                "reason": question.reasoning, 
            })
            return f"Your question is not valid. Reason: {question.reasoning}"
    
    except Exception as e:
        log_error("ValidationError", {
                "question": userQuestion,
                "error": str(e), 
            })
        raise
    total_cost +=cost
    
    try:
        entities, cost = await extract_entities(question.value)
        extracted_entities = EntityList.model_validate(json.loads(entities))
        
    except Exception as e:
        log_error("EntityExtractionError", {
                "question": question.value,
                "error": str(e), 
            })
        raise
    total_cost +=cost
    
    try:
        extracted_entities.entities, cost = await generate_entity_embeddings(extracted_entities.entities)
        entities_with_value = [e for e in extracted_entities.entities if e.value is not None]
        
        total_cost += cost
    except Exception as e:
            log_error("EmbeddingError", {
                "question": question.value,
                "error": str(e), 
            })
            raise
    try:
        
        if entities_with_value:
            start_sim = time.time()
            queries = generate_similarity_queries(entities_with_value)
            db_results = execute_multiple_queries(queries)
            similarity_results = parse_similarity_results(db_results)
            end_sim = time.time()
            log_data({
                "timestamp": datetime.now().isoformat(),
                "log_type": "database",
                "task_name": "similarity_calculation",
                "user_prompt": question.value,
                "final_response": json.dumps(similarity_results),
                "log_duration_sec": end_sim-start_sim,
            })
    except Exception as e:
            log_error("SimilarityError", {
                "question": question.value,
                "entities": [entity.value for entity in entities_with_value],
                "entity_types": [entity.type for entity in entities_with_value],
                "error": str(e), 
            })
            raise


    all_relevant_nodes = {} 
    entities_with_no_value = [e for e in extracted_entities.entities if e.value is None]
    for entity in entities_with_value:
        entity_type = entity.type
        if entity_type in similarity_results:
            all_relevant_nodes[entity_type] = similarity_results[entity_type]
        else:
            log_error("SimilarNotFoundError", {
                "question": question.value,
                "entity": entity.value,
                "entity_type": entity_type,
            })
            return f"No data found about {entity.value}."
    for entity in entities_with_no_value:
        all_relevant_nodes[entity.type] = None
    try:
        cypher_query, cost = await create_cypher_query(question.value, all_relevant_nodes)
    except Exception as e:
            log_error("CypherGenerationError", {
                "question": question.value,
                "nodes": all_relevant_nodes,
                "error": str(e), 
            })
            raise
    total_cost +=cost

    try:
        start_db = time.time()
        db_results = execute_query(cypher_query)
        related_nodes = parse_related_nodes_results(db_results)
        end_db = time.time()
        log_data({
            "timestamp": datetime.now().isoformat(),
            "log_type": "database",
            "task_name": "cypher_execution",
            "user_prompt": question.value,
            "final_response": cypher_query,
            "log_duration_sec": end_db-start_db,
        })
    except Exception as e:
        log_error("DatabaseQueryError", {
            "question": question.value,
            "query": cypher_query,
            "error": str(e), 
        })
        raise
    

    if len(related_nodes["entities"]) == 0 and len(related_nodes["relationships"]) == 0 and len(related_nodes["others"]) == 0:
        log_error("RelatedNodesNotFoundError", {
                    "question": question.value,
                    "query": cypher_query,
                    "nodes": all_relevant_nodes, 
                })
        return "No available information. Please, reword your question or try another one."

    try:
        final_answer, cost = await generate_final_answer(question.value, related_nodes)
    except Exception as e:
        log_error("ResponseGenerationError", {
            "question": question.value,
            "context": related_nodes,
            "error": str(e), 
        })
        raise
    total_cost +=cost

    end = time.time()
    elapsed_time = end-start
    log_data({
        "timestamp": datetime.now().isoformat(),
        "log_type": "register_query",
        "user_prompt": question.value,
        "final_response": final_answer,
        "log_duration_sec": elapsed_time,
        "cost": total_cost
    })

    return final_answer


#how to address the lack of flexibility in model variants to enchance model variants comparison that happens in EMF-based model variants?
#what problems do software developers have?
     