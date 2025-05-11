from logic.validator import validate_question
from data.database_connector import execute_query, generate_similarity_queries, execute_multiple_queries_with_apoc
from llm.llm_client import get_embedding, call_llm
from logic.question_processor import extract_entities
from logic.cypher_generator import create_cypher_query
from logic.prompt_enricher import enrich_prompt
from data.logger import log_data, log_error
import json
import time
from datetime import datetime
from models.entity import EntityList, Entity
from models.question import Question
import asyncio

async def process_question(userQuestion):
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
        entities_with_value = [e for e in extracted_entities.entities if not e.value is None]
        
        embedding_results = await asyncio.gather(*(get_embedding(e.value, task_name="embed_entity") for e in entities_with_value))

        for entity, (embedding, cost) in zip(entities_with_value, embedding_results):
            entity.embedding = embedding
            total_cost +=cost
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
            similarity_results = execute_multiple_queries_with_apoc(queries)
            end_sim = time.time()
            log_data({
                "timestamp": datetime.now().isoformat(),
                "log_type": "database",
                "task_name": "similarity_calculation",
                "user_prompt": question.value,
                "final_response": similarity_results,
                "log_duration_sec": end_sim-start_sim,
            })
    except Exception as e:
            log_error("SimilarityError", {
                "question": question.value,
                "entity": entity.value,
                "entity_type": entity.type,
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
            log_error("SimilarityError", {
                "question": question.value,
                "entity": entity.value,
                "entity_type": entity_type,
            })
            return f"No data found about {entity}."
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
        related_nodes = execute_query(cypher_query)
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
    

    if len(related_nodes["entities"]) == 0:
        log_error("RelatedNodesNotFoundError", {
                    "question": question.value,
                    "query": cypher_query,
                    "nodes": all_relevant_nodes, 
                })
        return "No available information. Please, reword your question or try another one."
    try:
        final_prompt, system_prompt = enrich_prompt(question.value,related_nodes)
        
    except Exception as e:
        log_error("EnrichmentError", {
            "question": question.value,
            "nodes": related_nodes,
            "error": str(e), 
        })
        raise


    try:
        final_answer, cost = await call_llm(final_prompt, system_prompt,task_name="rag_answer_generation")
    except Exception as e:
        log_error("ResponseGenerationError", {
            "question": question.value,
            "prompt": final_prompt,
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