from logic.validator import validate_question
from data.database_connector import get_similar_nodes_by_entity, execute_query
from llm.llm_client import get_embedding, call_llm
from logic.question_processor import extract_entities, determine_simplicity
from logic.cypher_generator import build_cypher_query, create_cypher_query
from logic.prompt_enricher import enrich_prompt
from data.logger import log_query, log_error
from data.cache_manager import get_embedding_cached, get_similarity_cached
import json
import time
from models.entity import EntityList, Entity
from models.question import Question
import asyncio

async def process_question(userQuestion):
    start = time.time()
    cost_gpt = cost_embed = 0
    try:
        try:
            validation, cost =  await validate_question(userQuestion)
            question = Question.model_validate(json.loads(validation))
            if not question.is_valid:
                log_error("InvalidQuestion", {
                    "question": question.value,
                    "reason": question.reasoning, 
                    "total_cost": cost_gpt + cost_embed
                })
                return f"Your question is not valid. Reason: {question.reasoning}"
        
        except Exception as e:
            log_error("ValidationError", {
                    "question": userQuestion,
                    "error": str(e), 
                    "total_cost": cost_gpt + cost_embed
                })
            return f"ValidationError: {str(e)}"
        print("VALIDATION COMPLETE")
        cost_gpt += cost
        
        try:
            entities, cost = await extract_entities(question.value)
            extracted_entities = EntityList.model_validate(json.loads(entities))
            
        except Exception as e:
            log_error("EntityExtractionError", {
                    "question": question.value,
                    "error": str(e), 
                    "total_cost": cost_gpt + cost_embed
                })
            return f"EntityExtractionError: {str(e)}"
        print("EXTRACTION COMPLETE")
        cost_gpt += cost
        
        try:
            entities_with_value = [e for e in extracted_entities.entities if not e.value is None]
            
            embedding_results = await asyncio.gather(*(get_embedding(e.value) for e in entities_with_value))

            for entity, (embedding, cost) in zip(entities_with_value, embedding_results):
                entity.embedding = embedding
                cost_embed += cost
        except Exception as e:
                log_error("EmbeddingError", {
                    "error": str(e), 
                    "total_cost": cost_gpt + cost_embed
                })
                return f"EmbeddingError: {str(e)}"
        print("EMBEDDINGS COMPLETE")
        try:
            similarity_tasks = [
                get_similar_nodes_by_entity(entity.type, entity.embedding)
                for entity in entities_with_value
            ]
            similarity_results = await asyncio.gather(*similarity_tasks)
        except Exception as e:
                log_error("SimilarityError", {
                    "entity": entity.value,
                    "entity_type": entity.type,
                    "error": str(e), 
                    "total_cost": cost_gpt + cost_embed
                })
                return f"SimilarityError: {str(e)}"

        print("SIMILARITY COMPLETE")

        all_relevant_nodes = {} 
        entities_with_no_value = [e for e in extracted_entities.entities if e.value is None]
        for entity, result in zip(entities_with_value, similarity_results):
            all_relevant_nodes[entity.type] = {"result":result, "primary":entity.primary}
        for entity in entities_with_no_value:
            all_relevant_nodes[entity.type] = {"result": None, "primary":entity.primary}
        print(all_relevant_nodes)
        print("STRUCTURING COMPLETE")
        try:
            question.is_simple, cost = await determine_simplicity(question.value)
        except Exception as e:
            log_error("SimplicityError", {
                    "question": question.value,
                    "error": str(e), 
                    "total_cost": cost_gpt + cost_embed
                })
            return f"SimplicityError: {str(e)}"
        print("SIMPLICITY COMPLETE")
        cost_gpt += cost

        try:
            if question.is_simple:
                cypher_query = build_cypher_query(all_relevant_nodes)
                print(cypher_query)
            else:
                cypher_query, cost = await create_cypher_query(question.value, all_relevant_nodes)
        except Exception as e:
                log_error("CypherGenerationError", {
                    "nodes": all_relevant_nodes,
                    "error": str(e), 
                    "total_cost": cost_gpt + cost_embed
                })
                return f"CypherGenerationError: {str(e)}"
        cost_gpt += cost
        print("CYPHER COMPLETE")
        try:
            related_nodes = await execute_query(cypher_query)
        except Exception as e:
            log_error("DatabaseQueryError", {
                "query": cypher_query,
                "error": str(e), 
                "total_cost": cost_gpt + cost_embed
            })
            return f"DatabaseQueryError: {str(e)}"
        
        print("RELATED COMPLETE")
        if not related_nodes:
            log_error("RelatedNodesNotFoundError", {
                        "query": cypher_query,
                        "nodes": all_relevant_nodes, 
                        "total_cost": cost_gpt + cost_embed
                    })
            return "No available information. Please, reword your question or try another one."
        try:
            final_prompt = enrich_prompt(question.value,related_nodes)
            
        except Exception as e:
            log_error("EnrichmentError", {
                "question": question.value,
                "nodes": related_nodes,
                "error": str(e), 
                "total_cost": cost_gpt + cost_embed
            })
            return f"EnrichmentError: {str(e)}"
        print("ENRICHING COMPLETE")

        try:
            final_answer, cost = await call_llm(final_prompt)
        except Exception as e:
            log_error("ResponseGenerationError", {
                "prompt": final_prompt,
                "error": str(e), 
                "total_cost": cost_gpt + cost_embed
            })
            return f"ResponseGenerationError: {str(e)}"
        cost_gpt+= cost
        print("RESPONSE COMPLETE")
        end = time.time()
        elapsed_time = end-start

        log_query({"question": question.value,
            "entities": entities,
            "simple_question": question.is_simple,
            "cypher_query": cypher_query,
            "prompt": final_prompt,
            "llm_response": final_answer,
            "time_elapsed": elapsed_time,
            "cost_gpt": cost_gpt,
            "cost_embed": cost_embed,
            "total_cost": cost_gpt + cost_embed
        })


        return final_answer
    except Exception as e:
        log_error("UnhandledException", {
            "question": userQuestion,
            "error": str(e), 
            "total_cost": cost_gpt + cost_embed
        })
        return "An unexpected error occurred while processing your question."

#how to address the lack of flexibility to enchance model variants comparison that happens in EMF-based model variants?
#what problems do software developers have?