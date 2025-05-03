from logic.validator import validate_question
from data.database_connector import get_similar_nodes_by_entity, execute_query, prob
from llm.llm_client import get_embedding, call_llm
from logic.question_processor import extract_entities, determine_simplicity
from logic.cypher_generator import build_cypher_query, create_cypher_query
from logic.prompt_enricher import enrich_prompt
from data.logger import log_query, log_error
from data.cache_manager import get_cached_embedding, cache_embedding
import time

async def process_question(question):
    start = time.time()
    cost_embed = 0
    cost_gpt = 0
    cost_agent = 0
    try:
        validation, cost_agent =  await validate_question(question)
        if not validation.is_valid_question:
            log_error("InvalidQuestion", {
                "question": question,
                "reason": validation.reasoning, 
                "cost_gpt": cost_gpt,
                "cost_embed": cost_embed,
                "cost_agent": cost_agent,
                "total_cost": cost_gpt + cost_embed + cost_agent
            })
            return f"Your question is not valid. Reason: {validation.reasoning}"
            
        
        try:
            extracted_entities, cost = await extract_entities(question)
        except Exception as e:
            log_error("EntityExtractionError", {
                    "question": question,
                    "error": str(e), 
                    "cost_gpt": cost_gpt,
                    "cost_embed": cost_embed,
                    "cost_agent": cost_agent,
                    "total_cost": cost_gpt + cost_embed + cost_agent
                })
            return f"EntityExtractionError: {str(e)}"
        
        cost_gpt += cost
        
        print(extracted_entities)

        all_relevant_nodes = {} 
        for entity_type,value in extracted_entities.items():

            if value == "PrimaryQuestion":
                all_relevant_nodes[entity_type] = value
                continue
            try:

                entity_embedding, cost = await get_embedding(value)
                cost_embed += cost
                most_similar_nodes = get_similar_nodes_by_entity(entity_type, entity_embedding)
                if most_similar_nodes.count == 0:
                    log_error("SimilarNodesNotFoundError", {
                        "entity": value,
                        "entity_type": entity_type, 
                        "cost_gpt": cost_gpt,
                        "cost_embed": cost_embed,
                        "cost_agent": cost_agent,
                        "total_cost": cost_gpt + cost_embed + cost_agent
                    })
                    return "No available information. Please, change reword your question or try another one."

                #unique_nodes = {node['node']: node for node in most_similar_nodes}.values()
                all_relevant_nodes[entity_type] = most_similar_nodes #list(unique_nodes)
            except Exception as e:
                log_error("EmbeddingOrSimilarityError", {
                    "entity": value,
                    "entity_type": entity_type,
                    "error": str(e), 
                    "cost_gpt": cost_gpt,
                    "cost_embed": cost_embed,
                    "cost_agent": cost_agent,
                    "total_cost": cost_gpt + cost_embed + cost_agent
                })
                return f"EmbeddingOrSimilarityError: {str(e)}"
        print(all_relevant_nodes)

        try:
            simplicity, cost = await determine_simplicity(question)
        except Exception as e:
            log_error("SimplicityError", {
                    "question": question,
                    "error": str(e), 
                    "cost_gpt": cost_gpt,
                    "cost_embed": cost_embed,
                    "cost_agent": cost_agent,
                    "total_cost": cost_gpt + cost_embed + cost_agent
                })
            return f"SimplicityError: {str(e)}"
        
        cost_gpt += cost

        try:
            if simplicity["is_simple"]:
                print("simple")
                cypher_query = build_cypher_query(all_relevant_nodes)
            else:
                print("complex")
                cypher_query, cost = await create_cypher_query(question, all_relevant_nodes)
        except Exception as e:
                log_error("CypherGenerationError", {
                    "nodes": all_relevant_nodes,
                    "error": str(e), 
                    "cost_gpt": cost_gpt,
                    "cost_embed": cost_embed,
                    "cost_agent": cost_agent,
                    "total_cost": cost_gpt + cost_embed + cost_agent
                })
                return f"CypherGenerationError: {str(e)}"
        cost_gpt += cost
        print(cypher_query)

        try:
            related_nodes = execute_query(cypher_query)
        except Exception as e:
            log_error("DatabaseQueryError", {
                "query": cypher_query,
                "error": str(e), 
                "cost_gpt": cost_gpt,
                "cost_embed": cost_embed,
                "cost_agent": cost_agent,
                "total_cost": cost_gpt + cost_embed + cost_agent
            })
            return f"DatabaseQueryError: {str(e)}"
        if not related_nodes:
            log_error("RelatedNodesNotFoundError", {
                        "query": cypher_query,
                        "nodes": all_relevant_nodes, 
                        "cost_gpt": cost_gpt,
                        "cost_embed": cost_embed,
                        "cost_agent": cost_agent,
                        "total_cost": cost_gpt + cost_embed + cost_agent
                    })
            return "No available information. Please, reword your question or try another one."
        print(related_nodes)
        try:
            final_prompt = enrich_prompt(question,related_nodes)
            print(final_prompt)
            
        except Exception as e:
            log_error("EnrichmentError", {
                "question": question,
                "nodes": related_nodes,
                "error": str(e), 
                "cost_gpt": cost_gpt,
                "cost_embed": cost_embed,
                "cost_agent": cost_agent,
                "total_cost": cost_gpt + cost_embed + cost_agent
            })
            return f"EnrichmentError: {str(e)}"
        try:
            final_answer, cost = await call_llm(final_prompt)
        except Exception as e:
            log_error("ResponseGenerationError", {
                "prompt": final_prompt,
                "error": str(e), 
                "cost_gpt": cost_gpt,
                "cost_embed": cost_embed,
                "cost_agent": cost_agent,
                "total_cost": cost_gpt + cost_embed + cost_agent
            })
            return f"ResponseGenerationError: {str(e)}"
        cost_gpt+= cost

        end = time.time()
        elapsed_time = end-start

        log_query({"question": question,
            "entities": extracted_entities,
            "simple_question": simplicity["is_simple"],
            "cypher_query": cypher_query,
            "prompt": final_prompt,
            "llm_response": final_answer,
            "time_elapsed": elapsed_time,
            "cost_gpt": cost_gpt,
            "cost_embed": cost_embed,
            "cost_agent": cost_agent,
            "total_cost": cost_gpt + cost_embed + cost_agent
        })


        return final_answer
    except Exception as e:
        log_error("UnhandledException", {
            "question": question,
            "error": str(e), 
            "cost_gpt": cost_gpt,
            "cost_embed": cost_embed,
            "cost_agent": cost_agent,
            "total_cost": cost_gpt + cost_embed + cost_agent
        })
        return "An unexpected error occurred while processing your question."

#how to address the lack of flexibility to enchance model variants comparison that happens in EMF-based model variants?
#what problems do software developers have?