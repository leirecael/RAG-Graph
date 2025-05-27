from cache import AsyncTTL
from data.neo4j_client import execute_query, execute_multiple_queries
from logic.llm_tasks import validate_question, extract_entities, create_cypher_query, generate_final_answer, generate_entity_embeddings
from logs.logger import log_data, log_error
from logic. neo4j_logic import generate_similarity_queries, parse_similarity_results, parse_related_nodes_results
import json
import time
from datetime import datetime
from models.entity import EntityList, Entity
from models.question import Question
import re
from presidio_analyzer import AnalyzerEngine

pii_analyzer = AnalyzerEngine()

def contains_pii(text: str) -> bool:
    """
    Check if a given text contains any PII (Personally Identifiable Information).
    
    Args:
        text (str): The input text to analyze.
        
    Returns:
        bool: True if PII is detected, False otherwise.
    """
    results = pii_analyzer.analyze(text=text, entities=[], language='en')
    return len(results) > 0

def sanitize_input(text: str) -> str:
    """
    Sanitize the user's input and remove all special characters except spaces and alphanumeric characters.

    Args:
        text (str): The user's input

    Returns:
        str: Sanitized user input
    """
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text) 
    return cleaned.strip()

@AsyncTTL(time_to_live=3600, maxsize=1024)
async def process_question(userQuestion: str) -> str:
    """
    Process a user's natural language question and generate a natural language response
    using only the information contained in the graph database.

    This function performs several tasks:
    1. Blocks personal information and sanitizes the input.
    2. Validates the user question.
    3. Extracts entities from the validated question.
    4. Generates embeddings for those entities.
    5. Does a similarity search in the database.
    6. Generates a Cypher query to retrieve relevant data from the database.
    7. Executes the query and parses the results.
    8. Generates a natural language response based on the retrieved information.

    Caches results using AsyncTTL with a 1-hour TTL and a maximum size of 1024 entries.

    Args:
        userQuestion (str): The question asked by the user.

    Returns:
        str: Final answer generated based on the retrieved data or an error message.
    """
    start = time.time()
    total_cost = 0

    #1. Block personal information and sanitize the input.
    #Check if the questions contains PII
    if contains_pii(userQuestion):
        log_error("InvalidQuestionPII", {
                "question": "PII containing question"
            })
        return "Invalid question, contains PII, try again."
    
    #Sanitize the user's input
    sanitized_question = sanitize_input(userQuestion)
    if len(sanitized_question) == 0:
        return "Invalid question, try again."

    #2. Validate the question
    try:      
        question, cost =  await validate_question(sanitized_question)
        
        # If validation fails, log the error and inform the user
        if not question.is_valid:           
            log_error("InvalidQuestion", {
                "question": question.value,
                "reason": question.reasoning, 
            })
            return f"Your question is not valid. Reason: {question.reasoning}"
    
    except Exception as e:
        log_error("ValidationError", {
                "question": sanitized_question,
                "error": str(e), 
            })
        raise
    total_cost +=cost
    
    #3. Extract entities from the question
    try:
        extracted_entities, cost = await extract_entities(question.value)
        
    except Exception as e:
        log_error("EntityExtractionError", {
                "question": question.value,
                "error": str(e), 
            })
        raise
    total_cost +=cost
    
    #4. Generate embeddings for the extracted entities
    try:
        extracted_entities.entities, cost = await generate_entity_embeddings(extracted_entities.entities)

        #Filter the entities that have a value
        entities_with_value = [e for e in extracted_entities.entities if e.value is not None]
        
        total_cost += cost
    except Exception as e:
            log_error("EmbeddingError", {
                "question": question.value,
                "error": str(e), 
            })
            raise
    
    #5. Do a similarity search in the database
    try:       
        if entities_with_value:
            start_sim = time.time()
            queries = generate_similarity_queries(entities_with_value)
            db_results = execute_multiple_queries(queries)
            similarity_results = parse_similarity_results(db_results)
            end_sim = time.time()

            #Log the similarity search's performance
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

    #Organize the retreived information for the Cypher query generation
    all_relevant_nodes = {} 
    entities_with_no_value = [e for e in extracted_entities.entities if e.value is None]

    for entity in entities_with_value:
        if entity.type in similarity_results:
            all_relevant_nodes[entity.type] = similarity_results[entity.type]
        else:
            #If the database didn't find any information about an entity, log the error and inform the user
            log_error("SimilarNotFoundError", {
                "question": question.value,
                "entity": entity.value,
                "entity_type": entity.type,
            })
            return f"No data found about {entity.value}."

    for entity in entities_with_no_value:
        all_relevant_nodes[entity.type] = None

    #6. Generate a Cypher query based on the retrieved nodes
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

    #7. Execute the Cypher query and parse the results
    try:
        start_db = time.time()
        db_results = execute_query(cypher_query)
        related_nodes = parse_related_nodes_results(db_results)
        end_db = time.time()

        #Log the query execution's performance
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
    
    # If the result is empty, inform the user.
    if len(related_nodes["entities"]) == 0 and len(related_nodes["relationships"]) == 0 and len(related_nodes["others"]) == 0:
        log_error("RelatedNodesNotFoundError", {
                    "question": question.value,
                    "query": cypher_query,
                    "nodes": all_relevant_nodes, 
                })
        return "No available information. Please, reword your question or try another one."

    #8. Generate the final answer in natural languague
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

    #Log the response generation's performance
    log_data({
        "timestamp": datetime.now().isoformat(),
        "log_type": "register_query",
        "user_prompt": question.value,
        "final_response": final_answer,
        "log_duration_sec": elapsed_time,
        "cost": total_cost
    })

    return final_answer
     