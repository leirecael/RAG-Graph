from cache import AsyncTTL
from data.neo4j_client import Neo4jClient
from logic.llm_tasks import LlmTasks
from logs.logger import Logger
from logic. neo4j_logic import Neo4jLogic
import time
from datetime import datetime
from models.entity import EntityList, Entity
from models.question import Question
import re
import json
from presidio_analyzer import AnalyzerEngine

class Orchestrator:
    """
    Central coordinator for handling natural language questions using a graph-based RAG pipeline.

    This class integrates multiple components including a Neo4j client, logic handlers,
    language model tasks, logging, and PII detection to process user questions.

    Attributes:
        neo4j_client (Neo4jClient): Communicates with the Neo4j graph database.
        llm_tasks (LlmTasks): Handles tasks involving the language model.
        neo4j_logic (Neo4jLogic): Handles the queries sent to the database and the responses received.
        logger (Logger): Used for logging data and errors during question processing.
        pii_analyzer (AnalyzerEngine): Detects personally identifiable information (PII) in user input.

    Methods:
        contains_pii(text): Detects whether the input contains PII.
        sanitize_input(text): Cleans input by removing special characters.
        process_question(userQuestion): Full RAG pipeline for processing and answering a user's question.
    """

    def __init__(self):
        """
        Initializes the Orchestrator and its supporting components.

        Sets up clients and services required for handling RAG logic, logging, and
        PII detection.
        """
        self.neo4j_client = Neo4jClient()
        self.llm_tasks = LlmTasks()
        self.neo4j_logic = Neo4jLogic()
        self.logger = Logger()
        self.pii_analyzer = AnalyzerEngine()

    def contains_pii(self, text: str) -> bool:
        """
        Check if a given text contains any PII (Personally Identifiable Information).
        
        Args:
            text (str): The input text to analyze.
            
        Returns:
            bool: True if PII is detected, False otherwise.
        """
        #PII entities to block
        PII_ENTITIES = [
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "IBAN_CODE",
            "PERSON",
            "IP_ADDRESS",
            "MEDICAL_LICENSE",
            "URL",
            "CRYPTO",
            "ES_NIF",
            "ES_NIE"
        ]
        results = self.pii_analyzer.analyze(text=text, entities=PII_ENTITIES, language='en')
        return len(results) > 0

    def sanitize_input(self, text: str) -> str:
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
    async def process_question(self, userQuestion: str) -> str:
        """
        Process a user's natural language question and generate a natural language response
        using only the information contained in the graph database.

        This function performs several tasks:
        1. Blocks personal information and sanitizes the input.
        2. Validates the user question.
        3. Extracts entities from the validated question.
        4. Generates embeddings for those entities.
        5. Does a similarity search in the database. First with entity types, and if there are no results, a free search.
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
        if self.contains_pii(userQuestion):
            self.logger.log_error("InvalidQuestionPII", {
                    "question": "PII containing question"
                })
            return "Invalid question, contains PII or other unauthorized text, try again."
        
        #Sanitize the user's input
        sanitized_question = self.sanitize_input(userQuestion)
        if len(sanitized_question) == 0:
            return "Invalid question, try again."

        #2. Validate the question
        try:      
            question, cost = await self.llm_tasks.validate_question(sanitized_question)
            
            # If validation fails, log the error and inform the user
            if not question.is_valid:           
                self.logger.log_error("InvalidQuestion", {
                    "question": question.value,
                    "reason": question.reasoning, 
                })
                return f"Your question is not valid. Reason: {question.reasoning}"
        except RuntimeError as e:
            raise 
        except Exception as e:
            self.logger.log_error("ValidationError", {
                    "question": sanitized_question,
                    "error": str(e), 
                })
            raise
        total_cost +=cost
        
        #3. Extract entities from the question
        try:
            extracted_entities, cost = await self.llm_tasks.extract_entities(question.value)
        except RuntimeError as e:
            raise
        except Exception as e:
            self.logger.log_error("EntityExtractionError", {
                    "question": question.value,
                    "error": str(e), 
                })
            raise
        total_cost +=cost
        
        #4. Generate embeddings for the extracted entities
        try:
            extracted_entities.entities, cost = await self.llm_tasks.generate_entity_embeddings(extracted_entities.entities)

            #Filter the entities that have a value
            entities_with_value = [e for e in extracted_entities.entities if e.value is not None]
            
            total_cost += cost
        except RuntimeError as e:
            raise 
        except Exception as e:
                self.logger.log_error("EmbeddingError", {
                    "question": question.value,
                    "error": str(e), 
                })
                raise
        
        #5. Do a similarity search in the database
        try:       
            if entities_with_value:
                start_sim = time.time()
                queries = self.neo4j_logic.generate_similarity_queries(entities_with_value)
                db_results = self.neo4j_client.execute_multiple_queries(queries)
                similarity_results = self.neo4j_logic.parse_similarity_results(db_results)

                end_sim = time.time()

                #Prepare queries for logging without embedding value
                queries_for_logging = []
                for q in queries:
                    q_log = q.copy()
                    if "params" in q_log:
                        q_log["params"] = {k: v for k, v in q_log["params"].items() if k != "embedding"}
                    queries_for_logging.append(q_log)

                #Log the similarity search's performance
                self.logger.log_data({
                    "timestamp": datetime.now().isoformat(),
                    "log_type": "database",
                    "task_name": "similarity_calculation",
                    "user_prompt": question.value,
                    "cypher_query": json.dumps([query for query in queries_for_logging]),
                    "final_response": json.dumps(similarity_results),
                    "log_duration_sec": end_sim-start_sim,
                })
        except Exception as e:
                self.logger.log_error("SimilarityError", {
                    "question": question.value,
                    "entities": [entity.value for entity in entities_with_value],
                    "entity_types": [entity.type for entity in entities_with_value],
                    "error": str(e), 
                })
                raise

        #Organize the retreived information for the Cypher query generation
        all_relevant_nodes = {} 
        entities_with_no_value = [e for e in extracted_entities.entities if e.value is None]
        not_found_list = []
        for entity in entities_with_value:
            if entity.type in similarity_results:
                all_relevant_nodes[entity.type] = similarity_results[entity.type]
            else:
                #Save the entity to try again with another method.
                not_found_list.append(entity)
                
        for entity in entities_with_no_value:
            all_relevant_nodes[entity.type] = None

        #If there were entities not found, try again.
        if len(not_found_list) > 0:
            try:
                #Try semantic search but with all entity types/labels
                start_sim = time.time()

                queries = self.neo4j_logic.generate_similarity_queries_no_label(not_found_list)
                db_results = self.neo4j_client.execute_multiple_queries(queries)
                similarity_results = self.neo4j_logic.parse_similarity_results(db_results)

                end_sim = time.time()

                #Prepare queries for logging without embedding value
                queries_for_logging = []
                for q in queries:
                    q_log = q.copy()
                    if "params" in q_log:
                        q_log["params"] = {k: v for k, v in q_log["params"].items() if k != "embedding"}
                    queries_for_logging.append(q_log)

                #Log the retry's performance
                self.logger.log_data({
                    "timestamp": datetime.now().isoformat(),
                    "log_type": "database",
                    "task_name": "similarity_retry",
                    "user_prompt": question.value,
                    "cypher_query": json.dumps([query for query in queries_for_logging]),
                    "final_response": json.dumps(similarity_results),
                    "log_duration_sec": end_sim-start_sim,
                })

                if not similarity_results:
                    #If the database didn't find any information about an entity, log the error and inform the user
                    self.logger.log_error("RetryNotFoundError", {
                            "question": question.value,
                            "entity": [ent.value for ent in not_found_list],
                        })
                    return f"No data found, even after retry."
                for label in similarity_results.keys():
                    all_relevant_nodes[label] = similarity_results[label]
            except Exception as e:
                self.logger.log_error("RetryError", {
                    "question": question.value,
                    "entities": [ent.value for ent in not_found_list],
                    "error": str(e), 
                })
                raise

        #6. Generate a Cypher query based on the retrieved nodes
        try:
            cypher_query, cost = await self.llm_tasks.create_cypher_query(question.value, all_relevant_nodes)
            if cypher_query == "":
                self.logger.log_error("NoCypherError", {
                    "question": question.value,
                    "nodes": all_relevant_nodes,
                })
                return "Query generation returned nothing, try another question."
        except RuntimeError as e:
            raise 
        except Exception as e:
                self.logger.log_error("CypherGenerationError", {
                    "question": question.value,
                    "nodes": all_relevant_nodes,
                    "error": str(e), 
                })
                raise
        total_cost +=cost

        #7. Execute the Cypher query and parse the results
        try:
            start_db = time.time()
            db_results = self.neo4j_client.execute_query(cypher_query)
            related_nodes = self.neo4j_logic.parse_related_nodes_results(db_results)
            end_db = time.time()

            #Log the query execution's performance
            self.logger.log_data({
                "timestamp": datetime.now().isoformat(),
                "log_type": "database",
                "task_name": "cypher_execution",
                "user_prompt": question.value,
                "cypher_query": cypher_query,
                "final_response": json.dumps(related_nodes),
                "log_duration_sec": end_db-start_db,
            })
        except Exception as e:
            self.logger.log_error("DatabaseQueryError", {
                "question": question.value,
                "query": cypher_query,
                "error": str(e), 
            })
            raise
        
        # If the result is empty, inform the user.
        entities_empty = all(not v for v in related_nodes["entities"].values())
        if entities_empty and not related_nodes["relationships"] and not related_nodes["others"]:
            self.logger.log_error("RelatedNodesNotFoundError", {
                        "question": question.value,
                        "query": cypher_query,
                        "nodes": all_relevant_nodes, 
                    })
            return "No available information. Please, reword your question or try another one."

        #8. Generate the final answer in natural languague
        try:
            final_answer, cost = await self.llm_tasks.generate_final_answer(question.value, related_nodes)
        except RuntimeError as e:
            raise
        except Exception as e:
            self.logger.log_error("ResponseGenerationError", {
                "question": question.value,
                "context": related_nodes,
                "error": str(e), 
            })
            raise
        total_cost +=cost

        end = time.time()
        elapsed_time = end-start

        #Log the response generation's performance
        self.logger.log_data({
            "timestamp": datetime.now().isoformat(),
            "log_type": "register_query",
            "user_prompt": question.value,
            "final_response": final_answer,
            "log_duration_sec": elapsed_time,
            "cost": total_cost
        })

        return final_answer
