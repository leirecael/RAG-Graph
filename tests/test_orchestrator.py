from unittest.mock import AsyncMock
import pytest
import json
from logic.orchestrator import process_question, contains_pii, sanitize_input
from app.models.entity import Entity, EntityList ,EntityEnum
from app.models.question import Question

#------contains_pii---------
def test_contains_pii_positive():
    """
    Test that PII is correctly identified in the text.

    Verifies:
        - The function detects personally identifiable information (e.g., emails).
        - Returns True when PII is present.
    """
    text_with_pii = "My name is Pedro and my email is pedro@gmail.com"
    assert contains_pii(text_with_pii) is True


def test_contains_pii_negative():
    """
    Test that non-PII text does not trigger false positives.

    Verifies:
        - The function returns False when no PII is detected.
        - Safe, generic text passes without issue.
    """
    safe_text = "What problems do software developers face?"
    assert contains_pii(safe_text) is False

#------sanitize_input---------
def test_sanitize_input_removes_special_characters():
    """
    Test that sanitize_input removes punctuation and special characters while preserving spaces and alphanumerics.

    Verifies:
        - All special characters and punctuation are removed.
        - Spaces and alphanumeric characters remain intact.
    """
    raw_text = "I have a question~~&---"
    cleaned = sanitize_input(raw_text)
    assert cleaned == "I have a question"


#------process_question---------
@pytest.mark.asyncio
async def test_process_question_rejects_pii(mocker):
    """
    Test that pii is rejected.

    Verifies:
        - PII is rejected and the corresponding message is sent.
    """
    mocker.patch("logic.orchestrator.contains_pii", return_value = True)

    response = await process_question("test@gmail.com")

    assert "Invalid question, contains PII, try again." in response

@pytest.mark.asyncio
async def test_process_question_empty_after_sanitize(mocker):
    """
    Test that a question that becomes empty after sanitization is rejected.

    Verifies:
        - Rejected empty question sends the corresponding message.
    """

    mocker.patch("logic.orchestrator.contains_pii", return_value = False)
    mocker.patch("logic.orchestrator.sanitize_input", return_value = "")

    response = await process_question("!!!")
    assert "Invalid question, try again." in response

@pytest.mark.asyncio
async def test_process_question_invalid_question(mocker):
    """
    Simulate an invalid question returned from the validator.

    Verifies:
        - Invalid question is rejected and the corresponding message is sent.
    """

    fake_q = Question(value="X", is_valid=False, reasoning="Too vague")
    mocker.patch("logic.orchestrator.contains_pii", return_value = False)
    mocker.patch("logic.orchestrator.sanitize_input", return_value = "X")
    mocker.patch("logic.orchestrator.validate_question", return_value = (fake_q, 0.1))

    response = await process_question("X")

    assert "Your question is not valid. Reason:" in response

@pytest.mark.asyncio
async def test_process_question_entity_not_found_in_db(mocker):
    """
    Test behavior when an entity is not found in the similarity results.

    Verifies:
        - When data is not found the corresponding message is sent.
    """

    valid_q = Question(value="What is X?", is_valid=True, reasoning=None)
    entities = [Entity(value="nonexistent", type="goal", embedding=[0.1, 0.2, 0.3])]
    
    mocker.patch("logic.orchestrator.contains_pii", return_value = False)
    mocker.patch("logic.orchestrator.sanitize_input", return_value = "What is X")
    mocker.patch("logic.orchestrator.validate_question", return_value = (valid_q, 0.1))
    mocker.patch("logic.orchestrator.extract_entities", return_value = (EntityList(entities=entities), 0.1))
    mocker.patch("logic.orchestrator.generate_entity_embeddings", return_value = (entities, 0.1))
    mocker.patch("logic.orchestrator.generate_similarity_queries", return_value = [])
    mocker.patch("logic.orchestrator.execute_multiple_queries", return_value = [])
    mocker.patch("logic.orchestrator.parse_similarity_results", return_value = {})

    response = await process_question("What is X?")
    assert "No data found about" in response

@pytest.mark.asyncio
async def test_process_question_no_related_nodes(mocker):
    """
    Simulate empty results from parse_related_nodes_results.

    Verifies:
        - When related nodes are not found the corresponding message is sent.
    """
    from logic.orchestrator import process_question, Question, Entity, EntityList

    valid_q = Question(value="How to fix it?", is_valid=True, reasoning=None)
    entities = [Entity(value="X", type="problem", embedding=[0.1, 0.2, 0.3])]

    mocker.patch("logic.orchestrator.contains_pii", return_value = False)
    mocker.patch("logic.orchestrator.sanitize_input", return_value = "How to fix it")
    mocker.patch("logic.orchestrator.validate_question", return_value = (valid_q, 0.1))
    mocker.patch("logic.orchestrator.extract_entities", return_value = (EntityList(entities=entities), 0.1))
    mocker.patch("logic.orchestrator.generate_entity_embeddings", return_value = (entities, 0.1))
    mocker.patch("logic.orchestrator.generate_similarity_queries", return_value = [])
    mocker.patch("logic.orchestrator.execute_multiple_queries", return_value = [])
    mocker.patch("logic.orchestrator.parse_similarity_results", return_value = {"problem": ["X"]})
    mocker.patch("logic.orchestrator.create_cypher_query", return_value = ("MATCH () RETURN 1", 0.1))
    mocker.patch("logic.orchestrator.execute_query", return_value = [])
    mocker.patch("logic.orchestrator.parse_related_nodes_results", return_value = {"entities": {}, "relationships": [], "others": {}})

    response = await process_question("How to fix it?")
    assert "No available information" in response

@pytest.mark.asyncio
async def test_process_question_basic_path(mocker):
    """
    Full end-to-end basic path test of process_question.

    Verifies:
        - Input question is sanitized and validated.
        - Entities are extracted and embeddings generated.
        - Similarity queries are generated and executed.
        - Related nodes are parsed correctly.
        - Final natural language answer is generated including relevant data.
    """
    mocker.patch("logic.orchestrator.contains_pii", return_value = False)
    
    mocker.patch("logic.orchestrator.sanitize_input", return_value = "What problems do developers face")

    mocker.patch("logic.orchestrator.validate_question", return_value=(Question(value="What problems do developers face?",is_valid=True,reasoning=None), 0.001))

    mocker.patch("logic.orchestrator.extract_entities", return_value=(EntityList(entities=[Entity(value="developers", type=EntityEnum.stakeholder, embedding=None), Entity(value=None,type=EntityEnum.problem,embedding=None)]), 0.002))

    mocker.patch("logic.orchestrator.generate_entity_embeddings", return_value=([Entity(value='developers', type='stakeholder', embedding=[-0.013434951193630695, 0.013434951193630695]), Entity(value=None, type='problem', embedding=None)], 0.1))

    mocker.patch("logic.orchestrator.generate_similarity_queries", return_value=["SIMILARITY_QUERY"])
    mocker.patch("logic.orchestrator.execute_multiple_queries", return_value=[{"value":{"name": "developers", "similarity": 0.7, "labels":["stakeholder"]}}])
    mocker.patch("logic.orchestrator.parse_similarity_results", return_value={
        "stakeholder": ["developers"]
    })

    mocker.patch("logic.orchestrator.create_cypher_query", return_value=("MATCH ...", 0.001))

    mocker.patch("logic.orchestrator.execute_query", return_value={
        "records": {"record": "Info"},
    })
    mocker.patch("logic.orchestrator.parse_related_nodes_results", return_value={
        "entities": {
            "problems": {
                "Latency Issue": {
                    "description": "High response time",
                    "labels": ["problem"],
                    "hypernym": "performance"
                }
            },
            "stakeholders": {
                "developers": {
                    "description": "People writing code.",
                    "labels": ["stakeholder"],
                    "hypernym": "team members"
                }
            }
        },
        "relationships": [
            {"from": "Latency Issue", "to": "developers", "type": "concerns"}
        ],
        "others": []
    })

    # Mock enrichment and LLM
    mocker.patch("logic.orchestrator.generate_final_answer", return_value=("They face latency issues.", 0.01))

    # Call the function
    response = await process_question("What problems do developers face?")
    assert isinstance(response, str)
    assert "latency" in response.lower()


@pytest.mark.asyncio
async def test_process_question_invalid_question(mocker):
    """
    Simulates a user asking an invalid or out-of-domain question.

    Verifies:
        - The validation step marks the question as invalid.
        - The function returns a user-friendly response explaining invalidity.
    """
    mocker.patch("logic.orchestrator.validate_question", return_value=(Question(value="What's the weather?",is_valid=False,reasoning="Non-technical domain."), 0.001))

    response = await process_question("What's the weather?")
    assert "not valid" in response.lower()
    assert "non-technical" in response.lower()


@pytest.mark.asyncio
async def test_process_question_no_entities_found(mocker):
    """
    Tests process_question when no relevant entities are found for a valid question.

    Verifies:
        - Question passes validation.
        - Entities are extracted but no related nodes found in the database.
        - Function returns a response indicating lack of available information.
    """
    mocker.patch("logic.orchestrator.validate_question", return_value=(Question(value="What problems do doctors face?",is_valid=True, reasoning=None), 0.001))

    mocker.patch("logic.orchestrator.extract_entities", return_value=(EntityList(entities=[Entity(value="doctors", type=EntityEnum.stakeholder, embedding=None)]), 0.002))

    mocker.patch("logic.orchestrator.generate_entity_embeddings", return_value=([Entity(value='doctors', type='stakeholder', embedding=[-0.013434951193630695, 0.013434951193630695]), Entity(value=None, type='problem', embedding=None)], 0.1))
    mocker.patch("logic.orchestrator.parse_similarity_results", return_value={
        "stakeholder": ["doctors"]
    })

    mocker.patch("logic.orchestrator.generate_similarity_queries", return_value=["SIM_QUERY"])
    mocker.patch("logic.orchestrator.execute_multiple_queries", return_value=[{"value":{"name": "doctors", "similarity": 0.7, "labels":["stakeholder"]}}])

    mocker.patch("logic.orchestrator.create_cypher_query", return_value=("MATCH ...", 0.001))

    mocker.patch("logic.orchestrator.execute_query", return_value={
        "records": {},
    })
    mocker.patch("logic.orchestrator.parse_related_nodes_results", return_value={
        "entities": {},
        "relationships": [],
        "others": []
    })

    response = await process_question("What problems do doctors face?")
    assert "no available information" in response.lower()

@pytest.mark.asyncio
async def test_process_question_complete_with_valid_question():
    """
    Full end-to-end test without mocks, using the real DB and LLM.

    Verifies:
        - A valid questions gets a response with a valid question.
    """

    question = "What problems do software developers face?"
    response = await process_question(question)

    assert isinstance(response, str)
    assert "developers" in response.lower() or "software" in response.lower()
    assert "problem" in response.lower() 

@pytest.mark.asyncio
async def test_process_question_raises_on_generate_final_answer(mocker):
    """
    Simulate exception during final answer generation.

    Verifies:
        - Exception is correctly raised.
        - Error is logged
    """

    valid_q = Question(value="What problems?", is_valid=True, reasoning=None)
    entities = [Entity(value="X", type="problem", embedding=[0.1, 0.2, 0.3])]

    mocker.patch("logic.orchestrator.contains_pii", return_value= False)
    mocker.patch("logic.orchestrator.sanitize_input", return_value= "What problems?")
    mocker.patch("logic.orchestrator.validate_question", return_value= (valid_q, 0.1))
    mocker.patch("logic.orchestrator.extract_entities", return_value= (EntityList(entities=entities), 0.1))
    mocker.patch("logic.orchestrator.generate_entity_embeddings", return_value= (entities, 0.1))
    mocker.patch("logic.orchestrator.generate_similarity_queries", return_value= [])
    mocker.patch("logic.orchestrator.execute_multiple_queries", return_value= [])
    mocker.patch("logic.orchestrator.parse_similarity_results", return_value= {"problem": ["X"]})
    mocker.patch("logic.orchestrator.create_cypher_query", return_value= ("MATCH () RETURN 1", 0.1))
    mocker.patch("logic.orchestrator.execute_query", return_value= [])
    mocker.patch("logic.orchestrator.parse_related_nodes_results", return_value= {"entities": {"problems": {"X": {}}}, "relationships": [], "others": {}})
    mocker.patch("logic.orchestrator.generate_final_answer", side_effect=Exception("Mock error"))
        
    mock_log = mocker.patch("logic.orchestrator.log_error")

    with pytest.raises(Exception) as e:
        await process_question("What problems?")

    assert "Mock error" in str(e.value)
    
    mock_log.assert_called_once()
    args, _ = mock_log.call_args
    assert args[0] == "ResponseGenerationError"
    assert "question" in args[1]
    assert "error" in args[1]
    assert "Mock error" in args[1]["error"]

@pytest.mark.asyncio
async def test_process_question_raises_on_execute_query(mocker):
    """
    Simulate exception during query execution.

    Verifies:
        - Exception is correctly raised.
        - Error is logged
    """

    valid_q = Question(value="What problems?", is_valid=True, reasoning=None)
    entities = [Entity(value="X", type="problem", embedding=[0.1, 0.2, 0.3])]

    mocker.patch("logic.orchestrator.contains_pii", return_value= False)
    mocker.patch("logic.orchestrator.sanitize_input", return_value= "What problems?")
    mocker.patch("logic.orchestrator.validate_question", return_value= (valid_q, 0.1))
    mocker.patch("logic.orchestrator.extract_entities", return_value= (EntityList(entities=entities), 0.1))
    mocker.patch("logic.orchestrator.generate_entity_embeddings", return_value= (entities, 0.1))
    mocker.patch("logic.orchestrator.generate_similarity_queries", return_value= [])
    mocker.patch("logic.orchestrator.execute_multiple_queries", return_value= [])
    mocker.patch("logic.orchestrator.parse_similarity_results", return_value= {"problem": ["X"]})
    mocker.patch("logic.orchestrator.create_cypher_query", return_value= ("MATCH () RETURN 1", 0.1))
    mocker.patch("logic.orchestrator.execute_query", side_effect=Exception("Mock error"))
        
    mock_log = mocker.patch("logic.orchestrator.log_error")

    with pytest.raises(Exception) as e:
        await process_question("What problems?")

    assert "Mock error" in str(e.value)
    
    mock_log.assert_called_once()
    args, _ = mock_log.call_args
    assert args[0] == "DatabaseQueryError"
    assert "question" in args[1]
    assert "error" in args[1]
    assert "Mock error" in args[1]["error"]

@pytest.mark.asyncio
async def test_process_question_raises_on_create_cypher_query(mocker):
    """
    Simulate exception during cypher query generation.

    Verifies:
        - Exception is correctly raised.
        - Error is logged
    """

    valid_q = Question(value="What problems?", is_valid=True, reasoning=None)
    entities = [Entity(value="X", type="problem", embedding=[0.1, 0.2, 0.3])]

    mocker.patch("logic.orchestrator.contains_pii", return_value= False)
    mocker.patch("logic.orchestrator.sanitize_input", return_value= "What problems?")
    mocker.patch("logic.orchestrator.validate_question", return_value= (valid_q, 0.1))
    mocker.patch("logic.orchestrator.extract_entities", return_value= (EntityList(entities=entities), 0.1))
    mocker.patch("logic.orchestrator.generate_entity_embeddings", return_value= (entities, 0.1))
    mocker.patch("logic.orchestrator.generate_similarity_queries", return_value= [])
    mocker.patch("logic.orchestrator.execute_multiple_queries", return_value= [])
    mocker.patch("logic.orchestrator.parse_similarity_results", return_value= {"problem": ["X"]})
    mocker.patch("logic.orchestrator.create_cypher_query", side_effect=Exception("Mock error"))
   
    mock_log = mocker.patch("logic.orchestrator.log_error")

    with pytest.raises(Exception) as e:
        await process_question("What problems?")

    assert "Mock error" in str(e.value)
    
    mock_log.assert_called_once()
    args, _ = mock_log.call_args
    assert args[0] == "CypherGenerationError"
    assert "question" in args[1]
    assert "error" in args[1]
    assert "Mock error" in args[1]["error"]

@pytest.mark.asyncio
async def test_process_question_raises_on_execute_multiple_queries(mocker):
    """
    Simulate exception during multiple queries execution.

    Verifies:
        - Exception is correctly raised.
        - Error is logged
    """

    valid_q = Question(value="What problems?", is_valid=True, reasoning=None)
    entities = [Entity(value="X", type="problem", embedding=[0.1, 0.2, 0.3])]

    mocker.patch("logic.orchestrator.contains_pii", return_value= False)
    mocker.patch("logic.orchestrator.sanitize_input", return_value= "What problems?")
    mocker.patch("logic.orchestrator.validate_question", return_value= (valid_q, 0.1))
    mocker.patch("logic.orchestrator.extract_entities", return_value= (EntityList(entities=entities), 0.1))
    mocker.patch("logic.orchestrator.generate_entity_embeddings", return_value= (entities, 0.1))
    mocker.patch("logic.orchestrator.generate_similarity_queries", return_value= [])
    mocker.patch("logic.orchestrator.execute_multiple_queries", side_effect=Exception("Mock error"))
    
    mock_log = mocker.patch("logic.orchestrator.log_error")

    with pytest.raises(Exception) as e:
        await process_question("What problems?")

    assert "Mock error" in str(e.value)
    
    mock_log.assert_called_once()
    args, _ = mock_log.call_args
    assert args[0] == "SimilarityError"
    assert "question" in args[1]
    assert "error" in args[1]
    assert "Mock error" in args[1]["error"]

@pytest.mark.asyncio
async def test_process_question_raises_on_generate_entity_embeddings(mocker):
    """
    Simulate exception during entity embeddings generation.

    Verifies:
        - Exception is correctly raised.
        - Error is logged
    """

    valid_q = Question(value="What problems?", is_valid=True, reasoning=None)
    entities = [Entity(value="X", type="problem", embedding=[0.1, 0.2, 0.3])]

    mocker.patch("logic.orchestrator.contains_pii", return_value= False)
    mocker.patch("logic.orchestrator.sanitize_input", return_value= "What problems?")
    mocker.patch("logic.orchestrator.validate_question", return_value= (valid_q, 0.1))
    mocker.patch("logic.orchestrator.extract_entities", return_value= (EntityList(entities=entities), 0.1))
    mocker.patch("logic.orchestrator.generate_entity_embeddings", side_effect=Exception("Mock error"))

    mock_log = mocker.patch("logic.orchestrator.log_error")

    with pytest.raises(Exception) as e:
        await process_question("What problems?")

    assert "Mock error" in str(e.value)
    
    mock_log.assert_called_once()
    args, _ = mock_log.call_args
    assert args[0] == "EmbeddingError"
    assert "question" in args[1]
    assert "error" in args[1]
    assert "Mock error" in args[1]["error"]

@pytest.mark.asyncio
async def test_process_question_raises_on_extract_entities(mocker):
    """
    Simulate exception during entities extraction.

    Verifies:
        - Exception is correctly raised.
        - Error is logged
    """

    valid_q = Question(value="What problems?", is_valid=True, reasoning=None)

    mocker.patch("logic.orchestrator.contains_pii", return_value= False)
    mocker.patch("logic.orchestrator.sanitize_input", return_value= "What problems?")
    mocker.patch("logic.orchestrator.validate_question", return_value= (valid_q, 0.1))
    mocker.patch("logic.orchestrator.extract_entities", side_effect=Exception("Mock error"))

    mock_log = mocker.patch("logic.orchestrator.log_error")

    with pytest.raises(Exception) as e:
        await process_question("What problems?")

    assert "Mock error" in str(e.value)
    
    mock_log.assert_called_once()
    args, _ = mock_log.call_args
    assert args[0] == "EntityExtractionError"
    assert "question" in args[1]
    assert "error" in args[1]
    assert "Mock error" in args[1]["error"]

@pytest.mark.asyncio
async def test_process_question_raises_on_validate_question(mocker):
    """
    Simulate exception during question validation.

    Verifies:
        - Exception is correctly raised.
        - Error is logged
    """

    mocker.patch("logic.orchestrator.contains_pii", return_value= False)
    mocker.patch("logic.orchestrator.sanitize_input", return_value= "What problems?")
    mocker.patch("logic.orchestrator.validate_question", side_effect=Exception("Mock error"))

    mock_log = mocker.patch("logic.orchestrator.log_error")
    

    with pytest.raises(Exception) as e:
        await process_question("What problems?")

    assert "Mock error" in str(e.value)

    mock_log.assert_called_once()
    args, _ = mock_log.call_args
    assert args[0] == "ValidationError"
    assert "question" in args[1]
    assert "error" in args[1]
    assert "Mock error" in args[1]["error"]

    