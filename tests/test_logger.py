import os
import json
import tempfile
import pytest
from unittest.mock import patch
from app.logs.logger import log_data, log_error

#------log_data-----------
def test_log_data_basic():
    """
    Test that log_data correctly writes a dictionary as a single JSON line into the data log file.

    Verifies:
        - The function writes exactly one line to the log file.
        - The JSON content written matches the dictionary passed in.
    """
    #Create a temporary directory to isolate file writing
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.jsonl")

        #Patch the global DATA_LOG path to redirect writes to the temp file
        with patch("app.logs.logger.DATA_LOG", data_path):
            log_data({"message": "test log", "value": 123})

            #Read the contents of the file to verify what was written
            with open(data_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        
        #Ensure only one line was written and contains the expected data
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["message"] == "test log"
        assert data["value"] == 123

def test_log_data_with_empty_dict():
    """
    Test that logging an empty dictionary still writes a valid JSON line.

    Verifies:
        - A single valid JSON line representing an empty dict is written.
        - No errors occur when logging empty data.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.jsonl")

        with patch("app.logs.logger.DATA_LOG", data_path):
            log_data({})

            with open(data_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data == {}

def test_log_data_rejects_non_dict():
    """
    Ensure log_data only accepts dicts.

    Verifies:
        - Passing a non-dict raises a TypeError.
    """
    with pytest.raises(TypeError):
        log_data("not a dict")

#------log_error-----------
def test_log_error_basic():
    """
    Test that log_error writes a JSON line with timestamp, error type, and details into the error log file.

    Verifies:
        - Exactly one JSON line is written.
        - The JSON includes 'timestamp', 'error_type', and 'details'.
        - The content of 'error_type' and 'details' matches the input.
    """
    #Create a temporary directory to isolate file writing
    with tempfile.TemporaryDirectory() as tmpdir:
        error_path = os.path.join(tmpdir, "errors.jsonl")

        #Patch the global ERROR_LOG path to redirect writes to the temp file
        with patch("app.logs.logger.ERROR_LOG", error_path):
            log_error("TestError", {"info": "something failed"})

            #Read the contents of the file to verify what was written
            with open(error_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        #Ensure only one log line was written with expected fields
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert "timestamp" in data
        assert data["error_type"] == "TestError"
        assert data["details"]["info"] == "something failed"

def test_log_error_with_empty_details():
    """
    Test that logging an error with an empty details dict still writes correctly.

    Verifies:
        - The log line contains the correct error_type.
        - Details field is an empty dict as provided.
        - Timestamp is included.
        - No exceptions are raised.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        error_path = os.path.join(tmpdir, "errors.jsonl")

        with patch("app.logs.logger.ERROR_LOG", error_path):
            log_error("EmptyDetailError", {})

            with open(error_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["error_type"] == "EmptyDetailError"
        assert isinstance(data["details"], dict)
        assert data["details"] == {}
        assert "timestamp" in data

def test_log_error_type_error():
    """
    Test that an unexpected type raises an error.

    Verifies:
        - A list instead of a dictionary raises TypeError
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        error_path = os.path.join(tmpdir, "errors.jsonl")

        with patch("app.logs.logger.ERROR_LOG", error_path):
            with pytest.raises(TypeError):
                log_error("TypeError", [])