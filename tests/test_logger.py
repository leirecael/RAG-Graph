import os
import json
import tempfile
from unittest.mock import patch
from app.data.logger import log_data, log_error

def test_log_data_writes_json_line():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.jsonl")

        with patch("app.data.logger.DATA_LOG", data_path):
            log_data({"message": "test log", "value": 123})

            with open(data_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["message"] == "test log"
        assert data["value"] == 123

def test_log_error_writes_timestamped_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        error_path = os.path.join(tmpdir, "errors.jsonl")

        with patch("app.data.logger.ERROR_LOG", error_path):
            log_error("TestError", {"info": "something failed"})

            with open(error_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        assert len(lines) == 1
        data = json.loads(lines[0])
        assert "timestamp" in data
        assert data["error_type"] == "TestError"
        assert data["details"]["info"] == "something failed"