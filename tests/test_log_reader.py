import os
import json
import tempfile
from unittest.mock import patch
from app.logs.log_reader import read_data_logs, read_error_logs

def write_lines(path: str, lines: list[dict]):
    """
    Helper function to write a list of dictionaries to a file as JSON lines.

    Args:
        path (str): File path to write to.
        lines (list[dict]): List of dictionaries to serialize as JSON lines.
    """
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

#------read_data_logs-------
def test_read_data_logs_reads_valid_lines():
    """
    Test that valid JSON entries in the data log file are read and returned correctly.

    Verifies:
        - Each valid line in the log is read correctly.
        - All entries are returned in order.
        - The content of each entry matches the original.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        data_log = os.path.join(tmpdir, "data.jsonl")
        entries = [
            {"message": "hey", "cost": 5},
            {"message": "hi", "cost": 0}
        ]
        write_lines(data_log, entries)

        #Patch log path and read entries
        with patch("app.logs.log_reader.DATA_LOG", data_log):
            logs = read_data_logs()

        assert len(logs) == 2
        assert logs[0]["message"] == "hey"

def test_read_data_logs_returns_empty_list_if_missing():
    """
    Test that reading from a nonexistent data log file returns an empty list.

    Verifies:
        - The function handles missing files correctly.
        - An empty list is returned instead of an exception.
    """
    with patch("app.logs.log_reader.DATA_LOG", "/non/existent/data.jsonl"):
        logs = read_data_logs()
        assert logs == []

def test_read_data_logs_skips_invalid_json():
    """
    Test that read_data_logs skips lines with invalid JSON and continues reading valid ones.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "data.jsonl")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("{invalid json line}\n")
            f.write(json.dumps({"message": "valid log"}) + "\n")

        with patch("app.logs.log_reader.DATA_LOG", log_path):
            logs = read_data_logs()

        assert isinstance(logs, list)
        assert len(logs) == 1 
        assert logs[0]["message"] == "valid log"

#------read_error_logs-------
def test_read_error_logs_parses_all_valid_lines():
    """
    Test that valid JSON entries in the error log file are read correctly.

    Verifies:
        - Each valid error entry is read correctly.
        - All expected fields (error_type, details, etc.) are preserved.
        - Entries are returned in the order they were written.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        error_log = os.path.join(tmpdir, "errors.jsonl")
        entries = [
            {"error_type": "ValidationError", "details": {"info": "bad input"}},
            {"error_type": "DatabaseError", "details": {"info": "query failed"}}
        ]
        write_lines(error_log, entries)

        #Patch log path and read entries
        with patch("app.logs.log_reader.ERROR_LOG", error_log):
            logs = read_error_logs()

        assert len(logs) == 2
        assert logs[0]["error_type"] == "ValidationError"

def test_read_error_logs_returns_empty_list_if_missing():
    """
    Test that reading from a nonexistent error log file returns an empty list.

    Verifies:
        - The function handles missing file paths correctly.
        - Returns an empty list rather than raising an exception.
    """
    with patch("app.logs.log_reader.ERROR_LOG", "/non/existent/path.jsonl"):
        logs = read_error_logs()
        assert logs == []

def test_read_error_logs_skips_invalid_json():
    """
    Test that read_error_logs skips lines with invalid JSON and continues reading valid ones.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "errors.jsonl")


        with open(log_path, "w", encoding="utf-8") as f:
            f.write("{invalid json line}\n")
            f.write(json.dumps({"message": "valid log"}) + "\n")

        with patch("app.logs.log_reader.ERROR_LOG", log_path):
            logs = read_error_logs()

        assert isinstance(logs, list)
        assert len(logs) == 1 
        assert logs[0]["message"] == "valid log"