import os
import json
import tempfile
import pandas as pd
from unittest.mock import patch
from app.data.log_reader import read_data_logs, read_error_logs, get_log_statistics_by_type

def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

def test_read_data_logs_groups_by_type():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_log = os.path.join(tmpdir, "data.jsonl")
        entries = [
            {"log_type": "register_query", "cost": 2.5},
            {"log_type": "embedding", "cost": 1.0},
            {"log_type": "register_query", "cost": 3.0},
            {"log_type": "other", "cost": 0.5}
        ]
        write_lines(data_log, entries)

        with patch("app.data.log_reader.DATA_LOG", data_log):
            logs = read_data_logs()

        assert "register_query" in logs
        assert len(logs["register_query"]) == 2
        assert len(logs["embedding"]) == 1
        assert len(logs["database"]) == 0

def test_read_error_logs_parses_all_valid_lines():
    with tempfile.TemporaryDirectory() as tmpdir:
        error_log = os.path.join(tmpdir, "errors.jsonl")
        entries = [
            {"error_type": "ValidationError", "details": {"info": "bad input"}},
            {"error_type": "DatabaseError", "details": {"info": "query failed"}}
        ]
        write_lines(error_log, entries)

        with patch("app.data.log_reader.ERROR_LOG", error_log):
            logs = read_error_logs()

        assert len(logs) == 2
        assert logs[0]["error_type"] == "ValidationError"

def test_get_log_statistics_by_type_returns_aggregates():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_log = os.path.join(tmpdir, "data.jsonl")
        entries = [
            {"log_type": "register_query", "cost": 2.5, "log_duration_sec": 1.0},
            {"log_type": "register_query", "cost": 3.5, "log_duration_sec": 2.0},
            {"log_type": "register_query", "cost": "invalid", "log_duration_sec": "invalid"},
            {"log_type": "embedding", "cost": 1.0, "log_duration_sec": 0.5},
        ]
        write_lines(data_log, entries)

        with patch("app.data.log_reader.DATA_LOG", data_log):
            stats = get_log_statistics_by_type()

        reg_stats = stats["register_query"]
        assert reg_stats["total_cost"] == 6.0
        assert round(reg_stats["avg_cost"], 2) == 2.0
        assert reg_stats["count"] == 3
        assert isinstance(reg_stats["df"], pd.DataFrame)