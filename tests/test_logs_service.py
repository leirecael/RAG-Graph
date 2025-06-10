import pandas as pd
from unittest.mock import patch
from app.logic.logs_service import LogsService

test_log_serv = LogsService()

#------parse_logs-------
def test_parse_logs_returns_correct_structure():
    """
    Test that parse_logs correctly categorizes log entries by type and returns error logs as a list.

    Verifies:
        - Logs are categorized into known types only.
        - Logs with unknown types are excluded.
        - Error logs are returned unchanged.
    """
    data_entries = [
        {"log_type": "register_query", "info": "query1"},
        {"log_type": "llm_call", "info": "llm1"},
        {"log_type": "embedding", "info": "embed1"},
        {"log_type": "database", "info": "db1"},
        {"log_type": "unknown_type", "info": "skip me"}
    ]
    error_entries = [
        {"error_type": "Timeout", "message": "API call failed"}
    ]

    with patch("app.logic.logs_service.LogReader.read_data_logs", return_value=data_entries), patch("app.logic.logs_service.LogReader.read_error_logs", return_value=error_entries):
        logs_by_type, error_logs = test_log_serv.parse_logs()

    assert len(logs_by_type["register_query"]) == 1
    assert len(logs_by_type["llm_call"]) == 1
    assert len(logs_by_type["embedding"]) == 1
    assert len(logs_by_type["database"]) == 1
    assert "unknown_type" not in logs_by_type
    assert error_logs == error_entries


#------get_log_statistics_by_type-------
def test_get_log_statistics_by_type_with_valid_data():
    """
    Test that statistics are correctly aggregated for a log type with valid cost and duration fields.

    Verifies:
        - Total cost is correctly summed.
        - Average cost and duration are computed correctly.
        - Count matches the number of log entries.
        - The returned data frame is a pandas DataFrame instance.
    """
    entries = [
        {"log_type": "register_query", "cost": 2.5, "log_duration_sec": 1.0},
        {"log_type": "register_query", "cost": 3.5, "log_duration_sec": 2.0}
    ]

    with patch("app.logic.logs_service.LogReader.read_data_logs", return_value=entries), patch("app.logic.logs_service.LogReader.read_error_logs", return_value=[]):
        stats = test_log_serv.get_log_statistics_by_type()

    reg_stats = stats["register_query"]
    assert reg_stats["total_cost"] == 6.0
    assert round(reg_stats["avg_cost"], 2) == 3.0
    assert reg_stats["avg_duration_s"] == 1.5
    assert reg_stats["count"] == 2
    assert isinstance(reg_stats["df"], pd.DataFrame)

def test_get_log_statistics_by_type_with_missing_fields():
    """
    Test that statistics handle logs with missing 'cost' and 'log_duration_sec' fields.

    Verifies:
        - total_cost and avg_duration_s are None if missing in data.
        - Count of logs is accurate despite missing fields.
    """
    entries = [
        {"log_type": "llm_call", "info": "call 1"},
        {"log_type": "llm_call", "info": "call 2"}
    ]

    with patch("app.logic.logs_service.LogReader.read_data_logs", return_value=entries), patch("app.logic.logs_service.LogReader.read_error_logs", return_value=[]):
        stats = test_log_serv.get_log_statistics_by_type()

    llm_stats = stats["llm_call"]
    assert llm_stats["total_cost"] is None
    assert llm_stats["avg_duration_s"] is None
    assert llm_stats["count"] == 2

def test_get_log_statistics_by_type_ignores_non_numeric_costs_and_durations():
    """
    Test that invalid cost/duration values are coerced to NaN and handled correctly in statistics.

    Verifies:
        - Non-numeric cost and duration values do not raise errors.
        - Numeric values are summed and averaged correctly.
        - Count reflects total entries despite invalid values.
    """
    entries = [
        {"log_type": "embedding", "cost": "abc", "log_duration_sec": "2"},
        {"log_type": "embedding", "cost": "3.0", "log_duration_sec": "xyz"}
    ]

    with patch("app.logic.logs_service.LogReader.read_data_logs", return_value=entries), patch("app.logic.logs_service.LogReader.read_error_logs", return_value=[]):
        stats = test_log_serv.get_log_statistics_by_type()

    emb_stats = stats["embedding"]
    assert round(emb_stats["total_cost"], 1) == 3.0
    assert round(emb_stats["avg_duration_s"], 1) == 1.0  # One is 2.0, the other is coerced to 0
    assert emb_stats["count"] == 2

def test_get_log_statistics_by_type_with_task_names():
    """
    Test that task-level statistics are generated when 'task_name' is included in log entries.

    Verifies:
        - Statistics are grouped correctly by task_name.
        - Counts and costs per task are accurate.
        - Task statistics exist in the returned data structure.
    """
    entries = [
        {"log_type": "embedding", "cost": 2.0, "log_duration_sec": 1.0, "task_name": "task_A"},
        {"log_type": "embedding", "cost": 3.0, "log_duration_sec": 2.0, "task_name": "task_A"},
        {"log_type": "embedding", "cost": 1.0, "log_duration_sec": 1.5, "task_name": "task_B"}
    ]

    with patch("app.logic.logs_service.LogReader.read_data_logs", return_value=entries), patch("app.logic.logs_service.LogReader.read_error_logs", return_value=[]):
        stats = test_log_serv.get_log_statistics_by_type()

    emb_stats = stats["embedding"]
    assert "tasks" in emb_stats
    assert emb_stats["tasks"]["task_A"]["count"] == 2
    assert emb_stats["tasks"]["task_A"]["total_cost"] == 5.0
    assert emb_stats["tasks"]["task_B"]["count"] == 1
    assert emb_stats["tasks"]["task_B"]["total_cost"] == 1.0