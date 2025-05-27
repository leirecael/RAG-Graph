from logs.log_reader import read_data_logs,read_error_logs
import pandas as pd
def get_logs()->tuple[dict,list[dict]]:
    """
    Retrieves and parses logs from the system.

    Returns:
        tuple[dict, list[dict]]: Dictionary containing parsed logs by type (e.g., 'register_query', 'llm_call', etc.). List of error log entries.
    """
    data_logs = read_data_logs()
    logs_by_type = {
        "register_query": [],
        "llm_call": [],
        "embedding": [],
        "database": []
    }

    #Assign logs to their corresponding categories based on 'log_type'
    for log in data_logs:
        log_type = log.get("log_type")
        if log_type in logs_by_type:
            logs_by_type[log_type].append(log)

    error_logs = read_error_logs()
    return logs_by_type, error_logs


def get_log_statistics_by_type() -> dict:
    """
    Computes statistics for each type of log type and task.

    Returns:
        dict: A dictionary containing statistics for each log type and for each log type its log tasks, including:
            - total_cost: Sum of cost for logs/tasks that include a 'cost' field.
            - avg_cost: Average cost of logs/tasks with a 'cost' field.
            - avg_duration_s: Average duration in seconds for logs/tasks with a 'log_duration_sec' field.
            - count: Total number of log entries of that type/task.
            - df: Original DataFrame.
    """
    logs_by_type, _ = get_logs()
    stats = {}

    for log_type, entries in logs_by_type.items():
        df = pd.DataFrame(entries)

        if df.empty:
            continue

        #If the log has 'cost' convert the value to a number and fill it with 0 if it is missing or invalid.
        if "cost" in df.columns:
            df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0)
        
        #If the log has 'log_duration_sec' convert the value to a number and fill it with 0 if it is missing or invalid.
        if "log_duration_sec" in df.columns:
            df["log_duration_sec"] = pd.to_numeric(df["log_duration_sec"], errors="coerce").fillna(0)

        #Calculate stats per log type
        log_stats = {
            "total_cost": df["cost"].sum() if "cost" in df else None,
            "avg_cost": df["cost"].mean() if "cost" in df else None,
            "avg_duration_s": df["log_duration_sec"].mean() if "log_duration_sec" in df else None,
            "count": len(df),
            "df": df
        }

        # Add per-task statistics if 'task_name' exists
        if "task_name" in df.columns:
            log_stats["tasks"] = {}
            for task_name in df["task_name"].unique():
                #Filter dataframe rows by task_name and return new df
                task_df = df[df["task_name"] == task_name]
                log_stats["tasks"][task_name] = {
                    "total_cost": task_df["cost"].sum() if "cost" in task_df else None,
                    "avg_cost": task_df["cost"].mean() if "cost" in task_df else None,
                    "avg_duration_s": task_df["log_duration_sec"].mean() if "log_duration_sec" in task_df else None,
                    "count": len(task_df),
                    "df": task_df
                }

        stats[log_type] = log_stats

    return stats

