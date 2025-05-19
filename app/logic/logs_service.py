from logs.log_reader import read_data_logs,read_error_logs
import pandas as pd
def get_logs()->tuple[dict,list[dict]]:
    data_logs = read_data_logs()
    logs_by_type = {
        "register_query": [],
        "llm_call": [],
        "embedding": [],
        "database": []
    }
    for log in data_logs:
        log_type = log.get("log_type")
        if log_type in logs_by_type:
            logs_by_type[log_type].append(log)

    error_logs = read_error_logs()
    return logs_by_type, error_logs


def get_log_statistics_by_type() -> dict:
    logs_by_type, _ = get_logs()
    stats = {}

    for log_type, entries in logs_by_type.items():
        df = pd.DataFrame(entries)

        if df.empty:
            continue

        if "cost" in df.columns:
            df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0)

        if "log_duration_sec" in df.columns:
            df["log_duration_sec"] = pd.to_numeric(df["log_duration_sec"], errors="coerce").fillna(0)

        stats[log_type] = {
            "total_cost": df["cost"].sum() if "cost" in df else None,
            "avg_cost": df["cost"].mean() if "cost" in df else None,
            "avg_duration_s": df["log_duration_sec"].mean() if "log_duration_sec" in df else None,
            "count": len(df),
            "df": df
        }

    return stats

