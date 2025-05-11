import json
import os
import pandas as pd
from typing import List, Dict

LOG_DIR = "app/logs"
DATA_LOG = os.path.join(LOG_DIR, "data.jsonl")
ERROR_LOG = os.path.join(LOG_DIR, "errors.jsonl")

def read_data_logs() -> Dict[str, List[Dict]]:
    logs_by_type = {
        "register_query": [],
        "llm_call": [],
        "embedding": [],
        "database": []
    }

    if os.path.exists(DATA_LOG):
        with open(DATA_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    log_type = entry.get("log_type", "other")
                    if log_type in logs_by_type:
                        logs_by_type[log_type].append(entry)
                except json.JSONDecodeError:
                    continue
    return logs_by_type

def read_error_logs() -> List[Dict]:
    logs = []
    if os.path.exists(ERROR_LOG):
        with open(ERROR_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return logs

def get_log_statistics_by_type():
    logs_by_type = read_data_logs()
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