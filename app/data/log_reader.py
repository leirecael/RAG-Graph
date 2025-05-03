import json
import os
from typing import List, Dict

LOG_DIR = "app/logs"
QUERY_LOG = os.path.join(LOG_DIR, "queries.jsonl")
ERROR_LOG = os.path.join(LOG_DIR, "errors.jsonl")

def read_query_logs() -> List[Dict]:
    logs = []
    if os.path.exists(QUERY_LOG):
        with open(QUERY_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return logs

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
