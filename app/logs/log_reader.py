import json
import os
from typing import List, Dict

LOG_DIR = "app/logs"
DATA_LOG = os.path.join(LOG_DIR, "data.jsonl")
ERROR_LOG = os.path.join(LOG_DIR, "errors.jsonl")

def read_data_logs() -> List[Dict]:
    logs = []
    if os.path.exists(DATA_LOG):
        with open(DATA_LOG, "r", encoding="utf-8") as f:
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