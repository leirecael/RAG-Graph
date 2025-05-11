import json
import os
from datetime import datetime

LOG_DIR = "app/logs"
DATA_LOG = os.path.join(LOG_DIR, "data.jsonl")
ERROR_LOG = os.path.join(LOG_DIR, "errors.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)

def log_data(data: dict):
    with open(DATA_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def log_error(error_type: str, error_details: dict):
    data = {
        "timestamp": datetime.now().isoformat(),
        "error_type": error_type,
        "details": error_details
    }
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")