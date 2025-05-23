import json
import os
from datetime import datetime

#Paths for logging
LOG_DIR = "app/logs" 
DATA_LOG = os.path.join(LOG_DIR, "data.jsonl") #File storing data logs
ERROR_LOG = os.path.join(LOG_DIR, "errors.jsonl") #File storing error logs

def log_data(data: dict) -> None:
    """
    Appends a structured data entry to the data log file (data.jsonl).

    Args:
        data (dict): A dictionary containing data to log.
    """
    if not isinstance(data, dict):
        raise TypeError("log_data expects a dictionary")
    with open(DATA_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def log_error(error_type: str, error_details: dict) -> None:
    """
    Appends a structured error entry to the error log file (errors.jsonl).

    Args:
        error_type (str): An identifier for the type of error (e.g., "ValidationError").
        error_details (dict): A dictionary containing error data.
    """
    if not isinstance(error_details, dict):
        raise TypeError("log_error expects a dictionary")
    data = {
        "timestamp": datetime.now().isoformat(),
        "error_type": error_type,
        "details": error_details
    }
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")