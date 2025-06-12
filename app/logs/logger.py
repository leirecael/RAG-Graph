import json
import os
from datetime import datetime

class Logger:
    """
    Logs both application data and errors in JSON Lines format.

    This class writes logs about data and errors that happen in the system. 
    The data can be used to analyze different types of metrics.

    Attributes:
        LOG_DIR (str): Directory where log files are stored.
        DATA_LOG (str): Path to the data log file (`data.jsonl`).
        ERROR_LOG (str): Path to the error log file (`errors.jsonl`).

    Methods:
        log_data(data): Logs general data.
        log_error(error_type, error_details): Logs errors.
    """
    
    def __init__(self):
        pass

    #Paths for logging
    LOG_DIR = "app/logs" 
    DATA_LOG = os.path.join(LOG_DIR, "data.jsonl") #File storing data logs
    ERROR_LOG = os.path.join(LOG_DIR, "errors.jsonl") #File storing error logs

    def log_data(self, data: dict) -> None:
        """
        Appends a structured data entry to the data log file (data.jsonl).

        Args:
            data (dict): A dictionary containing data to log.
        """
        if not isinstance(data, dict):
            raise TypeError("log_data expects a dictionary")
        with open(self.DATA_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def log_error(self, error_type: str, error_details: dict) -> None:
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
        with open(self.ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")