import json
import os
from typing import List, Dict


class LogReader:

    def __init__(self):
        pass

    #Paths for logging
    LOG_DIR = "app/logs"
    DATA_LOG = os.path.join(LOG_DIR, "data.jsonl") #File storing data logs
    ERROR_LOG = os.path.join(LOG_DIR, "errors.jsonl") #File storing error logs

    def read_data_logs(self) -> List[Dict]:
        """
        Reads and parses each line of the data log file as a JSON object.

        Returns:
            List[Dict]: A list of JSON log entries from data.jsonl.
        """
        logs = []
        if os.path.exists(self.DATA_LOG):
            with open(self.DATA_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        logs.append(json.loads(line)) # Parse each data log entry
                    except json.JSONDecodeError:
                        continue #Skip invalid lines
        return logs

    def read_error_logs(self) -> List[Dict]:
        """
        Reads and parses each line of the error log file as a JSON object.

        Returns:
            List[Dict]: A list of error log entries from errors.jsonl.
        """
        logs = []
        if os.path.exists(self.ERROR_LOG):
            with open(self.ERROR_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        logs.append(json.loads(line)) # Parse each error log entry
                    except json.JSONDecodeError:
                        continue #Skip invalid lines
        return logs