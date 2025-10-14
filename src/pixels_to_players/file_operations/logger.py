import os
from pathlib import Path
from datetime import datetime
import json
from .file_manager import FileManager

APP_NAME = "Pixels To Players"

class Logger:
    @staticmethod
    def log_event(data: dict):
        try:
            log_dir = Path(os.getenv("LOCALAPPDATA", ".")) / APP_NAME / "logs"
            log_file = log_dir / "last_invocation.json"

            json_data = json.dumps(data, indent=2, ensure_ascii=False)

            FileManager.write_to_file(log_file, json_data)
        except Exception as e:
            print(f"Failed to log event: {e}")

    @staticmethod
    def log_error(message: str):
        try:
            log_dir = Path(os.getenv("LOCALAPPDATA", ".")) / APP_NAME / "logs"
            error_log_file = log_dir / "errors.log"
            time_stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{time_stamp}] ERROR: {message}\n"

            FileManager.write_to_file(error_log_file, log_entry)
        except Exception as e:
            print(f"Failed to log error: {e}")