import logging
from pathlib import Path
from config.config import LOGGING_CONFIG, LOG_FILE

def setup_logging():
    log_dir = LOG_FILE.parent
    log_dir.mkdir(parents=True, exist_ok=True)  # Ensure logs directory exists
    if not logging.root.handlers:
        logging.basicConfig(
            level=LOGGING_CONFIG["level"],
            format=LOGGING_CONFIG["format"],
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler(),
            ],
        )