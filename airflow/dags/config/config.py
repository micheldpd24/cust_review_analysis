import os
from pathlib import Path
from typing import List

# Constants and Parameters
COMPANY_NAME = "backmarket"
BASE_URL = f"https://fr.trustpilot.com/review/www.{COMPANY_NAME}.fr"

# File Paths
MAX_PAGE_PARAM_FILE = Path("./parameters/max_page.txt")
NEW_REVIEW_PARAM_FILE = Path("./parameters/new_reviews.txt")
# LOG_FILE = Path("./logs_etl/concatenation.log")

# Directories
RAW_DATA_DIR = Path("./data/raw")
CLEANED_DATA_DIR = Path("./data/cleaned")
FULL_REVIEWS_DIR = Path("./data/full")
ARCHIVE_DIR = Path("./data/archive")

# HTTP Headers
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

# Logging Configuration
# LOGGING_CONFIG = {
#     "level": "INFO",
#     "format": "%(asctime)s - %(levelname)s - %(message)s",
#     "handlers": [
#         {"type": "FileHandler", "filename": LOG_FILE},
#         {"type": "StreamHandler"},
#     ],
# }

# Stopwords for Text Cleaning
STOP_WORDS_TO_ADD: List[str] = [
    "être", "leur", "leurs", "avoir", "cela", "les", "de", "pour", "des", "cette", "a",
    "j'ai", "car", "c'est", "chez", "tout", "fait", "chez", "donc", 
    "n'est", "si", "alors", "n'ai", "faire", "deux", "comme", "jour", "tr", "si", "ue",
    "back", "backmarket", "market"
]
