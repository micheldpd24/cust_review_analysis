"""
Configuration Module for Reviews Processing System

This module provides centralized configuration settings for the reviews scraping,
processing, and analysis pipeline. It includes file paths, HTTP settings, text
processing settings, and system parameters.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Company and URL settings
COMPANY_NAME = "backmarket"
BASE_URL = f"https://fr.trustpilot.com/review/www.{COMPANY_NAME}.fr"

# File path settings - using Path objects for better cross-platform compatibility
BASE_DIR = Path(".")
PARAMETERS_DIR = BASE_DIR / "parameters"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [PARAMETERS_DIR, DATA_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Parameter files
MAX_PAGE_PARAM_FILE = PARAMETERS_DIR / "max_page.txt"
LOG_FILE = LOGS_DIR / "reviews_pipeline.log"

# Data directories
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
FULL_REVIEWS_DIR = DATA_DIR / "full"
ARCHIVE_DIR = DATA_DIR / "archive"
MODELS_DIR = DATA_DIR / "models"

# Ensure data directories exist
for directory in [RAW_DATA_DIR, CLEANED_DATA_DIR, FULL_REVIEWS_DIR, ARCHIVE_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# HTTP Headers to simulate a browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}

# Requests settings
REQUEST_TIMEOUT = 10  # seconds
REQUEST_RETRY_COUNT = 3
REQUEST_RETRY_DELAY = 2  # seconds

# Scraping settings
MAX_PAGES_TO_SCRAPE = 10
DELAY_BETWEEN_REQUESTS = 1  # seconds

REVIEW_KEYS = [
    "id", "title", "review", "rating", "reply", 
    "experienceDate", "createdDateTime", "publishedDate", 
    "replyPublishedDate"
]

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": LOG_FILE,
            "formatter": "standard",
        },
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["file", "console"],
            "level": "INFO",
            "propagate": True,
        },
    },
}

# Stopwords for Text Cleaning
STOP_WORDS_TO_ADD: List[str] = [
    # Common French words to filter out
    "être", "leur", "leurs", "avoir", "cela", "les", "de", "pour", "des", "cette", "a",
    "j'ai", "car", "c'est", "chez", "tout", "fait", "chez", "donc", 
    "n'est", "si", "alors", "n'ai", "faire", "deux", "comme", "jour", "tr", "si", "ue",
    # Company-specific terms
    "back", "backmarket", "market"
]

# Text processing settings
MIN_REVIEW_LENGTH = 4  # Minimum number of characters in a cleaned review
TEXT_CLEANING_PATTERNS = [
    (r"#", ""),                            # Remove hashtags
    (r"&\w*;", ""),                        # Remove HTML entities
    (r"\$\w*", ""),                        # Remove dollar amounts
    (r"https?://[^\s/$.?#].[^\s]*", ""),   # Remove URLs
    (r"http(\S)+", ""),                    # Remove abbreviated URLs
    (r"http\s*\.\.\.", ""),                # Remove incomplete URLs
    (r"(RT|rt)\s*@\s*\S+", ""),            # Remove retweets
    (r"RT\s?@", ""),                       # Remove RT symbols
    (r"@\S+", ""),                         # Remove mentions
    (r"&", "and"),                         # Replace ampersands
    (r"\b\w{1,3}\b", " "),                 # Remove short words
]

# Sentiment analysis configuration
SENTIMENT_THRESHOLDS = {
    "positive": 4,  # Ratings >= 4 are positive
    "neutral": 3,   # Rating == 3 is neutral
    "negative": 2,  # Ratings <= 2 are negative
}

# ML dataset settings
ML_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "min_word_length": 2,
}

# Feature settings for ML
FEATURE_COLUMNS = [
    "review", "rating", "sentiment", "reviewLength", 
    "titleLength", "yearQuarter", "monthName"
]

# Export function to get configuration as dictionary
def get_config() -> Dict[str, Union[str, int, float, List, Dict, Path]]:
    """Return configuration settings as a dictionary."""
    # Gather all uppercase variables from this module
    config_dict = {key: value for key, value in globals().items() 
                  if key.isupper() and not key.startswith('_')}
    return config_dict

# Initialize configuration values from environment variables if available
def init_from_env() -> None:
    """Initialize config values from environment variables if available."""
    env_map = {
        "COMPANY_NAME": "REVIEWS_COMPANY_NAME",
        "MAX_PAGES_TO_SCRAPE": "REVIEWS_MAX_PAGES",
        "REQUEST_TIMEOUT": "REVIEWS_REQUEST_TIMEOUT",
    }
    
    for config_var, env_var in env_map.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            # Handle type conversion based on the existing value's type
            existing_value = globals().get(config_var)
            if existing_value is not None:
                if isinstance(existing_value, int):
                    globals()[config_var] = int(env_value)
                elif isinstance(existing_value, float):
                    globals()[config_var] = float(env_value)
                elif isinstance(existing_value, bool):
                    globals()[config_var] = env_value.lower() in ('true', 'yes', '1')
                else:
                    globals()[config_var] = env_value

# Initialize from environment variables
init_from_env()

# Update BASE_URL if COMPANY_NAME was changed from environment
if os.environ.get("REVIEWS_COMPANY_NAME"):
    BASE_URL = f"https://fr.trustpilot.com/review/www.{COMPANY_NAME}.fr"