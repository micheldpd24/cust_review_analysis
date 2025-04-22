import os
import json
import re
import time
import logging
import emoji
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from spacy.util import is_package
from functools import lru_cache

# Add the parent directory to the path
import sys
sys.path.append("/opt/airflow/dags")

# Configuration imports
from config import get_config

# Constants
SUCCESS = 1
NO_UPDATES = 0
ERROR = None

# Load configuration
CONFIG = get_config()

# Ensure Spacy model is installed and loaded
def ensure_model(model_name: str = CONFIG["spacy_model"]):
    """Ensure the specified Spacy model is installed."""
    if not is_package(model_name):
        logging.info(f"Model '{model_name}' is not installed. Downloading now...")
        spacy.cli.download(model_name)
    else:
        logging.info(f"Model '{model_name}' is already installed.")

def setup_spacy(model_name: str = CONFIG["spacy_model"]):
    """Load and return the Spacy model."""
    ensure_model(model_name)
    try:
        nlp = spacy.load(model_name)
        logging.info(f"Loaded SpaCy model '{model_name}' successfully.")
        return nlp
    except Exception as e:
        logging.error(f"Failed to load SpaCy model '{model_name}': {e}")
        raise

# Initialize Spacy model
nlp = setup_spacy()

# Cache stopwords for efficiency
@lru_cache(maxsize=1)
def get_stop_words():
    """Load and cache stopwords for text processing."""
    return set(stopwords.words("french")).union(set(CONFIG["stop_words_to_add"]))

class ReviewsManager:
    """Main class to manage the review processing pipeline."""
    def __init__(self):
        """Initialize the ReviewsManager with configuration."""
        self.config = CONFIG
        self.base_url = self.config["base_url"].format(company_name=self.config["company_name"])
        self.stop_words = get_stop_words()

    def get_end_page(self) -> Optional[int]:
        """
        Get the end page parameter from the configuration file.
        Returns:
            int: The page number to end scraping at, or None if not found.
        """
        file_path = Path(self.config["files"]["max_page_param_file"])
        try:
            with open(file_path, "r") as file:
                first_line = file.readline().strip()
                if first_line.isdigit():
                    return int(first_line)
                logging.error("The first line does not contain a valid integer.")
                return None
        except FileNotFoundError:
            logging.error(f"File '{file_path}' was not found.")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None

    def update_max_page_param(self, value: int) -> bool:
        """
        Update the max page parameter file with a new value.
        Args:
            value: The new page value to write.
        Returns:
            bool: True if successful, False otherwise.
        """
        file_path = Path(self.config["files"]["max_page_param_file"])
        try:
            with open(file_path, "w") as f:
                f.write(str(value))
            logging.info(f"Updated max page value to '{value}' in {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error updating max page parameter: {e}")
            return False

    def extract_reviews(self) -> Optional[str]:
        """
        Scrape reviews for a company and save raw data to CSV.
        Returns:
            str: Path to saved CSV file, or None if no reviews collected.
        """
        headers = self.config["http_headers"]
        base_url = self.config["base_url"].format(company_name=self.config["company_name"])
        keys = self.config["review_keys"]

        # Determine page range
        end_page = self.get_end_page()
        if not end_page:
            logging.error("Failed to get end page parameter")
            return None
        end_page = max(10, end_page)
        start_page = max(1, end_page - 10)

        reviews_list = []
        for page in range(end_page, start_page - 1, -1):
            logging.info(f"Processing page {page}")
            url_page = f"{base_url}?page={page}"
            response = self._fetch_page(url_page, headers)
            if not response:
                continue
            page_reviews = self._extract_page_reviews(response)
            if page_reviews:
                reviews_list.extend(page_reviews)
            time.sleep(1)

        if not reviews_list:
            logging.warning("No reviews collected.")
            return None

        # Save raw reviews to CSV
        output_path = self._save_raw_reviews(reviews_list, start_page, end_page)
        flag_value = start_page if start_page >= 10 else 10
        self.update_max_page_param(flag_value)
        logging.info(f"Finished processing pages {start_page}-{end_page}")
        return output_path

    def _fetch_page(self, url: str, headers: Dict[str, str]) -> Optional[requests.Response]:
        """
        Fetch a page from the target website.
        Args:
            url: URL to fetch.
            headers: HTTP headers to use.
        Returns:
            Response object or None if failed.
        """
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logging.error(f"Error accessing page {url}: {e}")
            return None

    def _extract_page_reviews(self, response: requests.Response) -> List[Dict[str, Any]]:
        """
        Extract reviews from a page response.
        Args:
            response: HTTP response object.
        Returns:
            List of review dictionaries.
        """
        keys = self.config["review_keys"]
        reviews = []
        try:
            soup = BeautifulSoup(response.content, "html.parser")
            script_content = soup.body.script.contents if soup.body and soup.body.script else None
            if not script_content:
                logging.warning("No data found in page")
                return reviews
            raw_data = json.loads(script_content[0])
            raw_data = raw_data.get("props", {}).get("pageProps", {}).get("reviews", [])
            for review in raw_data:
                review_data = {
                    "id": review.get("id"),
                    "title": review.get("title"),
                    "review": review.get("text"),
                    "rating": review.get("rating"),
                    "experienceDate": review.get("dates", {}).get("experiencedDate"),
                    "createdDateTime": review.get("labels", {}).get("verification", {}).get("createdDateTime"),
                    "publishedDate": review.get("dates", {}).get("publishedDate"),
                }
                reply_data = review.get("reply", {})
                if reply_data:
                    review_data["reply"] = reply_data.get("message")
                    review_data["replyPublishedDate"] = reply_data.get("publishedDate")
                else:
                    review_data["reply"] = None
                    review_data["replyPublishedDate"] = None
                reviews.append({key: review_data.get(key) for key in keys})
        except Exception as e:
            logging.error(f"Error processing page: {e}")
        return reviews

    def _save_raw_reviews(self, reviews: List[Dict[str, Any]], start_page: int, end_page: int) -> str:
        """
        Save raw reviews to CSV file.
        Args:
            reviews: List of review dictionaries.
            start_page: First page number in the range.
            end_page: Last page number in the range.
        Returns:
            str: Path to the saved file.
        """
        df_raw_reviews = pd.DataFrame(reviews)
        output_dir = Path(self.config["paths"]["raw_data_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_file_path = output_dir / f"raw_reviews_{start_page+1}-{end_page}.csv"
        df_raw_reviews.to_csv(raw_file_path, index=False)
        logging.info(f"Saved {len(reviews)} reviews to {raw_file_path}")
        return str(raw_file_path)

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters, URLs, stopwords, etc.
        Args:
            text: Raw text to clean.
        Returns:
            str: Cleaned text.
        """
        if not isinstance(text, str) or not text:
            return ""
        text = re.sub(r"#", "", text)
        text = re.sub(r"&\w*;", "", text)
        text = re.sub(r"\$\w*", "", text)
        text = re.sub(r"https?://[^\s/$.?#].[^\s]*", "", text)
        text = re.sub(r"http(\S)+", "", text)
        text = re.sub(r"http\s*\.\.\.", "", text)
        text = re.sub(r"(RT|rt)\s*@\s*\S+", "", text)
        text = re.sub(r"RT\s?@", "", text)
        text = re.sub(r"@\S+", "", text)
        text = re.sub(r"&", "and", text)
        text = re.sub(r"\b\w{1,3}\b", " ", text)
        text = "".join(c for c in text if ord(c) <= 0xFFFF)
        text = text.strip()
        text = emoji.demojize(text)
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = word_tokenize(text.lower(), language="french")
        tokens_alpha = [token for token in tokens if token.isalpha()]
        tokens_cleaned = [token for token in tokens_alpha if token not in self.stop_words]
        cleaned_text = " ".join(tokens_cleaned)
        return cleaned_text

    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatizes a given French text using SpaCy's French language model.
        Args:
            text: The input French text to be lemmatized.
        Returns:
            str: A string containing the lemmatized version of the input text.
        """
        doc = nlp(text)
        lemmatized_text = " ".join([token.lemma_ for token in doc])
        return lemmatized_text

    def process_reviews(self, raw_file: str) -> Optional[str]:
        """
        Process raw reviews by cleaning and enriching the data.
        Args:
            raw_file: Path to raw reviews CSV file.
        Returns:
            str: Path to cleaned reviews CSV file, or None if processing fails.
        """
        try:
            df = pd.read_csv(raw_file)
            logging.info(f"Loaded {len(df)} rows of raw review data")
            df = self._standardize_dates(df)
            df = self._extract_temporal_features(df)
            df = self._add_derived_columns(df)
            df = self._clean_filter_data(df)
            output_dir = Path(self.config["paths"]["cleaned_data_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            cleaned_file_path = output_dir / f"cleaned_reviews_{timestamp}.csv"
            df.to_csv(cleaned_file_path, index=False)
            logging.info(f"Cleaned data saved to: {cleaned_file_path}")
            return str(cleaned_file_path)
        except Exception as e:
            logging.error(f"Error in review processing pipeline: {e}")
            return None

    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert date columns to datetime format."""
        date_columns = ["experienceDate", "createdDateTime", "publishedDate"]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
                df[col] = df[col].dt.tz_localize(None)
            except Exception as e:
                logging.warning(f"Error converting {col} to datetime: {e}")
        try:
            df["reviewExperienceDelay"] = (df["createdDateTime"] - df["experienceDate"]).dt.total_seconds() / 60
        except Exception as e:
            logging.warning(f"Error calculating review delay: {e}")
            df["reviewExperienceDelay"] = None
        try:
            df["replyPublishedDate"] = pd.to_datetime(df["replyPublishedDate"])
        except Exception:
            df["replyPublishedDate"] = None
        logging.info("Date formats standardized")
        return df

    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract year, month, day features from dates."""
        try:
            df["date"] = pd.to_datetime(df["createdDateTime"]).dt.date
            df["year"] = pd.to_datetime(df["createdDateTime"]).dt.year
            df["yearQuarter"] = (
                pd.to_datetime(df["createdDateTime"]).dt.year.astype(str)
                + "-Q"
                + pd.to_datetime(df["createdDateTime"]).dt.quarter.astype(str)
            )
            df["yearMonth"] = pd.to_datetime(df["createdDateTime"]).dt.strftime("%Y-%m")
            df["month"] = pd.to_datetime(df["createdDateTime"]).dt.month
            df["monthName"] = pd.to_datetime(df["createdDateTime"]).dt.month_name()
            df["day"] = pd.to_datetime(df["createdDateTime"]).dt.day
            df["dayName"] = pd.to_datetime(df["createdDateTime"]).dt.day_name()
            df["hour"] = pd.to_datetime(df["createdDateTime"]).dt.hour
            reply_date_mask = pd.notna(df["replyPublishedDate"])
            if reply_date_mask.any():
                reply_dates = pd.to_datetime(df.loc[reply_date_mask, "replyPublishedDate"])
                df.loc[reply_date_mask, "replyYear"] = reply_dates.dt.year
                df.loc[reply_date_mask, "replyMonth"] = reply_dates.dt.month
                df.loc[reply_date_mask, "replyDay"] = reply_dates.dt.day
                df.loc[reply_date_mask, "replyHour"] = reply_dates.dt.hour
            else:
                for col in ["replyYear", "replyMonth", "replyDay", "replyHour"]:
                    df[col] = None
            logging.info("Temporal features extracted")
        except Exception as e:
            logging.error(f"Error extracting temporal features: {e}")
        return df

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns such as length and sentiment."""
        text_columns = ["review", "title"]
        for col in text_columns:
            try:
                df[f"{col}Length"] = df[col].str.len()
            except Exception as e:
                logging.warning(f"Error calculating {col} length: {e}")
                df[f"{col}Length"] = 0
        try:
            df["sentiment"] = df["rating"].apply(
                lambda x: "positive" if x >= self.config["sentiment_thresholds"]["positive"]
                else ("neutral" if x == self.config["sentiment_thresholds"]["neutral"] else "negative")
            )
            logging.info("Added derived columns")
        except Exception as e:
            logging.error(f"Error adding sentiment column: {e}")
        return df

    def _clean_filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text and filter out invalid or incomplete records."""
        initial_rows = len(df)
        required_fields = ["id", "review", "rating", "experienceDate", "createdDateTime", "publishedDate"]
        df.dropna(subset=required_fields, inplace=True)
        logging.info(f"Removed {initial_rows - len(df)} rows with missing values")
        valid_ratings = [1, 2, 3, 4, 5]
        df = df[df["rating"].isin(valid_ratings)]
        logging.info(f"Removed {initial_rows - len(df)} rows with invalid ratings")
        df["review_clean"] = df["review"].apply(self.clean_text)
        initial_rows = len(df)
        df = df[df["review"].str.len() > self.config["min_review_length"]]
        logging.info(f"Removed {initial_rows - len(df)} short reviews")
        df["review_lemmatized"] = df["review_clean"].apply(self.lemmatize_text)
        return df

    def load_reviews(self, cleaned_file: str) -> int:
        """
        Load and update the full reviews database from a cleaned CSV file.
        Args:
            cleaned_file: Path to the cleaned reviews CSV.
        Returns:
            int: SUCCESS, NO_UPDATES, or ERROR.
        """
        try:
            full_reviews_dir = Path(self.config["paths"]["full_reviews_dir"])
            archive_dir = Path(self.config["paths"]["archive_dir"])
            full_reviews_dir.mkdir(parents=True, exist_ok=True)
            archive_dir.mkdir(parents=True, exist_ok=True)
            df = pd.read_csv(cleaned_file)
            if df.empty:
                logging.warning("Input DataFrame is empty")
                return NO_UPDATES
            full_reviews_path = full_reviews_dir / "full_reviews.csv"
            pos_reviews_path = full_reviews_dir / "positive_reviews.csv"
            neu_reviews_path = full_reviews_dir / "neutral_reviews.csv"
            neg_reviews_path = full_reviews_dir / "negative_reviews.csv"
            df_full = pd.DataFrame()
            if full_reviews_path.exists():
                try:
                    df_full = pd.read_csv(full_reviews_path, low_memory=False)
                    logging.info(f"Loaded {len(df_full)} existing records")
                except pd.errors.EmptyDataError:
                    logging.warning(f"Empty CSV file found at {full_reviews_path}")
                except pd.errors.ParserError:
                    logging.error(f"Parsing error in {full_reviews_path}")
            else:
                logging.info(f"No existing file found, initializing empty DataFrame")
            initial_length = len(df_full)
            df_full_updated = pd.concat([df_full, df], ignore_index=True)
            df_full_updated = df_full_updated.drop_duplicates(subset=["id", "title"], keep="last")
            final_length = len(df_full_updated)
            new_records_added = final_length - initial_length
            logging.info(f"Original records: {initial_length}")
            logging.info(f"Updated records: {final_length}")
            logging.info(f"New records added: {new_records_added}")
            if new_records_added > 0:
                percentage_increase = 100 if initial_length == 0 else ((final_length - initial_length) / initial_length * 100)
                logging.info(f"Percentage increase in data: {percentage_increase:.2f}%")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = full_reviews_dir / f"full_reviews_backup_{timestamp}.csv"
                if not df_full.empty:
                    df_full.to_csv(backup_path, index=False)
                    logging.info(f"Backup created at {backup_path}")
                df_full_updated.to_csv(full_reviews_path, index=False)
                logging.info(f"Updated data saved to {full_reviews_path}")
                df_full_updated[df_full_updated["sentiment"] == "positive"].to_csv(pos_reviews_path, index=False)
                df_full_updated[df_full_updated["sentiment"] == "neutral"].to_csv(neu_reviews_path, index=False)
                df_full_updated[df_full_updated["sentiment"] == "negative"].to_csv(neg_reviews_path, index=False)
                logging.info(f"Positive, neutral, and negative reviews saved to {pos_reviews_path}, {neu_reviews_path}, and {neg_reviews_path}")
                return SUCCESS
            else:
                logging.info("No new reviews to process")
                return NO_UPDATES
        except Exception as e:
            logging.error(f"Error loading reviews: {e}")
            return ERROR