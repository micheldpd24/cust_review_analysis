import os
import json
import requests
import pandas as pd
import pytz
from bs4 import BeautifulSoup
from datetime import datetime
import time
import shutil
import re
from typing import List, Optional
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from pathlib import Path
import sys
sys.path.append("/opt/airflow/dags")

import logging

from config.config import (
    COMPANY_NAME,
    BASE_URL,
    MAX_PAGE_PARAM_FILE,
    NEW_REVIEW_PARAM_FILE,
    RAW_DATA_DIR,
    CLEANED_DATA_DIR,
    FULL_REVIEWS_DIR,
    ARCHIVE_DIR,
    HEADERS,
    STOP_WORDS_TO_ADD,
)
from logging_config import setup_logging



# Initialize logging
setup_logging()

# Preload stopwords outside the function to avoid repeated loading
STOP_WORDS = set(stopwords.words("french")).union(set(STOP_WORDS_TO_ADD))


# Function to check new_review parameter
def check_new_reviews(file_path=NEW_REVIEW_PARAM_FILE):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            first_line = file.readline().strip()
            return int(first_line) if first_line else None
    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None


# Function to get the end page to collect reviews from
def get_end_page(file_path=MAX_PAGE_PARAM_FILE):
    try:
        with open(file_path, "r") as file:
            first_line = file.readline().strip()
            if first_line.isdigit():
                return int(first_line)
            logging.error("Error: The first line does not contain a valid integer.")
            return None
    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None


# Function to scrape reviews
def extract_reviews(company_name=COMPANY_NAME):
    headers = HEADERS
    base_url = BASE_URL
    keys = [
        "id",
        "title",
        "review",
        "rating",
        "reply",
        "experienceDate",
        "createdDateTime",
        "publishedDate",
        "replyPublishedDate",
    ]
    end_page = get_end_page()
    end_page = max(10, end_page)
    start_page = max(1, end_page - 10)
    reviews_list = []
    counter = 0

    for page in range(end_page, start_page, -1):
        counter += 1
        logging.info(f"Processing page {page}")
        url_page = f"{base_url}?page={page}"
        try:
            response = requests.get(url_page, headers=headers, timeout=5)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Error accessing page {page}: {e}")
            continue

        try:
            soup = BeautifulSoup(response.content, "html.parser")
            script_content = soup.body.script.contents if soup.body and soup.body.script else None
            if not script_content:
                logging.warning(f"No data found in page {page}")
                continue
            raw_data = json.loads(script_content[0])
            raw_data = raw_data.get("props", {}).get("pageProps", {}).get("reviews", [])
            for review in raw_data:
                tmp = {}
                tmp["id"] = review.get("id")
                tmp["title"] = review.get("title")
                tmp["review"] = review.get("text")
                tmp["rating"] = review.get("rating")
                try:
                    tmp["reply"] = review.get("reply", {}).get("message")
                    tmp["replyPublishedDate"] = review.get("reply", {}).get("publishedDate")
                except:
                    tmp["reply"] = None
                    tmp["replyPublishedDate"] = None
                tmp["experienceDate"] = review.get("dates", {}).get("experiencedDate")
                tmp["createdDateTime"] = review.get("labels", {}).get("verification", {}).get("createdDateTime")
                tmp["publishedDate"] = review.get("dates", {}).get("publishedDate")
                reviews_list.append({key: tmp.get(key) for key in keys})
        except Exception as e:
            logging.error(f"Error processing page {page}: {e}")
            continue

    if not reviews_list:
        logging.warning("No reviews collected.")
        return

    df_raw_reviews = pd.DataFrame(reviews_list)
    output_dir = RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_file_path = output_dir / f"raw_reviews_{start_page+1}-{end_page}.csv"
    df_raw_reviews.to_csv(raw_file_path, index=False)
    logging.info(f"Saved reviews data to {raw_file_path}")

    flag_value = start_page if start_page >= 10 else 10
    with open(MAX_PAGE_PARAM_FILE, "w") as f:
        f.write(str(flag_value))
    logging.info(f"Wrote new max page value '{flag_value}' to {MAX_PAGE_PARAM_FILE}")
    logging.info(f"Finished processing pages {start_page}-{end_page}")
    return str(raw_file_path)


# Function to clean text
def clean_text(text: str) -> str:
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
    tokens_cleaned = [token for token in tokens_alpha if token not in STOP_WORDS]
    cleaned_text = " ".join(tokens_cleaned)
    return cleaned_text

# function to standardize date formats
def standardize_date_formats(df):
    df["experienceDate"] = pd.to_datetime(df["experienceDate"])
    df["experienceDate"] = df["experienceDate"].dt.tz_localize(None)
    df["createdDateTime"] = pd.to_datetime(df["createdDateTime"])
    df["createdDateTime"] = df["createdDateTime"].dt.tz_localize(None)
    df["publishedDate"] = pd.to_datetime(df["publishedDate"])
    df["publishedDate"] = df["publishedDate"].dt.tz_localize(None)
    df["reviewExperienceDelay"] = (df["createdDateTime"] - df["experienceDate"]).dt.total_seconds() / 60

    try:
        df["replyPublishedDate"] = pd.to_datetime(df["replyPublishedDate"])
        df["replyPublishedDate"] = df["replyPublishedDate"].dt.tz_localize(None)
    except:
        df["replyPublishedDate"] = None

# Function to process reviews
def process_reviews(raw_file):
    try:
        df = pd.read_csv(raw_file)
        logging.info(f"Successfully loaded {len(df)} rows of data.")
    except Exception as e:
        logging.error(f"Error loading raw data: {e}")
        return None

    try:
        df["experienceDate"] = pd.to_datetime(df["experienceDate"])
        df["experienceDate"] = df["experienceDate"].dt.tz_localize(None)
        df["createdDateTime"] = pd.to_datetime(df["createdDateTime"])
        df["createdDateTime"] = df["createdDateTime"].dt.tz_localize(None)
        df["publishedDate"] = pd.to_datetime(df["publishedDate"])
        df["publishedDate"] = df["publishedDate"].dt.tz_localize(None)
        df["reviewExperienceDelay"] = (df["createdDateTime"] - df["experienceDate"]).dt.total_seconds() / 60
        try:
            df["replyPublishedDate"] = pd.to_datetime(df["replyPublishedDate"])
        except:
            df["replyPublishedDate"] = None
        logging.info("Date formats standardized successfully.")
    except Exception as e:
        logging.error(f"Error standardizing date formats: {e}")
        return None

    try:
        df["date"] = pd.to_datetime(df["createdDateTime"]).dt.date
        df["year"] = pd.to_datetime(df["createdDateTime"]).dt.year
        df["yearQuarter"] = (
            pd.to_datetime(df["createdDateTime"]).dt.year.astype(str)
            + "-Q"
            + pd.to_datetime(df["createdDateTime"]).dt.quarter.astype(str)
        )
        df["yearMonth"] = pd.to_datetime(df["createdDateTime"]).dt.strftime("%Y" +"-"+ "%m")

        df["month"] = pd.to_datetime(df["createdDateTime"]).dt.month
        df["monthName"] = pd.to_datetime(df["createdDateTime"]).dt.month_name()
        df["day"] = pd.to_datetime(df["createdDateTime"]).dt.day
        df["dayName"] = pd.to_datetime(df["createdDateTime"]).dt.day_name()
        df["hour"] = pd.to_datetime(df["createdDateTime"]).dt.hour
        try:
            df["replyYear"] = pd.to_datetime(df["replyPublishedDate"]).dt.year
            df["replyMonth"] = pd.to_datetime(df["replyPublishedDate"]).dt.month
            df["replyDay"] = pd.to_datetime(df["replyPublishedDate"]).dt.day
            df["replyHour"] = pd.to_datetime(df["replyPublishedDate"]).dt.hour
        except:
            df["replyYear"] = None
            df["replyMonth"] = None
            df["replyDay"] = None
            df["replyHour"] = None
        logging.info("Temporal features extracted successfully.")
    except Exception as e:
        logging.error(f"Error extracting temporal features: {e}")
        return None
    
    # Add columns for review length and number of words
    try:
        df["reviewLength"] = df["review"].str.len()
        df["titleLength"] = df["title"].str.len()
    except Exception as e:
        logging.error(f"Error adding review length column: {e}")
        return None

    try:
        initial_rows = len(df)
        df.dropna(
            inplace=True,
            subset=["id", "review", "rating", "experienceDate", "createdDateTime", "publishedDate"],
        )
        removed_rows = initial_rows - len(df)
        logging.info(f"Removed {removed_rows} rows with missing values. Remaining rows: {len(df)}")
    except Exception as e:
        logging.error(f"Error removing rows with missing values: {e}")
        return None

    try:
        initial_rows = len(df)
        df = df[df["rating"].isin([1, 2, 3, 4, 5])]
        removed_rows = initial_rows - len(df)
        logging.info(f"Removed {removed_rows} rows with invalid ratings. Remaining rows: {len(df)}")
    except Exception as e:
        logging.error(f"Error removing rows with invalid ratings: {e}")
        return None

    try:
        df["sentiment"] = df["rating"].apply(lambda x: "positive" if x >= 4 else ("neutral" if x == 3 else "negative"))
        logging.info("Sentiment column added successfully.")
    except Exception as e:
        logging.error(f"Error adding sentiment column: {e}")
        return None

    try:
        df["review"] = df["review"].apply(clean_text)
        logging.info("Review text cleaned successfully.")
    except Exception as e:
        logging.error(f"Error cleaning review text: {e}")
        return None

    try:
        initial_rows = len(df)
        df = df[df["review"].str.len() > 4]
        removed_rows = initial_rows - len(df)
        logging.info(f"Removed {removed_rows} short reviews. Remaining rows: {len(df)}")
    except Exception as e:
        logging.error(f"Error removing short reviews: {e}")
        return None

    try:
        output_dir = CLEANED_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        cleaned_file_path = output_dir / f"cleaned_reviews_{timestamp}.csv"
        df.to_csv(cleaned_file_path, index=False)
        logging.info(f"Cleaned data saved to: {cleaned_file_path}")
    except Exception as e:
        logging.error(f"Error saving cleaned data: {e}")
        return None

    return str(cleaned_file_path)


# Function to load cleaned reviews into the full database
def load_reviews(cleaned_file):
    try:
        df = pd.read_csv(cleaned_file)
        # base_dir = Path("./data")
        full_reviews_folder = FULL_REVIEWS_DIR  # base_dir / "full"
        archive_folder = ARCHIVE_DIR  # base_dir / "archive"
        full_reviews_file = "full_reviews.csv"
        full_reviews_path = FULL_REVIEWS_DIR / full_reviews_file
        new_reviews_flag_path = NEW_REVIEW_PARAM_FILE

        full_reviews_folder.mkdir(parents=True, exist_ok=True)
        archive_folder.mkdir(parents=True, exist_ok=True)
        new_reviews_flag_path.parent.mkdir(parents=True, exist_ok=True)

        df_full = pd.DataFrame()
        if full_reviews_path.exists():
            try:
                df_full = pd.read_csv(full_reviews_path, low_memory=False)
                logging.info(f"Loaded existing data from {full_reviews_path}")
            except pd.errors.EmptyDataError:
                logging.warning(f"Empty CSV file found at {full_reviews_path}")
            except pd.errors.ParserError:
                logging.error(f"Parsing error in {full_reviews_path}")
        else:
            logging.info(f"No existing file found at {full_reviews_path}, initializing empty DataFrame")

        if df.empty:
            logging.warning("Input DataFrame is empty")
            return None

        initial_length = len(df_full)
        df_full_updated = pd.concat([df_full, df], ignore_index=True)
        df_full_updated = df_full_updated.drop_duplicates(subset=df.columns, keep="last")
        final_length = len(df_full_updated)

        logging.info(f"Original records: {initial_length}")
        logging.info(f"Updated records: {final_length}")
        new_records_added = final_length - initial_length
        logging.info(f"New records added: {new_records_added}")

        has_new_reviews = final_length > initial_length
        flag_value = "1" if has_new_reviews else "0"
        with open(new_reviews_flag_path, "w") as f:
            f.write(flag_value)
        logging.info(f"Wrote '{flag_value}' to {new_reviews_flag_path}")

        if not has_new_reviews:
            logging.info("No new reviews to process. Exiting without updates.")
            return df_full_updated

        percentage_increase = (
            100 if initial_length == 0 and final_length > 0 else ((final_length - initial_length) / initial_length * 100)
        )
        logging.info(f"Percentage increase in data: {percentage_increase:.2f}%")

        if full_reviews_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = full_reviews_folder / f"full_reviews_backup_{timestamp}.csv"
            df_full.to_csv(backup_path, index=False)
            logging.info(f"Backup created at {backup_path}")

        df_full_updated.to_csv(full_reviews_path, index=False)
        logging.info(f"Updated data saved to {full_reviews_path}")

        try:
            archive_path = archive_folder / cleaned_file.name
            shutil.move(cleaned_file, archive_path)
            logging.info(f"Archived cleaned review file to: {archive_path}")
        except Exception as e:
            logging.error(f"Error archiving cleaned review file: {e}")

        return str(full_reviews_path)  # Return the path to the updated DataFrame
    except Exception as e:
        logging.error(f"Error processing reviews: {str(e)}")
        return None