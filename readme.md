# Trustpilot Reviews Analysis Dashboard

This repository contains an end-to-end pipeline to scrape, process, and analyze customer reviews from **Trustpilot** for a given company. The processed data is visualized in an interactive dashboard built with **Dash**, which dynamically updates every 50 seconds if new data is detected in the `full_reviews.csv` file.

The pipeline is orchestrated using **Apache Airflow** running in a Docker container, and the Dash dashboard also runs in its own Docker container.

---

## Table of Contents

1. [Overview](#overview)
2. [Dashboard Features](#dashboard-features)
3. [Pipeline Workflow](#pipeline-workflow)
4. [Installation and Setup](#installation-and-setup)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

---

## Overview

The project automates the collection and analysis of customer reviews from Trustpilot. It performs the following tasks:

1. **Scraping**: Extracts review data (text, sentiment, rating, date, etc.) from Trustpilot.
2. **Processing**: Cleans and processes the scraped data to derive insights such as sentiment analysis and temporal features.
3. **Loading**: Saves the processed data into a CSV file (`full_reviews.csv`).
4. **Visualization**: Provides a Dash-based dashboard that dynamically updates every 50 seconds if new data is available.

The pipeline is scheduled using **Apache Airflow**, ensuring regular updates to the dataset and dashboard.

---

## Dashboard Features

The dashboard provides the following features:

### **1. Filters**
- Filter reviews by year, quarter, and month.

### **2. Overall Overview Tab**
- **Sentiment Distribution**: Pie chart showing the distribution of positive, neutral, and negative sentiments.
- **Rating Distribution**: Bar chart displaying the distribution of ratings (1–5 stars).
- **Reply Rate Trend**: Percentage of reviews with replies over time.
- **Yearly Review Volume and Average Rating**: Table summarizing yearly review volume and average ratings.

### **3. Trends Tab**
- **Review Volume Trends**: Line chart showing review volume trends (daily, monthly, quarterly, yearly).
- **Sentiment Trends**: Sentiment trends over time.
- **Average Rating Trends**: Average rating trends over time.

### **4. Textual Analysis Tab**
- **Word Cloud**: Visualization of the most frequent words for the selected sentiment (positive, neutral, negative).
- **Top N-Grams**: Table displaying the top 10 trigrams (groups of three consecutive words) for the selected sentiment.

### **5. Export Functionality**
- Export filtered data as a CSV file for offline analysis.

---

## Pipeline Workflow

The pipeline is orchestrated using **Apache Airflow** and consists of the following steps:

1. **Extract Reviews**:
   - Scrapes customer reviews from Trustpilot.
   - Outputs raw review data.

2. **Process Reviews**:
   - Cleans and transforms raw review data.
   - Adds derived features such as sentiment labels, year, quarter, and month.

3. **Load Reviews**:
   - Saves the processed data into `full_reviews.csv`.

4. **Check New Data**:
   - Verifies if new reviews were added and triggers updates accordingly.

5. **Dynamic Dashboard Updates**:
   - The dashboard checks `full_reviews.csv` every 50 seconds for updates.
   - If new data is detected, the dashboard reloads and refreshes the visualizations.

---

## Installation and Setup

### Prerequisites
- **Docker** and **Docker Compose** installed on your system.
- Python 3.9+ (for local development or custom scripts).

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/trustpilot-reviews-analysis.git
   cd trustpilot-reviews-analysis
   ```

2. **Run the `setup.sh` Script**:
   - Ensure the `setup.sh` script has execute permissions:
     ```bash
     chmod +x setup.sh
     ```
   - Execute the script to initialize the `data` and `parameters` folders:
     ```bash
     ./setup.sh
     ```

3. **Build and Launch Docker Containers**:
   - Use the provided `docker-compose.yml` file to launch Airflow, the Dash dashboard, and other dependencies:
     ```bash
     docker-compose up --build
     ```
   - This will start the following services:
     - **Airflow Webserver**: Accessible at `http://localhost:8080`.
     - **Airflow Scheduler**: Handles DAG execution.
     - **PostgreSQL**: Database for Airflow metadata.
     - **Redis**: Message broker for CeleryExecutor.
     - **Dash Dashboard**: Accessible at `http://localhost:8050`.

4. **Access the Dashboard**:
   - Once the containers are running, open the Dash app in your browser:
     ```bash
     http://localhost:8050
     ```

5. **Access Airflow**:
   - Open the Airflow web interface in your browser:
     ```bash
     http://localhost:8080
     ```
   - Default credentials:
     - Username: `airflow`
     - Password: `airflow`

---

## Usage

1. **Trigger the Pipeline**:
   - Trigger the Airflow DAG manually or wait for the scheduled run.

2. **View the Dashboard**:
   - Open the Dash app in your browser to explore the latest reviews and insights.

3. **Export Data**:
   - Use the export button on the dashboard to download filtered data for further analysis.

---

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to reach out with any questions or suggestions! 🚀