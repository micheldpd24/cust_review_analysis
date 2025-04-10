FROM apache/airflow:latest-python3.11

USER root

# Install required system tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpython3-dev \
    python3-dev \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy required files into the container
COPY requirements.txt .

# Install python dedendencies and download nltk data
RUN pip install --upgrade pip setuptools wheel && \
    pip install --prefer-binary -r requirements.txt && \
    python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords')"
