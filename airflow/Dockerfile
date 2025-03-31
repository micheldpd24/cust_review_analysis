FROM apache/airflow:2.10.3-python3.11

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefer-binary -r requirements.txt && \
    python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords')"