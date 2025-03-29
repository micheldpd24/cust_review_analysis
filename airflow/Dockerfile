FROM apache/airflow:2.10.3-python3.11

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --prefer-binary -r requirements.txt
