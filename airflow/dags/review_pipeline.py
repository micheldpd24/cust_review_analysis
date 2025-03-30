from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

from utils import check_new_reviews 
from utils import extract_reviews
from utils import process_reviews
from utils import load_reviews
import logging

def check_new_data(ti):
    """Check if there are new reviews."""
    logging.info("Verifying if there are new reviews.")
    new_review_flag  = ti.xcom_pull(task_ids='load_reviews')
    logging.info(f"New review flag: {new_review_flag}")
    if new_review_flag == 1:
        return "train_task" 
    else: 
        return "end_task"

# Define default arguments
default_args = {
    'owner': 'dj-analytics',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 28),
    'retries': 1,
    'retry_delay': timedelta(seconds=20),
}

# Define DAG
dag = DAG(
    'reviews_analysis_pipeline',
    default_args=default_args,
    description='pipeline for customer reviews analysis',
    schedule_interval='*/5 * * * *',  # Every  minutes
    catchup=False,
    max_active_runs=1
)

# Define tasks

# Create the end task 
end_task = EmptyOperator(
    task_id='end_task',
    dag=dag,
)

# Create the empty train task 
train_task = EmptyOperator(
    task_id='train_task',
    dag=dag,
)
extract_reviews_task = PythonOperator(
    task_id='extract_reviews',
    python_callable=extract_reviews,
    provide_context=True,
    dag=dag,
)

process_reviews_task = PythonOperator(
    task_id='process_reviews',
    python_callable=process_reviews,
    op_args=["{{ ti.xcom_pull(task_ids='extract_reviews') }}"],
    provide_context=True,
    dag=dag,
)

load_reviews_task = PythonOperator(
    task_id='load_reviews',
    python_callable=load_reviews,
    op_args=["{{ ti.xcom_pull(task_ids='process_reviews') }}"],
    dag=dag,
)

check_new_data_task = BranchPythonOperator(
    task_id='check_new_data',
    python_callable=check_new_data,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
extract_reviews_task >> process_reviews_task >> load_reviews_task>> check_new_data_task
check_new_data_task >> [train_task, end_task]
