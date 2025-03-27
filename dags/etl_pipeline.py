from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta

from utils import check_new_reviews 
from utils import extract_reviews
from utils import process_reviews
from utils import load_reviews

def check_new_data(**kwargs):
    """Check if there are new reviews."""
    print("Verifying if there are new reviews.")
    new_review_flag  = check_new_reviews()
    kwargs['ti'].xcom_push(key='new_review_flag', value=new_review_flag)
    return "load_reviews" if new_review_flag == 1 else "stop_pipeline"

def stop_pipeline():
    """Stop the pipeline if no new reviews are found."""
    print("No new reviews found. Stopping the pipeline.")

# Define default arguments
default_args = {
    'owner': 'dj-analytics',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 27),
    'retries': 1,
    'retry_delay': timedelta(seconds=20),
}

# Define DAG
dag = DAG(
    'customer_reviews_analysis',
    default_args=default_args,
    description='pipeline for customer reviews analysis',
    schedule_interval='*/5 * * * *',  # Every 5 minutes
    catchup=False,
    max_active_runs=1
)

# Define tasks
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

check_new_data_task = BranchPythonOperator(
    task_id='check_new_data',
    python_callable=check_new_data,
    provide_context=True,
    dag=dag,
)

load_reviews_task = PythonOperator(
    task_id='load_reviews',
    python_callable=load_reviews,
    op_args=["{{ ti.xcom_pull(task_ids='process_reviews') }}"],
    dag=dag,
)

stop_pipeline_task = PythonOperator(
    task_id='stop_pipeline',
    python_callable=stop_pipeline,
    dag=dag,
)

# Define task dependencies
extract_reviews_task >> process_reviews_task >> check_new_data_task 
check_new_data_task >> [load_reviews_task, stop_pipeline_task]
