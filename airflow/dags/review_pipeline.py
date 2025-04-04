from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
from review_manager import ReviewsManager  # Import the ReviewsManager class

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 4, 4),
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# Instantiate the DAG
dag = DAG(
    'reviews_processing_pipeline',
    default_args=default_args,
    description='A pipeline to scrape, process, and analyze customer reviews.',
    schedule_interval='*/60 * * * *',
    catchup=False,
)

# Initialize the ReviewsManager instance
manager = ReviewsManager()

# Define the tasks
def extract_reviews_task():
    """Task to extract reviews."""
    raw_file = manager.extract_reviews()
    if not raw_file:
        raise ValueError("Review extraction failed.")
    return raw_file

def process_reviews_task(**kwargs):
    """Task to process raw reviews."""
    ti = kwargs['ti']
    raw_file = ti.xcom_pull(task_ids='extract_reviews')
    cleaned_file = manager.process_reviews(raw_file)
    if not cleaned_file:
        raise ValueError("Review processing failed.")
    return cleaned_file

def load_reviews_task(**kwargs):
    """Task to load cleaned reviews into the database."""
    ti = kwargs['ti']
    cleaned_file = ti.xcom_pull(task_ids='process_reviews')
    result = manager.load_reviews(cleaned_file)
    if result == None:  # ERROR case
        raise ValueError("Failed to load reviews into the database.")
    elif result == 0:  # NO_UPDATES case
        print("No new reviews to add to the database.")
    else:
        return "SUCCESS"  # Return SUCCESS status to indicate successful load

def branch_task(**kwargs):
    """Decide which branch to take based on load_reviews result."""
    ti = kwargs['ti']
    load_result = ti.xcom_pull(task_ids='load_reviews')
    
    if load_result == "SUCCESS":
        return 'prepare_ml_datasets'
    else:
        return 'skip_ml_preparation'

def prepare_ml_datasets_task():
    """Task to prepare machine learning datasets."""
    try:
        manager.prepare_ml_datasets()
    except Exception as e:
        raise ValueError(f"Error preparing ML datasets: {e}")

# Define the tasks using PythonOperator
extract_reviews = PythonOperator(
    task_id='extract_reviews',
    python_callable=extract_reviews_task,
    dag=dag,
)

process_reviews = PythonOperator(
    task_id='process_reviews',
    python_callable=process_reviews_task,
    provide_context=True,
    dag=dag,
)

load_reviews = PythonOperator(
    task_id='load_reviews',
    python_callable=load_reviews_task,
    provide_context=True,
    dag=dag,
)

# Branch operator to decide next steps
branch = BranchPythonOperator(
    task_id='branch_task',
    python_callable=branch_task,
    provide_context=True,
    dag=dag,
)

prepare_ml_datasets = PythonOperator(
    task_id='prepare_ml_datasets',
    python_callable=prepare_ml_datasets_task,
    dag=dag,
)

# Add a dummy operator for the skip path
stop_pipeline_task = DummyOperator(
    task_id='stop_pipeline_task',   
    dag=dag,
)


# Set task dependencies
extract_reviews >> process_reviews >> load_reviews >> branch
branch >> [prepare_ml_datasets, stop_pipeline_task]
