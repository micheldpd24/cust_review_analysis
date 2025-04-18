from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

from review_manager import ReviewsManager  # Import the ReviewsManager class
import docker

# ----
# Function to manage the container running the model training & evaluation 
# and the model's performance dashboard
# ----

def check_and_stop_bertopic_container(**kwargs):
    """
    Check if bertopic' container is running. If so, stop it and remove it. 
    otherwise, do nothing.
    """
    client = docker.from_env()
    container_name = 'bertopic'
    
    try:
        # Search the container by name
        containers = client.containers.list(all=True, filters={"name": container_name})
        
        if containers:
            container = containers[0]
            container_status = container.status
            
            print(f"Container 'bertopic' found with status: {container_status}")
            
            # Si le container est en cours d'exécution, l'arrêter
            if container_status == 'running':
                print("Stopping bertopic' container ...")
                container.stop()
                print("Container 'bertopic' stopped successfully")
                
            # Removing the container
            print("Removing 'bertopic'container ...")
            container.remove()
            print("'bertopic' container removed successfully")
            
        else:
            print("No container named 'bertopic' found. Continuing the pipeline.")
            
    except Exception as e:
        print(f"Error during container verification/stopping: {e}")
        # Do not fail the pipeline in case of error
        print("pipeline continuation despite the error")

# ----

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 4, 4),
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    # 'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# Instantiate the DAG
dag = DAG(
    'reviews_processing_pipeline',
    default_args=default_args,
    description='A pipeline to scrape, process, and analyze customer reviews.',
    schedule_interval='*/120 * * * *',
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
        return "NO_UPDATES"
    else:
        return "SUCCESS"  # Return SUCCESS status to indicate successful load

def branch_task(**kwargs):
    """Decide which branch to take based on load_reviews result."""
    ti = kwargs['ti']
    load_result = ti.xcom_pull(task_ids='load_reviews')
    
    if load_result == "SUCCESS":
        return 'topic_modeling_task'
    else:
        return 'stop_pipeline_task'


# Tâche de vérification et suppression du container si nécessaire
check_container_task = PythonOperator(
    task_id='check_and_stop_bertopic_container',
    python_callable=check_and_stop_bertopic_container,
    provide_context=True,
    dag=dag,
)


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

# Add a dummy operator for the pipeline stop task
stop_pipeline_task = DummyOperator(
    task_id='stop_pipeline_task',   
    dag=dag,
)

# Add a python operator for the train model task
topic_modeling_task = DockerOperator(
    task_id='topic_modeling_task', 
    image='topic-container:latest',
    container_name='bertopic',
    docker_url='unix:///var/run/docker.sock',
    mounts=[
        Mount(source="/Users/micheldpd/Projects/rev_analysis/data/full", target="/data/full", type="bind"),
        Mount(source="/Users/micheldpd/Projects/rev_analysis/data/results", target="/data/results", type="bind"),
        Mount(source="/Users/micheldpd/Projects/rev_analysis/topic_modeling", target="/app",type="bind")
    ],
    network_mode='bridge',
    auto_remove='never',
    # tty=True,
    port_bindings={8050: 8051},
    dag=dag,
)

# Set task dependencies
check_container_task >> extract_reviews >> process_reviews >> load_reviews
load_reviews >> branch >> [topic_modeling_task, stop_pipeline_task]

