from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Ajout du dossier ml_scripts au path pour pouvoir importer les modules
sys.path.append("/opt/airflow/ml_scripts")

from preprocess import run_preprocess
from train import run_training

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

with DAG('mlops_face_filter', default_args=default_args, schedule_interval=None, catchup=False) as dag:
    
    t_preprocess = PythonOperator(
        task_id='preprocess',
        python_callable=run_preprocess
    )

    t_train = PythonOperator(
        task_id='train',
        python_callable=run_training
    )

    t_preprocess >> t_train

with DAG('mlops_face_filter_train', default_args=default_args, schedule_interval=None, catchup=False) as dag:
    
    t_train = PythonOperator(
        task_id='train',
        python_callable=run_training
    )