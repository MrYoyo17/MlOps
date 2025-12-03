from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os
from airflow.models.param import Param

# Ajout du dossier ml_scripts au path pour pouvoir importer les modules
sys.path.append("/opt/airflow/ml_scripts")

from preprocess import run_preprocess
from train import run_training
from annotate import run_aggregation
from predict_batch import run_prediction
from detect_drift import run_drift_detection

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

# Fonction wrapper pour récupérer le paramètre depuis l'interface Airflow
def prediction_wrapper(**context):
    # On récupère la configuration passée lors du trigger
    # Si rien n'est passé, on utilise 's8' par défaut
    conf = context['dag_run'].conf or {}
    prefix_param = conf.get('prefix', 's8')
    
    print(f"Lancement du DAG avec le préfixe : {prefix_param}")
    run_prediction(prefix=prefix_param)

def drift_wrapper(**context):
    conf = context['dag_run'].conf or {}
    prefix_param = conf.get('prefix', 's8')
    run_drift_detection(prefix=prefix_param)

with DAG('mlops_face_filter', default_args=default_args, schedule_interval=None, catchup=False) as dag:
    t_annotate = PythonOperator(
        task_id='annotate',
        python_callable=run_aggregation
    )

    t_preprocess = PythonOperator(
        task_id='preprocess',
        python_callable=run_preprocess
    )

    t_train = PythonOperator(
        task_id='train',
        python_callable=run_training
    )

    t_predict = PythonOperator(
        task_id='batch_predict',
        python_callable=run_prediction(prefix='s') # Prédiction sur toutes les images
    )

    [t_annotate, t_preprocess] >> t_train >> t_predict

with DAG('mlops_face_filter_train', default_args=default_args, schedule_interval=None, catchup=False) as dag:
    
    t_train = PythonOperator(
        task_id='train',
        python_callable=run_training
    )

with DAG('on_demand_prediction', default_args=default_args, schedule_interval=None, catchup=False,
         params={
             "prefix": Param("s8", type="string", description="Préfixe des images (ex: s8)")
         }) as dag:

    t_preprocess = PythonOperator(
        task_id='preprocess',
        python_callable=run_preprocess
    )

    t_predict = PythonOperator(
        task_id='batch_predict',
        python_callable=prediction_wrapper,
        provide_context=True # Important pour accéder à 'dag_run'
    )

    t_drift = PythonOperator(
        task_id='detect_drift',
        python_callable=drift_wrapper,
        provide_context=True
    )

    t_preprocess >> t_predict >> t_drift