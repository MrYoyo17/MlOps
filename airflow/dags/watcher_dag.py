from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago
from pathlib import Path
import os
import re

# === CONFIGURATION ===
# Dossier à surveiller (Raw data ou Prepro, selon votre besoin)
# Ici on surveille les données brutes pour déclencher tout le pipeline si besoin
WATCH_DIR = Path("/opt/airflow/data") 
TARGET_DAG_ID = "on_demand_prediction" # Le DAG à déclencher

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
}

def scan_for_new_batches(**context):
    print(f"🔍 Scan du dossier : {WATCH_DIR}")
    
    if not WATCH_DIR.exists():
        print("⚠️ Dossier introuvable.")
        return []

    # 1. Récupérer tous les fichiers images
    files = [f.name for f in WATCH_DIR.glob("*") if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    
    # 2. Extraire les préfixes uniques (ex: s8, s9, s10)
    # On cherche ce qui est avant le premier underscore : "s8_0001.jpg" -> "s8"
    current_prefixes = set()
    for f in files:
        match = re.match(r"^([a-zA-Z0-9]+)_", f)
        if match:
            current_prefixes.add(match.group(1))
            
    print(f"📂 Lots trouvés sur le disque : {current_prefixes}")

    # 3. Récupérer l'historique des lots déjà traités (depuis les Variables Airflow)
    # La variable s'appellera 'processed_batches_list'
    # On stocke ça sous forme de liste séparée par des virgules
    processed_str = Variable.get("processed_batches_list", default_var="")
    processed_prefixes = set(processed_str.split(",")) if processed_str else set()

    # 4. Identifier les nouveaux
    new_prefixes = list(current_prefixes - processed_prefixes)
    
    if not new_prefixes:
        print("✅ Rien de nouveau.")
        return None # Rien à faire

    print(f"🚀 Nouveaux lots détectés : {new_prefixes}")
    
    # 5. Mettre à jour la variable TOUT DE SUITE pour ne pas les relancer au prochain scan
    # On ajoute les nouveaux à l'existant
    updated_processed = processed_prefixes.union(new_prefixes)
    Variable.set("processed_batches_list", ",".join(updated_processed))

    # 6. Retourner la liste pour l'étape suivante
    return new_prefixes

def trigger_target_dags(ti):
    # Récupérer la liste des nouveaux préfixes depuis la tâche précédente (XCom)
    new_prefixes = ti.xcom_pull(task_ids='scan_files')
    
    if not new_prefixes:
        return

    from airflow.api.common.experimental.trigger_dag import trigger_dag
    
    # Pour chaque nouveau préfixe, on lance le DAG de prédiction
    for prefix in new_prefixes:
        print(f"⚡ Déclenchement du DAG {TARGET_DAG_ID} pour le lot {prefix}...")
        try:
            trigger_dag(
                dag_id=TARGET_DAG_ID,
                conf={"prefix": prefix}, # On passe le paramètre !
                replace_microseconds=False,
            )
        except Exception as e:
            print(f"❌ Erreur lors du déclenchement pour {prefix}: {e}")

with DAG('file_watcher_sensor', 
         default_args=default_args, 
         schedule_interval='*/5 * * * *', # Scan toutes les 5 minutes
         catchup=False) as dag:

    # Étape 1 : Scanner et mettre à jour la mémoire
    t_scan = PythonOperator(
        task_id='scan_files',
        python_callable=scan_for_new_batches
    )

    # Étape 2 : Déclencher les DAGs correspondants
    t_trigger = PythonOperator(
        task_id='trigger_predictions',
        python_callable=trigger_target_dags
    )

    t_scan >> t_trigger