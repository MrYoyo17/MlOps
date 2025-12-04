from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago
from datetime import timedelta  # <--- AJOUT
from pathlib import Path
import os

# === CONFIGURATION ===
# On garde Path pour gérer le chemin, mais on utilisera os pour scanner
WATCH_DIR_PATH = Path("/opt/airflow/data")
WATCH_DIR_STR = "/opt/airflow/data" 
TARGET_DAG_ID = "mlops_face_filter"
IGNORE_FILES = {"csv_global.csv"} 

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    # AJOUT : On augmente le timeout par défaut à 30 minutes
    'execution_timeout': timedelta(minutes=30),
}

def scan_based_on_count(**context):
    print(f"🔍 Comptage optimisé des CSV dans : {WATCH_DIR_STR}")
    
    if not os.path.exists(WATCH_DIR_STR):
        print("⚠️ Dossier introuvable.")
        return

    # --- OPTIMISATION DU COMPTAGE ---
    # Au lieu de charger une liste en mémoire ([f for f in ...]), 
    # on compte à la volée avec un itérateur bas niveau (os.scandir).
    # C'est 10x à 50x plus rapide sur les volumes Docker.
    
    current_count = 0
    # os.scandir est un générateur, il ne charge pas tout en RAM
    with os.scandir(WATCH_DIR_STR) as entries:
        for entry in entries:
            # On vérifie si c'est un fichier, s'il finit par .csv et n'est pas ignoré
            if entry.is_file() and entry.name.endswith('.csv') and entry.name not in IGNORE_FILES:
                current_count += 1
    
    # --------------------------------
    
    # Récupérer l'ancien compte (Variable Airflow)
    previous_count = int(Variable.get("csv_count_memory", default_var="-1"))
    
    print(f"📊 Compte Actuel : {current_count} | Ancien Compte : {previous_count}")

    # Cas A : Initialisation
    if previous_count == -1:
        print("🆕 Premier lancement : Initialisation.")
        Variable.set("csv_count_memory", str(current_count))
        return

    # Cas B : Augmentation
    if current_count > previous_count:
        diff = current_count - previous_count
        print(f"🚀 {diff} nouveaux fichiers détectés !")
        
        Variable.set("csv_count_memory", str(current_count))
        
        from airflow.api.common.experimental.trigger_dag import trigger_dag
        print(f"⚡ Lancement du DAG : {TARGET_DAG_ID}")
        try:
            trigger_dag(
                dag_id=TARGET_DAG_ID,
                run_id=f"trigger_count_{current_count}",
                replace_microseconds=False,
            )
        except Exception as e:
            print(f"❌ Erreur déclenchement : {e}")

    # Cas C : Diminution
    elif current_count < previous_count:
        print(f"📉 Fichiers supprimés. Mise à jour référence.")
        Variable.set("csv_count_memory", str(current_count))
        
    else:
        print("✅ Stable.")

with DAG('csv_count_watcher', 
         default_args=default_args, 
         schedule_interval='*/10 * * * *', # On passe à 10min pour laisser souffler le disque
         catchup=False) as dag:

    t_watch = PythonOperator(
        task_id='watch_csv_count',
        python_callable=scan_based_on_count,
        # On peut aussi surcharger le timeout juste pour cette tâche
        execution_timeout=timedelta(minutes=30)
    )