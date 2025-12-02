import pandas as pd
import os
import argparse
from pathlib import Path
import mlflow
try:
    # 1. Le rapport principal
    from evidently.report import Report
    
    # 2. Les presets (DataDrift, TargetDrift)
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    
    # 3. Le ColumnMapping (Attention, il a bougé à la racine dans les versions récentes)
    try:
        from evidently import ColumnMapping
    except ImportError:
        # Fallback pour les versions plus anciennes (0.2 - 0.4)
        from evidently.pipeline.column_mapping import ColumnMapping

except ImportError as e:
    print(f"❌ ERREUR CRITIQUE D'IMPORT EVIDENTLY : {e}")
    print("Vérifiez que evidently >= 0.4 est installé.")
    raise e

# === CONFIGURATION ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = Path(os.environ.get("BASE_DATA_DIR", PROJECT_ROOT / "data"))

# Chemins
TRAIN_DATA_PATH = BASE_DATA_DIR / "csv_global.csv"
PREDICTIONS_DIR = BASE_DATA_DIR / "predictions"
REPORTS_DIR = BASE_DATA_DIR / "reports"

# MLflow
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Data_Drift_Monitoring")

def run_drift_detection(prefix="s8"):
    print(f"--- Analyse du Drift pour le lot : {prefix} ---")

    # 1. Chargement des données de RÉFÉRENCE (Entraînement)
    if not TRAIN_DATA_PATH.exists():
        print(f"❌ Données d'entraînement introuvables : {TRAIN_DATA_PATH}")
        return
    
    ref_df = pd.read_csv(TRAIN_DATA_PATH)
    
    # 2. Chargement des données ACTUELLES (Prédictions)
    curr_path = PREDICTIONS_DIR / f"predictions_{prefix}.csv"
    if not curr_path.exists():
        print(f"❌ Prédictions introuvables : {curr_path}. Lancez d'abord predict_batch.py")
        return
        
    curr_df = pd.read_csv(curr_path)
    
    print(f"📊 Reference: {len(ref_df)} lignes | Current: {len(curr_df)} lignes")

    # 3. Configuration du Mapping
    # On compare les colonnes communes (les attributs prédits)
    # Dans Reference on a : barbe, moustache, lunette, long, court...
    # Dans Current on a : barbe, moustache, lunettes, taille_cheveux, couleur_cheveux...
    
    # Pour que ça marche, il faut harmoniser les colonnes ou se concentrer sur les labels communs.
    # Ici, nous allons surveiller les colonnes BINAIRES communes.
    
    # Renommage pour aligner 'lunette' (train) et 'lunettes' (predict) si besoin
    if 'lunette' in ref_df.columns:
        ref_df = ref_df.rename(columns={'lunette': 'lunettes'})

    # Colonnes à surveiller (Catégorielles)
    target_columns = ['barbe', 'moustache', 'lunettes']
    
    # Mapping Evidently
    col_mapping = ColumnMapping()
    col_mapping.categorical_features = target_columns
    # On ignore les colonnes qui n'existent pas dans les deux
    
    # Création de sous-dataframes propres
    ref_data = ref_df[target_columns].dropna()
    curr_data = curr_df[target_columns].dropna()

    # 4. Génération du Rapport
    report = Report(metrics=[
        DataDriftPreset(),   # Vérifie si la distribution des données a changé
    ])

    print("Calcul du drift en cours...")
    report.run(reference_data=ref_data, current_data=curr_data, column_mapping=col_mapping)

    # 5. Sauvegarde HTML
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"drift_report_{prefix}.html"
    report.save_html(str(report_path))
    print(f"✅ Rapport HTML généré : {report_path}")

    # 6. Envoi des métriques à MLflow
    # On extrait un dictionnaire Python des résultats
    results = report.as_dict()
    
    # Récupération du booléen "Dataset Drift Detected"
    drift_share = results['metrics'][0]['result']['drift_share']
    dataset_drift = results['metrics'][0]['result']['dataset_drift']
    
    with mlflow.start_run(run_name=f"Drift_Check_{prefix}"):
        mlflow.log_param("prefix", prefix)
        mlflow.log_metric("drift_share", drift_share)
        mlflow.log_metric("dataset_drift", int(dataset_drift))
        
        # On attache le rapport HTML comme artefact MLflow pour le voir dans l'UI
        mlflow.log_artifact(str(report_path), "drift_reports")
        
        if dataset_drift:
            print("⚠️ DRIFT DÉTECTÉ ! Vérifiez le rapport.")
        else:
            print("👍 Aucun drift majeur détecté.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="s8")
    args = parser.parse_args()
    run_drift_detection(prefix=args.prefix)