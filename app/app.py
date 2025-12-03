import os
import glob
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Gauge, Counter

DATA_DIR = os.environ.get("DATA_DIR", "/data")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")
PREDICTIONS_DF = None

app = Flask(__name__)
CORS(app)

# --- 1. CONFIGURATION PROMETHEUS ---

# group_by='path' permet d'avoir des métriques HTTP par route (/images, /health, etc.)
metrics = PrometheusMetrics(app, group_by='path')
metrics.info('app_info', 'Face Filter API', version='1.0.0')

# Métrica 1 : Compteur global de prédictions chargées
PREDICTIONS_TOTAL = Gauge('mlops_predictions_loaded_total', 'Total number of predictions currently in memory')

# Métrica 2 : Attributs binaires (Barbe, Moustache, Lunettes)
# Utilisation de labels pour éviter de multiplier les variables
FEATURE_COUNT = Gauge('mlops_feature_count', 'Number of people with a specific feature', ['feature'])

# Métrica 3 : Distribution des attributs capillaires (Couleur, Taille)
HAIR_DISTRIBUTION = Gauge('mlops_hair_distribution_count', 'Distribution of hair attributes', ['attribute', 'value'])

# Métrica 4 : Compteur custom pour les appels API (optionnel car PrometheusMetrics le fait déjà, mais utile pour du tracking métier précis)
SEARCH_REQUESTS = Counter('mlops_search_requests_total', 'Total number of search requests with filters', ['has_filter'])


# === 2. LOGIQUE MÉTIER & CHARGEMENT ===

def load_predictions_from_csv():
    """
    Charge, fusionne et met à jour les métriques Prometheus.
    """
    global PREDICTIONS_DF
    print("--- Chargement des prédictions CSV ---")
    
    csv_pattern = os.path.join(PREDICTIONS_DIR, "predictions_*.csv")
    files = glob.glob(csv_pattern)
    
    if not files:
        print("⚠️ Aucun fichier CSV trouvé.")
        PREDICTIONS_DF = pd.DataFrame()
        # Reset des métriques à 0 si pas de fichiers
        PREDICTIONS_TOTAL.set(0)
        return

    # Tri par date
    files.sort(key=os.path.getctime)
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"❌ Erreur lecture {f} : {e}")

    if not dfs:
        PREDICTIONS_DF = pd.DataFrame()
        return

    full_df = pd.concat(dfs, ignore_index=True)
    
    # Dédoublonnage sur image_name
    if 'image_name' in full_df.columns:
        full_df.drop_duplicates(subset=['image_name'], keep='last', inplace=True)
    
    try:
        # --- Mapping et Nettoyage ---
        full_df['beard'] = full_df['barbe'].astype(bool)
        full_df['mustache'] = full_df['moustache'].astype(bool)
        full_df['glasses'] = full_df['lunettes'].astype(bool)
        
        taille_map = {0: 'bald', 1: 'short', 2: 'long'}
        full_df['hairLength'] = full_df['taille_cheveux'].map(taille_map)
        
        couleur_map = {0: 'blond', 1: 'lightBrown', 2: 'red', 3: 'darkBrown', 4: 'grayBlue'}
        full_df['hairColor'] = full_df['couleur_cheveux'].map(couleur_map)
        
        PREDICTIONS_DF = full_df
        print(f"✅ DataFrame chargé : {len(PREDICTIONS_DF)} lignes.")

        # --- MISE À JOUR DES MÉTRIQUES PROMETHEUS ---
        update_prometheus_metrics(full_df)

    except Exception as e:
        print(f"❌ Erreur processing DataFrame : {e}")
        PREDICTIONS_DF = pd.DataFrame()

def update_prometheus_metrics(df):
    """Calcule et pousse les stats dans les Gauges"""
    try:
        # 1. Total
        count = len(df)
        PREDICTIONS_TOTAL.set(count)

        # 2. Features binaires (Labels)
        beard_count = int(df['beard'].sum())
        mustache_count = int(df['mustache'].sum())
        glasses_count = int(df['glasses'].sum())

        FEATURE_COUNT.labels(feature='beard').set(beard_count)
        FEATURE_COUNT.labels(feature='mustache').set(mustache_count)
        FEATURE_COUNT.labels(feature='glasses').set(glasses_count)

        # 3. Distribution Cheveux (Couleur)
        # On compte les valeurs uniques et on boucle
        if 'hairColor' in df.columns:
            color_counts = df['hairColor'].value_counts()
            for color, count in color_counts.items():
                HAIR_DISTRIBUTION.labels(attribute='color', value=str(color)).set(count)

        # 4. Distribution Cheveux (Taille)
        if 'hairLength' in df.columns:
            length_counts = df['hairLength'].value_counts()
            for length, count in length_counts.items():
                HAIR_DISTRIBUTION.labels(attribute='length', value=str(length)).set(count)
        
        print("📊 Prometheus Metrics updated successfully.")
        
    except Exception as e:
        print(f"⚠️ Failed to update metrics: {e}")


# === 3. ROUTES API ===

@app.route('/images', methods=['GET'])
def list_images():
    if PREDICTIONS_DF is None or PREDICTIONS_DF.empty:
        return jsonify([])

    # Paramètres
    try:
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))
    except ValueError:
        limit = 20
        offset = 0

    # Filtres
    f_beard = request.args.get('beard')
    f_mustache = request.args.get('mustache')
    f_glasses = request.args.get('glasses')
    f_hair_color = request.args.get('hairColor')
    f_hair_length = request.args.get('hairLength')

    # Métrique: On incrémente le compteur de recherche
    # On regarde si l'utilisateur a utilisé au moins un filtre
    has_filter = any([f_beard, f_mustache, f_glasses, f_hair_color, f_hair_length])
    SEARCH_REQUESTS.labels(has_filter=str(has_filter)).inc()

    # Filtrage Pandas
    df = PREDICTIONS_DF.copy()

    if f_beard is not None:
        df = df[df['beard'] == (f_beard.lower() == 'true')]
    if f_mustache is not None:
        df = df[df['mustache'] == (f_mustache.lower() == 'true')]
    if f_glasses is not None:
        df = df[df['glasses'] == (f_glasses.lower() == 'true')]
    if f_hair_color and f_hair_color != 'any':
        df = df[df['hairColor'] == f_hair_color]
    if f_hair_length and f_hair_length != 'any':
        df = df[df['hairLength'] == f_hair_length]

    # Pagination
    paginated_df = df.iloc[offset : offset + limit]
    
    # Formatage de la réponse
    results = []
    for _, row in paginated_df.iterrows():
        results.append({
            "id": row['image_name'],
            "imageUrl": f"/images/{row['image_name']}",
            "processeds": [{
                "result": {
                    "beard": bool(row['beard']),
                    "mustache": bool(row['mustache']),
                    "glasses": bool(row['glasses']),
                    "hairLength": row['hairLength'],
                    "hairColor": row['hairColor']
                }
            }]
        })
    
    return jsonify(results)

@app.route('/images/<path:filename>')
def serve_image(filename):
    from flask import send_from_directory
    return send_from_directory(os.path.join(DATA_DIR), filename)

# Endpoint bonus pour recharger les données sans redémarrer le pod
@app.route('/refresh-data', methods=['POST'])
def refresh_data():
    load_predictions_from_csv()
    return jsonify({"status": "refreshed", "total_images": len(PREDICTIONS_DF)})

if __name__ == '__main__':
    load_predictions_from_csv()
    app.run(host='0.0.0.0', port=5001)