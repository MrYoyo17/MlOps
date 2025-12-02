import os
import sys
import io
import glob
import json
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from prometheus_flask_exporter import PrometheusMetrics


DATA_DIR = os.environ.get("DATA_DIR", "/data")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")
PREDICTIONS_CACHE = {}
PREDICTIONS_DF = None

app = Flask(__name__)
CORS(app)
metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Face Filter API', version='1.0.0')

# === 1. MAPPINGS (Identiques au script batch) ===

# Rappel de la logique d'agrégation (Modèle) : 0=Long, 1=Court, 2=Chauve
# Rappel de la demande utilisateur : 0=Chauve, 1=Court, 2=Long

# Conversion Index Modèle -> Code Utilisateur
MAP_TAILLE_MODEL_TO_USER = {
    0: 2, # Modèle Long -> User Long (2)
    1: 1, # Modèle Court -> User Court (1)
    2: 0  # Modèle Chauve -> User Chauve (0)
}

# Labels pour l'affichage dans l'interface Web (basé sur le Code Utilisateur)
LABEL_TAILLE = {
    0: "Chauve",
    1: "Court", 
    2: "Long"
}

LABEL_COULEUR = {
    0: "Blond", 
    1: "Chatain", 
    2: "Roux", 
    3: "Brun", 
    4: "Gris/Bleu"
}

# === 4. CONFIGURATION ET CHARGEMENT ===

def resolve_path(source_path):
    """Corrige le chemin entre Docker et Mac si nécessaire"""
    if source_path.startswith("file://"):
        source_path = source_path[7:]
    
    path_obj = os.path.normpath(source_path)
    
    # Si le chemin existe tel quel (Docker), on le garde
    if os.path.exists(path_obj):
        return path_obj
    
    # Si on est en local et que le chemin commence par /mlartifacts
    # On suppose que le volume est monté ou accessible localement
    # (Dans l'App Docker, le volume EST monté dans /mlartifacts, donc le cas 1 suffit généralement)
    # Cette sécurité est surtout utile si vous lancez app.py hors docker.
    return path_obj


# === 5. PRÉ-TRAITEMENT ===

def load_predictions_from_csv():
    """
    Charge les prédictions depuis TOUS les fichiers CSV générés par Airflow/Batch.
    Combine les fichiers et supprime les doublons (garde le plus récent).
    """
    global PREDICTIONS_DF
    print("--- Chargement des prédictions CSV ---")
    
    # On cherche tous les CSV dans predictions/
    csv_pattern = os.path.join(PREDICTIONS_DIR, "predictions_*.csv")
    files = glob.glob(csv_pattern)
    
    if not files:
        print("⚠️ Aucun fichier CSV de prédiction trouvé.")
        PREDICTIONS_DF = pd.DataFrame()
        return

    # Tri par date de modification (ancien -> récent)
    files.sort(key=os.path.getctime)
    print(f"📂 {len(files)} fichiers CSV trouvés. Chargement et fusion...")

    dfs = []
    for f in files:
        try:
            print(f"   - Lecture : {os.path.basename(f)}")
            df_temp = pd.read_csv(f)
            dfs.append(df_temp)
        except Exception as e:
            print(f"❌ Erreur lecture {f} : {e}")

    if not dfs:
        PREDICTIONS_DF = pd.DataFrame()
        return

    # Fusion
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Suppression des doublons (on garde le dernier = le plus récent car trié par date)
    if 'image_name' in full_df.columns:
        full_df.drop_duplicates(subset=['image_name'], keep='last', inplace=True)
    
    try:
        # Mapping des colonnes CSV vers API
        # CSV: image_name, barbe, moustache, lunettes, taille_cheveux, couleur_cheveux
        # API: id, beard, mustache, glasses, hairLength, hairColor
        
        full_df['beard'] = full_df['barbe'].astype(bool)
        full_df['mustache'] = full_df['moustache'].astype(bool)
        full_df['glasses'] = full_df['lunettes'].astype(bool)
        
        # Mapping Taille (0=Chauve, 1=Court, 2=Long) -> (bald, short, long)
        taille_map = {0: 'bald', 1: 'short', 2: 'long'}
        full_df['hairLength'] = full_df['taille_cheveux'].map(taille_map)
        
        # Mapping Couleur (0=Blond, 1=Chatain, 2=Roux, 3=Brun, 4=Gris) -> (blond, lightBrown, red, darkBrown, grayBlue)
        couleur_map = {0: 'blond', 1: 'lightBrown', 2: 'red', 3: 'darkBrown', 4: 'grayBlue'}
        full_df['hairColor'] = full_df['couleur_cheveux'].map(couleur_map)
        
        PREDICTIONS_DF = full_df
        print(f"✅ DataFrame chargé et fusionné : {PREDICTIONS_DF.shape}")
        print(PREDICTIONS_DF.head())
        
    except Exception as e:
        print(f"❌ Erreur processing DataFrame : {e}")
        PREDICTIONS_DF = pd.DataFrame()

# === 6. ROUTES API ===

@app.route('/images', methods=['GET'])
def list_images():
    """
    Liste les images pré-traitées avec filtrage et pagination via Pandas.
    """
    if PREDICTIONS_DF is None:
        return jsonify([])

    # 1. Pagination Params
    try:
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))
    except ValueError:
        limit = 20
        offset = 0

    # 2. Filters
    f_beard = request.args.get('beard')
    f_mustache = request.args.get('mustache')
    f_glasses = request.args.get('glasses')
    f_hair_color = request.args.get('hairColor')
    f_hair_length = request.args.get('hairLength')

    # Start with full DF
    df = PREDICTIONS_DF.copy()

    # Apply Filters
    if f_beard is not None:
        val = f_beard.lower() == 'true'
        df = df[df['beard'] == val]
    
    if f_mustache is not None:
        val = f_mustache.lower() == 'true'
        df = df[df['mustache'] == val]
        
    if f_glasses is not None:
        val = f_glasses.lower() == 'true'
        df = df[df['glasses'] == val]

    if f_hair_color is not None and f_hair_color != 'any':
        df = df[df['hairColor'] == f_hair_color]

    if f_hair_length is not None and f_hair_length != 'any':
        df = df[df['hairLength'] == f_hair_length]

    # 3. Pagination
    # Slice the dataframe
    paginated_df = df.iloc[offset : offset + limit]
    
    # 4. Format Response
    results = []
    for _, row in paginated_df.iterrows():
        results.append({
            "id": row['image_name'],
            "imageUrl": f"/images/{row['image_name']}",
            "processeds": [{
                "result": {
                    "beard": row['beard'],
                    "mustache": row['mustache'],
                    "glasses": row['glasses'],
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

if __name__ == '__main__':
    load_predictions_from_csv()
    app.run(host='0.0.0.0', port=5001)