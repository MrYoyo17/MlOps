import os
import sys
import io
import glob
import json
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow.pytorch
import mlflow.tracking
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

# === 2. HACK POUR L'IMPORT (Pickle Error) ===
# Redirige 'trainning' et 'train' vers ce script pour trouver la classe CNN
sys.modules['trainning'] = sys.modules[__name__]
sys.modules['train'] = sys.modules[__name__]

# === 3. DÉFINITION DU MODÈLE (Classe CNN) ===
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,8,3,padding=1), nn.ReLU(),
            nn.Conv2d(8,16,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.common_fc = nn.Sequential(nn.Linear(32*16*16, 256), nn.ReLU())
        self.classifier_barbe = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.classifier_moustache = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.classifier_lunettes = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.classifier_cheveux_features = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.classifier_taille_cheveux = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
        self.classifier_couleur_cheveux = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        common_features = self.common_fc(x)
        cheveux_features = self.classifier_cheveux_features(common_features)
        return [
            self.classifier_barbe(common_features),
            self.classifier_moustache(common_features),
            self.classifier_lunettes(common_features),
            self.classifier_taille_cheveux(cheveux_features),
            self.classifier_couleur_cheveux(cheveux_features)
        ]

# === 4. CONFIGURATION ET CHARGEMENT ===

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
model = None
device = torch.device("cpu")

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

def load_production_model():
    global model
    try:
        model_name = "FaceFilterModel"
        print(f"Connexion à MLflow ({MLFLOW_URI})...")
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"❌ Aucun modèle '{model_name}' trouvé.")
            return

        latest = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
        print(f"Chargement version {latest.version}...")
        
        # Récupération et correction du chemin
        local_path = resolve_path(latest.source)
        
        model = mlflow.pytorch.load_model(local_path, map_location=device)
        model.eval()
        print("✅ Modèle chargé !")
    except Exception as e:
        print(f"❌ ERREUR chargement modèle : {e}")

# === 5. PRÉ-TRAITEMENT ===
transform_pipeline = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform_pipeline(image)
    return tensor.unsqueeze(0)

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

@app.route('/', methods=['GET'])
def index():
    return """
    <!doctype html>
    <style>
        body { font-family: sans-serif; text-align: center; padding: 50px; }
        form { margin-top: 20px; }
        input[type=file] { padding: 10px; border: 1px dashed #aaa; }
        input[type=submit] { background: #007bff; color: white; border: none; padding: 10px 20px; cursor: pointer; }
    </style>
    <h1>Face Filter API</h1>
    <p>Chargez une image pour tester le modèle (Mapping v2)</p>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required>
      <br><br>
      <input type="submit" value="Analyser l'image">
    </form>
    """

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        load_production_model()
        if model is None:
            return jsonify({"error": "Modèle indisponible"}), 503

    if 'file' not in request.files:
        return jsonify({"error": "Fichier manquant"}), 400
    
    file = request.files['file']
    
    try:
        input_tensor = process_image(file.read()).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # --- 1. Binaires (0:Non, 1:Oui) ---
            val_barbe = 1 if torch.sigmoid(outputs[0]).item() > 0.5 else 0
            val_moustache = 1 if torch.sigmoid(outputs[1]).item() > 0.5 else 0
            val_lunettes = 1 if torch.sigmoid(outputs[2]).item() > 0.5 else 0
            
            # --- 2. Multi-classes (Raw Index) ---
            raw_taille = torch.argmax(outputs[3], dim=1).item()
            raw_couleur = torch.argmax(outputs[4], dim=1).item()

            # --- 3. Mapping (Model -> User) ---
            # Taille: 0=Chauve, 1=Court, 2=Long
            user_taille_code = MAP_TAILLE_MODEL_TO_USER.get(raw_taille, 0)
            
            # Couleur: 0=Blond, 1=Chatain, 2=Roux, 3=Brun, 4=Gris
            user_couleur_code = raw_couleur # Pas de changement d'index (1:1)

        # Réponse JSON complète
        response = {
            "prediction": {
                "barbe": val_barbe,
                "moustache": val_moustache,
                "lunettes": val_lunettes,
                "taille_cheveux": user_taille_code,
                "couleur_cheveux": user_couleur_code
            },
            "labels": {
                "taille": LABEL_TAILLE.get(user_taille_code, "Inconnu"),
                "couleur": LABEL_COULEUR.get(user_couleur_code, "Inconnu")
            },
            "image_name": file.filename
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
    load_production_model()
    load_predictions_from_csv()
    app.run(host='0.0.0.0', port=5001)