import os
import sys
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
import mlflow.pytorch
import mlflow.tracking
from prometheus_flask_exporter import PrometheusMetrics


app = Flask(__name__)
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

if __name__ == '__main__':
    load_production_model()
    app.run(host='0.0.0.0', port=5001)