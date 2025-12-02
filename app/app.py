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

sys.modules['train'] = sys.modules[__name__]

app = Flask(__name__)

# --- 1. DÉFINITION DE LA CLASSE (OBLIGATOIRE POUR LE CHARGEMENT) ---
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
        
        # Input 64x64 -> 32*16*16 features
        self.common_fc = nn.Sequential(
            nn.Linear(32*16*16, 256), nn.ReLU(),
        )

        # Têtes Binaires
        self.classifier_barbe = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.classifier_moustache = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.classifier_lunettes = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        
        # Branche "Cheveux"
        self.classifier_cheveux_features = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.classifier_taille_cheveux = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
        self.classifier_couleur_cheveux = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        common_features = self.common_fc(x)

        out_barbe = self.classifier_barbe(common_features)
        out_moustache = self.classifier_moustache(common_features)
        out_lunettes = self.classifier_lunettes(common_features)
        
        cheveux_features = self.classifier_cheveux_features(common_features)
        out_taille_cheveux = self.classifier_taille_cheveux(cheveux_features)
        out_couleur_cheveux = self.classifier_couleur_cheveux(cheveux_features)

        return [out_barbe, out_moustache, out_lunettes, out_taille_cheveux, out_couleur_cheveux]

# --- 2. CONFIGURATION ET CHARGEMENT DU MODÈLE ---

# Mapping des index vers des noms lisibles (A ADAPTER SELON VOTRE DATASET)
HAIR_LENGTH_MAP = {0: "Court", 1: "Mi-long", 2: "Long"}
HAIR_COLOR_MAP = {0: "Noir", 1: "Brun", 2: "Blond", 3: "Roux", 4: "Gris/Autre"}

# Configuration MLflow
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_uri)

model = None
device = torch.device("cpu") # On utilise le CPU pour l'inférence simple de l'API

def load_production_model():
    global model
    try:
        model_name = "FaceFilterModel" # Le nom donné dans train.py lors du log
        # On essaie de charger la version en "Production", sinon la dernière version ("None")
        print(f"Connexion à MLflow ({mlflow_uri})...")
        # Note: Si vous n'avez pas encore taggé de modèle en production, mettez `alias="latest"` ou une version spécifique
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"ERREUR: Aucun modèle nommé '{model_name}' trouvé dans MLflow.")
            return

        # On trie pour avoir la dernière version créée
        latest_version = sorted(versions, key=lambda x: x.version, reverse=True)[0]
        version_id = latest_version.version
        model_uri = f"models:/{model_name}/{version_id}" # Charge la dernière version
        
        print(f"Chargement du modèle depuis {model_uri} avec la version {version_id}...")
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        model.eval()
        print("Modèle chargé avec succès !")
    except Exception as e:
        print(f"ERREUR lors du chargement du modèle : {e}")
        print("L'API démarrera mais les prédictions échoueront tant que le modèle n'est pas dispo.")

# --- 3. PRÉ-TRAITEMENT ---
# Doit être IDENTIQUE à l'entraînement (+ Resize obligatoire)
transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform_pipeline(image)
    return tensor.unsqueeze(0) # Ajoute la dimension batch [1, 3, 64, 64]

# --- 4. ROUTES API ---
@app.route('/', methods=['GET'])
def index():
    return """
    <!doctype html>
    <title>Test Face Filter</title>
    <h1>Envoyer une photo pour obtenir la prédiction</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*">
      <input type="submit" value="Analyser">
    </form>
    """

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        # Tentative de rechargement si échec au démarrage
        load_production_model()
        if model is None:
            return jsonify({"error": "Modèle non disponible"}), 503

    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier envoyé"}), 400
    
    file = request.files['file']
    
    try:
        input_tensor = process_image(file.read()).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            # Rappel de l'ordre : [barbe, moustache, lunettes, taille, couleur]
            
            # Traitement Binaires (Sigmoid > 0.5)
            has_beard = torch.sigmoid(outputs[0]).item() > 0.5
            has_mustache = torch.sigmoid(outputs[1]).item() > 0.5
            has_glasses = torch.sigmoid(outputs[2]).item() > 0.5
            
            # Traitement Multi-classes (Argmax)
            hair_len_idx = torch.argmax(outputs[3], dim=1).item()
            hair_col_idx = torch.argmax(outputs[4], dim=1).item()

        response = {
            "beard": has_beard,
            "mustache": has_mustache,
            "glasses": has_glasses,
            "hair_length": HAIR_LENGTH_MAP.get(hair_len_idx, "Inconnu"),
            "hair_color": HAIR_COLOR_MAP.get(hair_col_idx, "Inconnu"),
            # On renvoie aussi les indices bruts pour le débogage
            "debug_indices": {"length": hair_len_idx, "color": hair_col_idx}
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Charger le modèle au démarrage
    load_production_model()
    app.run(host='0.0.0.0', port=5002)