import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import mlflow.pytorch
import argparse
from pathlib import Path
from urllib.parse import urlparse

# === CONFIGURATION ===

# 1. Définition de la racine du projet (Compatible Mac et Docker)
# On remonte de 2 crans depuis ce script (ml_scripts/predict_batch.py -> racine)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Chemins des données
# Si la variable d'env existe (Docker), on l'utilise. Sinon on construit le chemin local.
BASE_DATA_DIR = Path(os.environ.get("BASE_DATA_DIR", PROJECT_ROOT / "data"))
IMG_DIR = BASE_DATA_DIR / "dataset_prepro"
OUTPUT_DIR = BASE_DATA_DIR / "predictions"

# 3. Chemin des artefacts (C'est là que se joue la correction)
LOCAL_ARTIFACTS_DIR = PROJECT_ROOT / "ml-artifacts"

# 4. MLflow URI
DEFAULT_MLFLOW_URI = "http://localhost:5050"
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)
mlflow.set_tracking_uri(MLFLOW_URI)

# Mappings
HAIR_LENGTH_MAP = {0: "Court", 1: "Mi-long", 2: "Long"}
HAIR_COLOR_MAP = {0: "Noir", 1: "Brun", 2: "Blond", 3: "Roux", 4: "Gris/Autre"}

# === DÉFINITION DU MODÈLE ===
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

# Correctif Pickle
sys.modules['train'] = sys.modules[__name__]
sys.modules['trainning'] = sys.modules[__name__]

# === FONCTION MAGIQUE DE CORRECTION DE CHEMIN ===

def resolve_path(source_path):
    """
    Adapte le chemin renvoyé par MLflow (qui est un chemin Docker absolu)
    pour qu'il fonctionne sur le Mac si nécessaire.
    """
    # Nettoyage du préfixe file:// si présent
    if source_path.startswith("file://"):
        source_path = source_path[7:]
    
    path_obj = Path(source_path)

    # Cas 1 : Le chemin existe (On est dans Docker ou le dossier est bien à la racine)
    if path_obj.exists():
        return str(path_obj)
    
    # Cas 2 : On est sur Mac, et le chemin commence par /mlartifacts
    # On remplace /mlartifacts par le vrai chemin local
    if str(path_obj).startswith("/mlartifacts"):
        # On enlève "/mlartifacts" du début et on colle le reste à notre dossier local
        relative_part = str(path_obj).replace("/mlartifacts", "", 1).lstrip("/")
        corrected_path = LOCAL_ARTIFACTS_DIR / relative_part
        
        if corrected_path.exists():
            print(f"🔧 Chemin corrigé (Docker -> Mac) : {corrected_path}")
            return str(corrected_path)
            
    # Si on ne trouve rien, on renvoie l'original et ça plantera probablement
    return source_path

def get_latest_model():
    model_name = "FaceFilterModel"
    print(f"Recherche du modèle '{model_name}' sur {MLFLOW_URI}...")
    
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        print(f"❌ Impossible de contacter MLflow : {e}")
        return None

    if not versions:
        print(f"❌ Aucun modèle '{model_name}' trouvé.")
        return None

    # Version la plus récente
    latest = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
    print(f"✅ Version trouvée : {latest.version}")
    
    # --- CORRECTION DU CHARGEMENT ---
    # Au lieu de laisser load_model télécharger aveuglément, on récupère le chemin source
    source_path = latest.source
    
    # On applique notre correctif de chemin
    local_path = resolve_path(source_path)
    
    print(f"📂 Chargement depuis : {local_path}")
    
    device = torch.device("cpu")
    # On charge directement depuis le dossier local corrigé
    model = mlflow.pytorch.load_model(local_path, map_location=device)
    model.eval()
    return model

def run_prediction(prefix="s8"):
    print(f"--- Démarrage Batch Prediction ---")
    print(f"Filtre : '{prefix}'")
    
    model = get_latest_model()
    if model is None:
        return

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not IMG_DIR.exists():
        print(f"❌ Dossier introuvable : {IMG_DIR}")
        return

    # Recherche des images
    files = [f for f in IMG_DIR.glob(f"{prefix}*") if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    
    if not files:
        print(f"⚠️ Aucune image trouvée commençant par '{prefix}' dans {IMG_DIR}")
        return
    
    print(f"📸 {len(files)} images trouvées.")

    results = []
    with torch.no_grad():
        for filepath in files:
            try:
                img = Image.open(filepath).convert('RGB')
                input_tensor = transform(img).unsqueeze(0)
                outputs = model(input_tensor)
                
                has_beard = torch.sigmoid(outputs[0]).item() > 0.5
                has_mustache = torch.sigmoid(outputs[1]).item() > 0.5
                has_glasses = torch.sigmoid(outputs[2]).item() > 0.5
                hair_len_idx = torch.argmax(outputs[3], dim=1).item()
                hair_col_idx = torch.argmax(outputs[4], dim=1).item()
                
                results.append({
                    "filename": filepath.name,
                    "barbe": has_beard,
                    "moustache": has_mustache,
                    "lunettes": has_glasses,
                    "cheveux_taille": HAIR_LENGTH_MAP.get(hair_len_idx, "Inconnu"),
                    "cheveux_couleur": HAIR_COLOR_MAP.get(hair_col_idx, "Inconnu")
                })
            except Exception as e:
                print(f"Erreur sur {filepath.name}: {e}")

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    csv_path = OUTPUT_DIR / f"predictions_{prefix}.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"✅ Sauvegardé : {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="s4")
    args = parser.parse_args()
    run_prediction(prefix=args.prefix)