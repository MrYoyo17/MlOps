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

# === CONFIGURATION ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = Path(os.environ.get("BASE_DATA_DIR", PROJECT_ROOT / "data"))
IMG_DIR = BASE_DATA_DIR / "dataset_prepro"
OUTPUT_DIR = BASE_DATA_DIR / "predictions"
LOCAL_ARTIFACTS_DIR = PROJECT_ROOT / "ml-artifacts"

# MLflow
DEFAULT_MLFLOW_URI = "http://localhost:5050"
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)
mlflow.set_tracking_uri(MLFLOW_URI)

# === MAPPINGS DE SORTIE (Conversion Modèle -> Format demandé) ===

# Votre modèle (basé sur aggregate.py) sort : 0=Long, 1=Court, 2=Chauve
# Vous voulez : 0=Chauve, 1=Court, 2=Long
MAP_TAILLE_TO_USER = {
    0: 2, # Modèle Long -> User Long (2)
    1: 1, # Modèle Court -> User Court (1)
    2: 0  # Modèle Chauve -> User Chauve (0)
}

# Votre modèle sort : 0=Blond, 1=Chatain, 2=Roux, 3=Brun, 4=Gris
# Vous voulez : Idem. Pas de changement nécessaire.
MAP_COULEUR_TO_USER = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4
}

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

# === FONCTIONS UTILITAIRES ===

def resolve_path(source_path):
    """Corrige le chemin entre Docker et Mac"""
    if source_path.startswith("file://"):
        source_path = source_path[7:]
    path_obj = Path(source_path)
    if path_obj.exists(): return str(path_obj)
    
    # Tentative de correction si on est sur Mac
    if str(path_obj).startswith("/mlartifacts"):
        relative_part = str(path_obj).replace("/mlartifacts", "", 1).lstrip("/")
        corrected_path = LOCAL_ARTIFACTS_DIR / relative_part
        if corrected_path.exists():
            return str(corrected_path)
    return source_path

def get_latest_model():
    model_name = "FaceFilterModel"
    print(f"Recherche du modèle '{model_name}'...")
    client = mlflow.tracking.MlflowClient()
    
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        print(f"❌ Impossible de contacter MLflow : {e}")
        return None

    if not versions:
        print(f"❌ Aucun modèle trouvé.")
        return None

    latest = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
    print(f"✅ Version trouvée : {latest.version}")
    
    local_path = resolve_path(latest.source)
    device = torch.device("cpu")
    model = mlflow.pytorch.load_model(local_path, map_location=device)
    model.eval()
    return model

def run_prediction(prefix="s8"):
    print(f"--- Batch Prediction ({prefix}) ---")
    
    model = get_latest_model()
    if model is None: return

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. Récupérer et TRIER les fichiers (s2_00000 avant s2_00001)
    if not IMG_DIR.exists():
        print(f"❌ Dossier introuvable : {IMG_DIR}")
        return

    # On utilise un tri alphabétique simple sur le nom de fichier
    files = sorted(
        [f for f in IMG_DIR.glob(f"{prefix}*") if f.suffix.lower() in ['.jpg', '.png', '.jpeg']],
        key=lambda x: x.name
    )
    
    if not files:
        print(f"⚠️ Aucune image trouvée pour '{prefix}'")
        return
    
    print(f"📸 {len(files)} images à traiter.")

    results = []
    with torch.no_grad():
        for filepath in files:
            try:
                img = Image.open(filepath).convert('RGB')
                input_tensor = transform(img).unsqueeze(0)
                outputs = model(input_tensor)
                
                # --- Récupération des valeurs ---
                # Binaires (0 ou 1)
                val_barbe = 1 if torch.sigmoid(outputs[0]).item() > 0.5 else 0
                val_moustache = 1 if torch.sigmoid(outputs[1]).item() > 0.5 else 0
                val_lunettes = 1 if torch.sigmoid(outputs[2]).item() > 0.5 else 0
                
                # Multi-classes (Indices bruts du modèle)
                raw_taille = torch.argmax(outputs[3], dim=1).item()
                raw_couleur = torch.argmax(outputs[4], dim=1).item()
                
                # --- Mapping vers format utilisateur ---
                final_taille = MAP_TAILLE_TO_USER.get(raw_taille, 0)
                final_couleur = MAP_COULEUR_TO_USER.get(raw_couleur, 0)
                
                results.append({
                    "image_name": filepath.name, # Nom exact demandé
                    "barbe": val_barbe,
                    "moustache": val_moustache,
                    "lunettes": val_lunettes,
                    "taille_cheveux": final_taille,
                    "couleur_cheveux": final_couleur
                })
            except Exception as e:
                print(f"Erreur sur {filepath.name}: {e}")

    # 2. Sauvegarde CSV
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    csv_path = OUTPUT_DIR / f"predictions_{prefix}.csv"
    
    # Création DataFrame avec ordre des colonnes FORCÉ
    cols_order = ["image_name", "barbe", "moustache", "lunettes", "taille_cheveux", "couleur_cheveux"]
    df = pd.DataFrame(results)
    
    # Sécurité si le DataFrame est vide
    if not df.empty:
        df = df[cols_order]
        df.to_csv(csv_path, index=False)
        print(f"✅ Sauvegardé : {csv_path}")
    else:
        print("⚠️ Aucune prédiction réussie.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="s8")
    args = parser.parse_args()
    run_prediction(prefix=args.prefix)