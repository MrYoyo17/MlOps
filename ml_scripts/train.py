import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import cv2
import os
import numpy as np
print('numpy OK, version =', np.__version__, 'from', np.__file__)
from PIL import Image
from tqdm import tqdm
import mlflow
import mlflow.pytorch  # Important pour sauvegarder les modèles PyTorch


DEFAULT_BASE_DIR = os.path.join(os.path.dirname(__file__), "../data")
BASE_DATA_DIR = os.environ.get("BASE_DATA_DIR", DEFAULT_BASE_DIR)

CSV_PATH = os.path.join(BASE_DATA_DIR, "csv_global.csv")
IMG_DIR = os.path.join(BASE_DATA_DIR, "dataset_prepro")

print(f"Loading CSV from: {CSV_PATH}")
print(f"Loading Images from: {IMG_DIR}")

# --- 1. CONFIGURATION MLFLOW ---
# On récupère l'URI via variable d'env (défini dans docker-compose) ou on met un défaut local
DEFAULT_MLFLOW_URI = "http://localhost:5050"
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Image_Filter_Project")

class DatasetFaces(Dataset):   
    def __init__(self, csv_file, img_dir, transform=None, file_extension='.jpg'):
        self.annotations = pd.read_csv(csv_file, index_col='filename')
        self.img_names = self.annotations.index.tolist()
        self.img_dir = img_dir
        self.transform = transform
        self.file_extension = file_extension

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name_base = self.img_names[idx]
        img_name_with_ext = img_name_base + self.file_extension
        img_path = os.path.join(self.img_dir, img_name_with_ext)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Erreur : Image non trouvée à {img_path}")
            return None, None 

        labels = self.annotations.loc[img_name_base].values
        labels = torch.tensor(labels.astype(float), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels
    
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
        self.common_fc = nn.Sequential(
            nn.Linear(32*16*16, 256), nn.ReLU(),
        )
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
        out_barbe = self.classifier_barbe(common_features)
        out_moustache = self.classifier_moustache(common_features)
        out_lunettes = self.classifier_lunettes(common_features)
        cheveux_features = self.classifier_cheveux_features(common_features)
        out_taille_cheveux = self.classifier_taille_cheveux(cheveux_features)
        out_couleur_cheveux = self.classifier_couleur_cheveux(cheveux_features)
        return [out_barbe, out_moustache, out_lunettes, out_taille_cheveux, out_couleur_cheveux]
    
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

full_dataset = DatasetFaces(csv_file=CSV_PATH,
                              img_dir=IMG_DIR,
                              transform=data_transform,
                              file_extension=".png")

def run_training():
    print(f"--- Démarrage Entraînement ---")
    print(f"Données : {IMG_DIR}")
    print(f"Tracking MLflow : {MLFLOW_URI}")
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    test_size = total_size - train_size

    print(f"Total: {total_size} | Train: {train_size} | Test: {test_size}")

    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)

    # Hyperparamètres (sortis en variables pour le logging)
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device) 
    criterion_binaire = nn.BCEWithLogitsLoss().to(device)
    criterion_multiclasse = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 2. DÉMARRAGE DU RUN MLFLOW ---
    with mlflow.start_run() as run:
        
        # A. Logger les Hyperparamètres
        params = {
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "device": str(device)
        }
        mlflow.log_params(params)
        print("Paramètres loggés dans MLflow.")

        # --- Boucle d'entraînement ---
        for epoch in range(NUM_EPOCHS):
            print(f"\n{'='*10} Epoch {epoch+1}/{NUM_EPOCHS} {'='*10}")
            
            model.train()
            epoch_loss = 0.0
            total_samples = 0
            
            correct_barbe = 0
            correct_moustache = 0
            correct_lunettes = 0
            correct_taille = 0
            correct_couleur = 0

            progress_bar = tqdm(train_loader, desc="Entraînement", unit="batch")

            for images, all_labels_batch in progress_bar:
                images = images.to(device)
                all_labels_batch = all_labels_batch.to(device)
                
                outputs = model(images)
                out_barbe, out_moustache, out_lunettes, out_taille, out_couleur = outputs
                
                lab_barbe = all_labels_batch[:, 0].float()
                lab_moustache = all_labels_batch[:, 1].float()
                lab_lunettes = all_labels_batch[:, 2].float()
                lab_taille = torch.argmax(all_labels_batch[:, 3:6], dim=1)
                lab_couleur = torch.argmax(all_labels_batch[:, 6:11], dim=1)

                loss_barbe = criterion_binaire(out_barbe.squeeze(), lab_barbe)
                loss_moustache = criterion_binaire(out_moustache.squeeze(), lab_moustache)
                loss_lunettes = criterion_binaire(out_lunettes.squeeze(), lab_lunettes)
                loss_taille = criterion_multiclasse(out_taille, lab_taille)
                loss_couleur = criterion_multiclasse(out_couleur, lab_couleur)
                
                total_loss = loss_barbe + loss_moustache + loss_lunettes + loss_taille + loss_couleur
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_size = images.size(0)
                total_samples += batch_size
                
                with torch.no_grad():
                    pred_barbe = (torch.sigmoid(out_barbe.squeeze()) > 0.5).float()
                    correct_barbe += (pred_barbe == lab_barbe).sum().item()
                    
                    pred_moustache = (torch.sigmoid(out_moustache.squeeze()) > 0.5).float()
                    correct_moustache += (pred_moustache == lab_moustache).sum().item()
                    
                    pred_lunettes = (torch.sigmoid(out_lunettes.squeeze()) > 0.5).float()
                    correct_lunettes += (pred_lunettes == lab_lunettes).sum().item()
                    
                    pred_taille = torch.argmax(out_taille, dim=1)
                    correct_taille += (pred_taille == lab_taille).sum().item()
                    
                    pred_couleur = torch.argmax(out_couleur, dim=1)
                    correct_couleur += (pred_couleur == lab_couleur).sum().item()

                epoch_loss += total_loss.item()
                
                curr_acc_taille = (pred_taille == lab_taille).sum().item() / batch_size
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.3f}", 
                    'acc_taille': f"{curr_acc_taille:.0%}"
                })
            
            avg_loss = epoch_loss / len(train_loader)
            acc_barbe = correct_barbe / total_samples
            acc_moustache = correct_moustache / total_samples
            acc_lunettes = correct_lunettes / total_samples
            acc_taille = correct_taille / total_samples
            acc_couleur = correct_couleur / total_samples

            print(f"\n📊 Fin de l'Epoch {epoch+1}")
            print(f"   📉 Loss Moyenne : {avg_loss:.4f}")
            print(f"   🧔 Barbe Acc    : {acc_barbe:.2%}")

            # B. Logger les métriques pour cette epoch dans MLflow
            # 'step' permet d'avoir de beaux graphiques temporels
            metrics = {
                "train_loss": avg_loss,
                "acc_barbe": acc_barbe,
                "acc_moustache": acc_moustache,
                "acc_lunettes": acc_lunettes,
                "acc_taille": acc_taille,
                "acc_couleur": acc_couleur
            }
            mlflow.log_metrics(metrics, step=epoch)

        # C. Sauvegarder le modèle final dans MLflow
        print("Sauvegarde du modèle dans MLflow...")
        # On loggue le modèle complet. 
        # Pour un déploiement plus simple, on peut aussi utiliser log_state_dict mais log_model est plus direct.
        log_info = mlflow.pytorch.log_model(model, "model")
        print(f"Entraînement terminé. Run ID: {run.info.run_id}")

        model_uri = log_info.model_uri
        print(f"Artefacts sauvegardés à : {model_uri}")

        # 2. Ensuite, on force l'enregistrement dans le Registry
        model_name = "FaceFilterModel"
        print(f"Enregistrement dans le Model Registry sous le nom '{model_name}'...")
        
        try:
            result = mlflow.register_model(model_uri, model_name)
            print(f"✅ Modèle enregistré avec succès ! Version : {result.version}")
        except Exception as e:
            print(f"⚠️ Erreur lors de l'enregistrement dans le registre : {e}")

if __name__ == "__main__":
    run_training()