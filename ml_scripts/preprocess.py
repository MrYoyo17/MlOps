#!/usr/bin/env python3
import os
import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_BASE_DIR = os.path.join(os.path.dirname(__file__), "../data") # Relatif au script
BASE_DATA_DIR = os.environ.get("BASE_DATA_DIR", DEFAULT_BASE_DIR)
INPUT_DIR = BASE_DATA_DIR
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, 'dataset_prepro')
MAX_FILES = 0  # 0 = tous
TAILLE_CIBLE = (64, 64) # (largeur, hauteur)

# === Fonctions ===

def gather_image_files(input_dir, exts=('.png', '.jpg', '.jpeg')):
    files = []
    for root, _, filenames in os.walk(input_dir):
        for f in filenames:
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    return files

def process_image(path):
    """
    Fonction qui combine load, preprocess et finalize pour une image donnée.
    Retourne True si succès, False sinon.
    """
    
    # 1. LOAD
    img = cv2.imread(path)
    if img is None:
        return False

    # 2. PREPROCESS
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Seuillage (Fond blanc -> binaire inversé pour avoir l'objet en blanc)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Trouver les contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_final = None

    if len(contours) == 0:
        return False
    else:
        # Trouver le plus grand contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Obtenir la boîte englobante
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Rogner l'image
        img_cropped = img[y:y+h, x:x+w]

        # Sécurité : vérifier si le crop n'est pas vide
        if img_cropped.size == 0:
            return False

        # Redimensionner en 64x64 (sans proportionnalité)
        img_final = cv2.resize(img_cropped, TAILLE_CIBLE, interpolation=cv2.INTER_AREA)

    # 3. FINALIZE (Sauvegarde)
    if img_final is not None:
        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Construire le nom de fichier de sortie
        out_name = os.path.basename(path)
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        cv2.imwrite(out_path, img_final)
        return True
    
    return False

def run_preprocess():
    # Cette fonction sera appelée par Airflow
    print(f"--- Démarrage Pré-traitement ---")
    print(f"Source : {INPUT_DIR}")
    print(f"Destination : {OUTPUT_DIR}")
    
    files = gather_image_files(INPUT_DIR)
    success_count = 0
    # On évite tqdm dans Airflow pour ne pas pourrir les logs, on peut mettre une condition simple
    iterator = tqdm(files) if "airflow" not in os.environ.get("USER", "") else files
    
    for f in iterator:
        if process_image(f):
            success_count += 1
    print(f"Terminé : {success_count} images traitées.")

if __name__ == "__main__":
    run_preprocess()