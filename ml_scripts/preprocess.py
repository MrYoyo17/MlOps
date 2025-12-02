import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# === GESTION INTELLIGENTE DES CHEMINS ===
DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_DATA_DIR = Path(os.environ.get("BASE_DATA_DIR", DEFAULT_BASE_DIR))

INPUT_DIR = BASE_DATA_DIR
OUTPUT_DIR = BASE_DATA_DIR / 'dataset_prepro'
TAILLE_CIBLE = (64, 64)

def gather_image_files(input_dir, exts=('.png', '.jpg', '.jpeg')):
    input_dir = Path(input_dir)
    files = []
    if not input_dir.exists():
        print(f"ATTENTION: Le dossier {input_dir} n'existe pas.")
        return []
        
    for f in input_dir.rglob("*"):
        if f.suffix.lower() in exts:
            files.append(f)
    return files

def process_single_image(path):
    """
    Retourne :
     1 : Succès
     0 : Échec
    -1 : Skipped
    """
    # 1. On s'assure que tout est bien un objet Path
    path_obj = Path(path)
    out_name = path_obj.name
    
    # Sécurité : on force OUTPUT_DIR en Path aussi
    out_path = Path(OUTPUT_DIR) / out_name

    # --- VÉRIFICATION D'EXISTENCE (CORRECTION ICI) ---
    if out_path.exists():
        return -1

    # --- LOAD ---
    # cv2 a besoin d'une string, pas d'un Path
    img = cv2.imread(str(path_obj))
    if img is None:
        return 0

    # --- PREPROCESS ---
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_final = None

        if len(contours) == 0:
            return 0
        else:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            img_cropped = img[y:y+h, x:x+w]

            if img_cropped.size == 0:
                return 0

            img_final = cv2.resize(img_cropped, TAILLE_CIBLE, interpolation=cv2.INTER_AREA)

        # --- FINALIZE ---
        if img_final is not None:
            # On s'assure que le dossier parent existe
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(out_path), img_final)
            return 1
            
    except Exception as e:
        print(f"Erreur sur {out_name}: {e}")
        return 0
    
    return 0

def run_preprocess():
    print(f"--- Démarrage Pré-traitement ---")
    print(f"Source : {INPUT_DIR}")
    print(f"Destination : {OUTPUT_DIR}")
    
    files = gather_image_files(INPUT_DIR)
    print(f"{len(files)} images trouvées dans le dossier source.")

    count_success = 0
    count_skipped = 0
    count_error = 0
    
    # Désactiver tqdm dans Airflow
    disable_tqdm = os.environ.get("AIRFLOW_CTX_DAG_ID") is not None
    iterator = tqdm(files, desc="Traitement", unit="img", disable=disable_tqdm)
    
    for f in iterator:
        res = process_single_image(f)
        if res == 1:
            count_success += 1
        elif res == -1:
            count_skipped += 1
        else:
            count_error += 1
            
    print(f"\n--- Bilan ---")
    print(f"✅ Traitées  : {count_success}")
    print(f"⏩ Ignorées  : {count_skipped} (Déjà existantes)")
    print(f"❌ Erreurs   : {count_error}")
    print(f"----------------")

if __name__ == "__main__":
    run_preprocess()