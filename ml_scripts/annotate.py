import pandas as pd
from pathlib import Path
import argparse
import sqlite3
from tqdm import tqdm
import os
import sys

# --- GESTION INTELLIGENTE DES CHEMINS (Mac vs Docker) ---
# Si on est dans Docker (Airflow), cette variable sera définie
DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_DATA_DIR = Path(os.environ.get("BASE_DATA_DIR", DEFAULT_BASE_DIR))

# Valeurs par défaut relatives au BASE_DATA_DIR
DOSSIER_DEFAULT = BASE_DATA_DIR
OUTPUT_DIR_DEFAULT = BASE_DATA_DIR
OUTPUT_BASENAME_DEFAULT = "csv_global"

# --- CONSTANTES ---
ATTRIBUTS = ["facial_hair", "glasses", "hair", "hair_color"]

# facial_hair
FACIAL_HAIR_BARBE = {0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13}
FACIAL_HAIR_MOUSTACHE = {0, 2, 3, 5, 6, 7, 8, 9, 10, 11}

def map_barbe(v):
    if pd.isna(v): return pd.NA
    return 1 if int(v) in FACIAL_HAIR_BARBE else 0

def map_moustache(v):
    if pd.isna(v): return pd.NA
    return 1 if int(v) in FACIAL_HAIR_MOUSTACHE else 0

# glasses
GLASSES_NO = {11}
GLASSES_YES = set(range(0, 11))

def map_glasses_bool(v):
    if pd.isna(v): return pd.NA
    v = int(v)
    if v in GLASSES_YES: return 1
    if v in GLASSES_NO: return 0
    return pd.NA

# hair
HAIR_BALD = {0, 1, 2, 108, 109, 110}
HAIR_SHORT = {8, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 31, 32, 39, 43, 44, 45, 46, 49, 50, 51, 52, 53, 54, 61, 62, 66, 72, 73, 74, 76, 77, 83, 89, 92, 99, 105, 107}
HAIR_LONG = {3, 4, 5, 6, 7, 9, 10, 13, 15, 16, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38, 40, 41, 42, 47, 48, 55, 56, 57, 58, 59, 60, 63, 64, 65, 67, 68, 69, 70, 71, 75, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 93, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 106}
ALL_HAIR = HAIR_BALD | HAIR_SHORT | HAIR_LONG

def map_hair_long(v):
    if pd.isna(v): return pd.NA
    v = int(v)
    return 1 if v in HAIR_LONG else (0 if v in ALL_HAIR else pd.NA)

def map_hair_short(v):
    if pd.isna(v): return pd.NA
    v = int(v)
    return 1 if v in HAIR_SHORT else (0 if v in ALL_HAIR else pd.NA)

def map_hair_bald(v):
    if pd.isna(v): return pd.NA
    v = int(v)
    return 1 if v in HAIR_BALD else (0 if v in ALL_HAIR else pd.NA)

# hair_color
HAIR_BLOND = {0, 1, 4}
HAIR_CHA = {3, 5, 6}
HAIR_RED = {2}
HAIR_BROWN = {7}
HAIR_GREY = {8, 9}
ALL_HAIR_COLOR = HAIR_BLOND | HAIR_CHA | HAIR_RED | HAIR_BROWN | HAIR_GREY

def map_blonds(v):
    if pd.isna(v): return pd.NA
    v = int(v)
    return 1 if v in HAIR_BLOND else (0 if v in ALL_HAIR_COLOR else pd.NA)

def map_chatains(v):
    if pd.isna(v): return pd.NA
    v = int(v)
    return 1 if v in HAIR_CHA else (0 if v in ALL_HAIR_COLOR else pd.NA)

def map_roux(v):
    if pd.isna(v): return pd.NA
    v = int(v)
    return 1 if v in HAIR_RED else (0 if v in ALL_HAIR_COLOR else pd.NA)

def map_brun(v):
    if pd.isna(v): return pd.NA
    v = int(v)
    return 1 if v in HAIR_BROWN else (0 if v in ALL_HAIR_COLOR else pd.NA)

def map_gris_bleu(v):
    if pd.isna(v): return pd.NA
    v = int(v)
    return 1 if v in HAIR_GREY else (0 if v in ALL_HAIR_COLOR else pd.NA)


# ---------- LOGIQUE MÉTIER ----------

def process_files(dossier: Path, csv_output_name: str) -> pd.DataFrame:
    """Traite les fichiers CSV du dossier et retourne un DataFrame global."""
    
    # Conversion en Path si ce n'est pas déjà le cas
    dossier = Path(dossier)

    files_to_process = [
        csv_path for csv_path in dossier.glob("*.csv")
        if csv_path.name != csv_output_name
    ]

    if not files_to_process:
        print(f"Aucun fichier CSV à traiter dans {dossier}.")
        return pd.DataFrame()
    
    lignes = []

    print(f"Traitement de {len(files_to_process)} fichiers CSV...")
    
    # DÉTECTION AIRFLOW : Si on est dans Airflow, on désactive tqdm pour éviter de polluer les logs
    disable_tqdm = os.environ.get("AIRFLOW_CTX_DAG_ID") is not None
    
    iterator = tqdm(files_to_process, desc="🔄 Traitement", unit="fichier", disable=disable_tqdm)
    
    for csv_path in iterator:
        try:
            df = pd.read_csv(csv_path, header=None, names=["C1", "C2", "C3"], dtype=str)
        except pd.errors.EmptyDataError:
            continue 

        if df.empty:
            continue

        df["C2"] = pd.to_numeric(df["C2"], errors="coerce")
        sous_df = df[df["C1"].isin(ATTRIBUTS)].set_index("C1")["C2"]

        ligne = {"filename": csv_path.stem}
        
        # Récupération valeurs
        f_hair = sous_df.get("facial_hair", pd.NA)
        glasses = sous_df.get("glasses", pd.NA)
        hair = sous_df.get("hair", pd.NA)
        h_color = sous_df.get("hair_color", pd.NA)

        # Application mappings
        ligne.update({
            "barbe": map_barbe(f_hair),
            "moustache": map_moustache(f_hair),
            "lunette": map_glasses_bool(glasses),
            "long": map_hair_long(hair),
            "court": map_hair_short(hair),
            "chauve": map_hair_bald(hair),
            "blonds": map_blonds(h_color),
            "chatains": map_chatains(h_color),
            "roux": map_roux(h_color),
            "brun": map_brun(h_color),
            "gris_bleu": map_gris_bleu(h_color)
        })

        lignes.append(ligne)

    global_df = pd.DataFrame(lignes)

    # Colonnes finales garanties
    colonnes_finales = [
        "filename", "barbe", "moustache", "lunette",
        "long", "court", "chauve",
        "blonds", "chatains", "roux", "brun", "gris_bleu",
    ]
    for col in colonnes_finales:
        if col not in global_df.columns:
            global_df[col] = pd.NA
            
    return global_df[colonnes_finales]


# --- FONCTION PRINCIPALE (Appelable par Airflow ou Main) ---

def run_aggregation(input_dir=None, output_dir=None, output_basename=None, output_format="csv"):
    """
    Fonction wrapper pour être appelée depuis un autre script Python (ex: Airflow).
    Si les arguments sont None, utilise les défauts basés sur BASE_DATA_DIR.
    """
    # 1. Résolution des chemins
    input_dir = Path(input_dir) if input_dir else DOSSIER_DEFAULT
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR_DEFAULT
    basename = output_basename if output_basename else OUTPUT_BASENAME_DEFAULT

    # Création dossier sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_csv = output_dir / f"{basename}.csv"
    file_db = output_dir / f"{basename}.db"

    print(f"--- Démarrage Aggrégation ---")
    print(f"Source : {input_dir}")
    print(f"Destination : {output_dir}")

    # 2. Traitement
    df_final = process_files(input_dir, file_csv.name)

    if df_final.empty:
        print("Résultat vide. Arrêt.")
        return

    # 3. Sauvegarde CSV
    if output_format in ["csv", "both", None]:
        try:
            df_final.to_csv(file_csv, index=False)
            print(f"✅ CSV sauvegardé : {file_csv}")
        except Exception as e:
            print(f"❌ Erreur CSV : {e}")

    # 4. Sauvegarde SQLite
    if output_format in ["sqlite", "both", None]:
        try:
            conn = sqlite3.connect(str(file_db))
            table_name = "data"
            df_final.to_sql(table_name, conn, if_exists="replace", index=False)
            conn.close()
            print(f"✅ SQLite sauvegardé : {file_db}")
        except Exception as e:
            print(f"❌ Erreur SQLite : {e}")


# --- CLI (Lancement manuel) ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dossier", type=str, help="Dossier source")
    parser.add_argument("--output-dir", type=str, help="Dossier de sortie")
    parser.add_argument("--base-name", type=str, default=OUTPUT_BASENAME_DEFAULT)
    parser.add_argument("--format", choices=["csv", "sqlite", "both"], default="both")
    
    args = parser.parse_args()

    # Si arguments fournis en ligne de commande, ils sont prioritaires.
    # Sinon run_aggregation utilisera les défauts.
    run_aggregation(
        input_dir=args.dossier,
        output_dir=args.output_dir,
        output_basename=args.base_name,
        output_format=args.format
    )
