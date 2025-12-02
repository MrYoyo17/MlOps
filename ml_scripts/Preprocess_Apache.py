#!/usr/bin/env python3
from pyspark.sql import SparkSession
import os
import cv2
from PIL import Image
import numpy as np


# === Configuration (modifiable) ===
INPUT_DIR = 'dataset/'
OUTPUT_DIR = 'dataset_prepro/'
PARTITIONS = 4
MAX_FILES = 0  # 0 = tous
TAILLE_CIBLE = (64, 64) # (largeur, hauteur)

# === Initialisation Spark en local ===
spark = SparkSession.builder.master('local[*]').appName('PreprocessFaces').getOrCreate()
sc = spark.sparkContext

# === Fonctions de prétraitement ===
def gather_image_files(input_dir, exts=('.png')):
	files = []
	for root, _, filenames in os.walk(input_dir):
		for f in filenames:
			if f.lower().endswith(exts):
				files.append(os.path.join(root, f))
	return files


def load(path):
	# lit l'image et renvoie un dict minimal
	img = cv2.imread(path)
	if img is None:
		return None
	return {'path': path, 'img': img}



def preprocess(item):
	img = item['img']
	# Convertir en niveaux de gris
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Appliquer un seuillage pour obtenir une image binaire
    # Étant donné que le fond est blanc et l'objet est coloré,
    # on peut inverser le seuil pour que l'objet soit blanc sur fond noir.
    # On utilise cv2.THRESH_BINARY_INV pour un fond blanc (255) et un objet noir (0)
    # puis on inverse pour avoir l'objet blanc sur fond noir
	_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Trouver les contours
    # cv2.RETR_EXTERNAL ne récupère que les contours externes
    # cv2.CHAIN_APPROX_SIMPLE compresse les segments de lignes horizontaux, verticaux et diagonaux
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if len(contours) == 0:
		print("Aucun contour détecté. Assurez-vous que l'objet est bien contrasté par rapport au fond.")
	else:
		# Trouver le plus grand contour (qui devrait être notre visage)
		largest_contour = max(contours, key=cv2.contourArea)

		# Obtenir la boîte englobante (bounding box) de ce contour
		x, y, w, h = cv2.boundingRect(largest_contour)

		# --- Rogner l'image ---
		# On utilise les coordonnées obtenues pour trancher l'image
		img_cropped = img[y:y+h, x:x+w]

        # --- Redimensionner l'image rognée en 64x64 (sans garder la proportionnalité) ---
		img_final_resized = cv2.resize(img_cropped, TAILLE_CIBLE, interpolation=cv2.INTER_AREA)
		item['img'] = img_final_resized
	return item

def finalize(item):
	# sauvegarde l'image traitée et retourne le chemin
	# Si preprocess a déjà sauvegardé l'image, on retourne simplement le chemin
	if isinstance(item, dict) and 'out_path' in item:
		return item['out_path']

	# Ancien comportement (au cas où on reçoit une dict avec 'img' numpy)
	out_dir = OUTPUT_DIR
	os.makedirs(out_dir, exist_ok=True)
	out_name = os.path.basename(item['path'])
	out_path = os.path.join(out_dir, out_name)
	if 'img' in item and item['img'] is not None:
		cv2.imwrite(out_path, item['img'])
		return out_path

	return None


# === Pipeline Spark ===
files = gather_image_files(INPUT_DIR)
if MAX_FILES and MAX_FILES > 0:
	files = files[:MAX_FILES]

data = sc.parallelize(files, PARTITIONS)
result = data.map(load).filter(lambda x: x is not None).map(preprocess).filter(lambda x: x is not None).map(finalize).collect()

print(f"Processed {len(result)} images, saved to {OUTPUT_DIR}")

spark.stop()

