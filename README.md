# Projet MlOps groupe 2 : Eliott SUBLIN, Camille Pelé, Florian Marchive

## Lancement de l'app web :

Se placer dans le répertoire racine où se trouve le dossier `data/`.

Une fois le `.tar` téléchargé, on peut charger les images des conteneurs dans Docker :
```bash
docker load -i mlops-project-images.tar
```

On démarre un conteneur contenant l'API en exposant le port `5002`et en liant notre répertoire `data`à celui du conteneur :

```bash
docker run -d \
  --name api-groupe2 \
  -p 5002:5001 \
  -v "./data":/data \
  mlops-face-filter-app
```

On démarre un conteneur contenant le front en exposant le port `5173`:

```bash
docker run -d \
  --name front-groupe2 \
  -p 5173:5173 \
  -e VITE_API_URL='http://localhost:5002' \
  mlops-front
```

On ajoute toutes les images dans le dossier `data/` ainsi que le fichier csv `prediction_s.csv` de prediction de toutes les images que nous avons généré dans le dossier `data/prediction/`.

Pour terminer, on profite de cette belle interface Web ! ``http://localhost:5173``

## Lancement en développement :

Réaliser un clone du dépôt GitHub : 
https://github.com/MrYoyo17/MlOps.git

Pour lancer le projet :
```bash
cd MlOps
docker compose up -d
```
Cette commande peut prendre du temps puisque Docker doit créer toutes les images des conteneurs.

Une fois lancé, il faut recréer le compte admin de connexion à airflow avec les deux commandes suivantes :
```bash
docker compose exec airflow airflow users delete --username admin
```

```bash
docker compose exec airflow airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```
Ensuite, redémarrer Airflow :
```bash
docker compose restart airflow
```

Si airflow génère une erreur, c'est sans doute à cause de la librairie evidently. Effectuer la commande :

```bash
docker compose exec airflow python -m pip install evidently==0.5.0
```

Créer un répertoire `predictions` dans le dossier `data/`.
Mettre les images et les csv associés dans le dossier `data/`.

Exécuter le script de preprocessing des images directement depuis l'ordinateur permet de gagner beaucoup de temps pour éviter que ce soit le conteneur airflow qui le fasse (c'est exactement le même script qui est utilisé): 

```bash
python ml_scripts/preprocess.py
```

### Configuration airflow

Ajouter les variables suivantes (dans le volet `Admin`puis `Variables`) :
 - `csv_count_memory` = 0 : cela va lancer l'entraînement du modèle.
 - `processed_batches_list`= `s7,s8,s1,s9,s4,s3,s6,s2,s5`: Si une image commence par le préfix `sx`et qu'il n'est pas présent dans cette liste, alors airflow lancera une prediction sur les images de ce lot.