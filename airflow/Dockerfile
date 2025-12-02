FROM apache/airflow:2.7.1

USER root

# Installation des libs système (version corrigée 'libgl1')
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# ÉTAPE 1 : On installe Torch CPU depuis le dépôt PyTorch
# Cela évite de télécharger la version GPU qui est énorme
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ÉTAPE 2 : On installe les autres librairies depuis le PyPI standard
# (On a retiré l'index-url ici pour qu'il trouve mlflow)
RUN pip install --no-cache-dir \
    mlflow==2.7.1 \
    pandas \
    scikit-learn \
    pillow \
    numpy \
    psycopg2-binary \
    opencv-python-headless \
    tqdm