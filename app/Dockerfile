FROM python:3.9-slim

USER root

# CORRECTION ICI : On utilise 'libgl1' au lieu de 'libgl1-mesa-glx'
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/app.py .
COPY data/predictions /app/predictions

ENV MLFLOW_TRACKING_URI=http://mlflow:5000
EXPOSE 5001

CMD ["python", "app.py"]