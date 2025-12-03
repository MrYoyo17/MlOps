## Lancement de l'app web :

```bash
docker load -i mlops-project-images.tar
docker compose -f render-compose.yml up
```

## Lancement en développement :

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