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