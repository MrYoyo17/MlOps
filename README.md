```bash
CREATE DATABASE airflow_db;
CREATE DATABASE mlflow_db;
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO "user";
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO "user";
```

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