# Weather-Prediction-MLOps
Deployment of ML model into Production

## Local Climatological Data (LCD)

Dataset link - https://www.ncdc.noaa.gov/cdo-web/datatools/lcd <br>
Metadata link - https://www.ncei.noaa.gov/pub/data/cdo/documentation/LCD_documentation.pdf<br>
Range - Aug-2012 ~ July-2022<br>
County - Suffolk, Boston<br>


## Model

* MLflow - For experiemnt tracking local registry
* Prefect - For orchestration of retraining the model and promoting to model registry

Creating virtual environment
```bash
cd model
pipenv install -r requirments.txt
pipenv shell
```

Local MLflow Configuration
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Prefect Configuration
```bash
# Activate virtual env
prefect orion start
```

Prefect Deployment Configuration
```bash
# Activate virtual env

# Deployment-01
prefect deployment build train.py:main -n bos-pred-first-train -q dev
prefect deployment apply main-deployment.yaml

# Deployment-02
prefect deployment build retraining.py:main -n bos-pred-re-training -q dev
# Update the yaml file with cron scheduler
prefect deployment apply main-deployment.yaml

# Agent
prefect agent start -q 'dev'
```

Update the main-deployment.yaml for Deployment-02
```yaml
schedule:
  cron: 0 2 * * *
  timezone: America/Chicago
```

## WebApp

Model is deployed as a webapp using FastAPI


Creating virtual environment
```bash
cd webapp
pipenv install
pipenv shell
```

Run webapp
```bash
# Activate virtual env
# uvicorn predict:app --reload
gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8090 predict:app
```

Request Body
```json
{
  "HourlyDryBulbTemperature": 81,
  "LastHourDryBulbTemperature": 82,
  "HourlyPrecipitation": 0,
  "HourlyWindSpeed": 8
}
```

CURL Request
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "HourlyDryBulbTemperature": 81,
  "LastHourDryBulbTemperature": 82,
  "HourlyPrecipitation": 0,
  "HourlyWindSpeed": 8
}'
```

Response
```json
{
  "predicted": 81
}
```

## Unit Testing

pytest - unit testing for the predict prepare features function

```bash
# Activate virtual env
pytest test/test_unit.py
```
