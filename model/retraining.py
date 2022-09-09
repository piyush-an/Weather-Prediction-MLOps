import numpy as np
import pandas as pd
import random
import mlflow
import os
import sys
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule, CronSchedule

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.environ['PREFECT_LOGGING_LEVEL'] = 'DEBUG'
# MLFlow Configuration

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
SET_EXPERIMENT = os.getenv("SET_EXPERIMENT", "bos-weather")
MODEL_REGISTER_NAME = os.getenv("MODEL_REGISTER_NAME", "Boston-Temp-Predict")
MODEL_REGISTER_TAGS = os.getenv("MODEL_REGISTER_TAGS", {"LOB": "MLOps Zoomcamp"})
MODEL_REGISTER_DESC = os.getenv("MODEL_REGISTER_DESC", "This is a Boston Weather Prediction Model")

# MLFlow Connection
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(SET_EXPERIMENT)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


# User Defined Function
@task(name="read_input_data")
def read_input_data(file_path):
    df = pd.read_csv(file_path)
    df['LastHourDryBulbTemperature'] = df['HourlyDryBulbTemperature'].shift(1)
    return df

@task(name="clean_df")
def clean_df(df):
    df = df[['DATE', 'HourlyDryBulbTemperature', 'LastHourDryBulbTemperature', 'HourlyPrecipitation','HourlyWindSpeed']]
    df.dropna(inplace=True)
    df = df[~df.isin(["*", "VRB", "T"]).any(axis=1)]
    df.reset_index(drop=True, inplace=True)
    df["DATE"] = pd.to_datetime(df['DATE']).dt.strftime('%Y-%m-%dT%H:%M:%S')
    df["HourlyDryBulbTemperature"] = df["HourlyDryBulbTemperature"].replace(r's$', '', regex=True).astype(int)
    df["LastHourDryBulbTemperature"] = df["LastHourDryBulbTemperature"].replace(r's$', '', regex=True).astype(int)
    df["HourlyPrecipitation"] = df["HourlyPrecipitation"].replace(r's$', '', regex=True).astype(float)
    df["HourlyWindSpeed"] = df["HourlyWindSpeed"].replace(r's$', '', regex=True).astype(int)
    return df


@task(name="random_sample")
def random_sample(df):
    rand_yr = random.randrange(2012, 2020)
    rand_month = random.randrange(1, 13)
    start_date = f"{rand_yr}-{rand_month}-01"
    end_date = f"{rand_yr + 2}-12-31"
    df = df.query(f"'{start_date}' <= DATE <= '{end_date}'")
    return df


@task(name="train_val_df_split")
def train_val_df_split(df):
    df.drop('DATE', axis=1, inplace=True)
    X = df.drop('HourlyDryBulbTemperature', axis=1)
    Y = df['HourlyDryBulbTemperature']
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
    
    return x_train, x_val, y_train, y_val


@task(name="test_df_split")
def test_df_split(df):
    # df.drop('DATE', axis=1, inplace=True)
    X = df.drop('HourlyDryBulbTemperature', axis=1)
    Y = df['HourlyDryBulbTemperature']
    return X, Y

@task(name="train_model")
def train_model(train, valid, y_val):
    booster = mlflow.pyfunc.load_model(f"models:/{MODEL_REGISTER_NAME}/Production")
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("developer", "Piyush")
            mlflow.set_tag("env", "dev")
            mlflow.set_tag("columns", "5")
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            nonlocal booster 
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, 'HourlyDryBulbTemperature')],
                early_stopping_rounds=50
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("RMSE", rmse)
        return {'loss': rmse, 'status': STATUS_OK}
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=25,
        trials=Trials()
    )
    return


@task(name="calculate_RMSE")
def calculate_RMSE(x_test, y_test, stage):
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_REGISTER_NAME}/{stage}")
    y_pred = model.predict(x_test)
    return mean_squared_error(y_test, y_pred, squared=False)


@task(name="find_para_best_exp")
def find_para_best_exp():
        runs = client.search_runs(
                        experiment_ids='1',
                        run_view_type=ViewType.ACTIVE_ONLY,
                        max_results=1,
                        order_by=["metrics.RMSE ASC"]
                        )
        return runs[0].data.params
    

@task(name="retrain_with_best_param")
def retrain_with_best_param(train, valid, y_val, params):
    with mlflow.start_run():
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("RMSE", rmse)
        mlflow.xgboost.log_model(booster, artifact_path="model")
        return mlflow.active_run().info.run_id


@task(name="push_to_model_register")
def push_to_model_register(run_id):
    model_uri = f"runs:/{run_id}/model"
    output = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTER_NAME)
    return output.run_id, output.name, output.version


@task(name="move_model_to_staging")
def move_model_to_staging(stagged_model_name, stagged_model_version):
    latest_versions = client.get_latest_versions(name=MODEL_REGISTER_NAME)
    client.transition_model_version_stage(
        name=stagged_model_name,
        version=stagged_model_version,
        stage='Staging',
        archive_existing_versions=True
    )
    output = client.update_model_version(
        name=latest_versions[-1].name,
        version=latest_versions[-1].version,
        description=f"The model version {latest_versions[-1].version} was transitioned to Staging on {datetime.today().date()}"
    )
    return output


@task(name="move_staging_to_production")
def move_staging_to_production(model_version):
    return client.transition_model_version_stage(
        name=MODEL_REGISTER_NAME,
        version=model_version,
        stage="Production",
        archive_existing_versions=True
    )

@task(name="retrain_model")
def retrain_model(train, valid, y_val, stage: str = "Production"):
    booster = mlflow.pyfunc.load_model(f"models:/{MODEL_REGISTER_NAME}/Production")
    params = (client.get_run(booster.metadata.run_id)).data.params
    with mlflow.start_run():
        mlflow.set_tag("developer", "Piyush")
        mlflow.set_tag("env", "dev")
        mlflow.set_tag("columns", "5")
        mlflow.set_tag("model", "xgboost")
        booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, 'HourlyDryBulbTemperature')],
                early_stopping_rounds=50
                )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("RMSE", rmse)
        mlflow.xgboost.log_model(booster, artifact_path="model")
        return mlflow.active_run().info.run_id, rmse
    

# MLflow Flow
@flow(task_runner=SequentialTaskRunner())
def main():
    # logger_task()
    logger = get_run_logger()
    test_data_file_path = r"data/last_decade.csv"
    df = read_input_data(test_data_file_path)
    df = clean_df(df)
    df2 = random_sample(df)
    x_train, x_val, y_train, y_val = train_val_df_split(df2)
    train = xgb.DMatrix(x_train, label=y_train)
    valid = xgb.DMatrix(x_val, label=y_val)
    candidate_model_run_id, candidate_model_rmse = retrain_model(train, valid, y_val)
    stagged_model_run_id, stagged_model_name, stagged_model_version = push_to_model_register(candidate_model_run_id)
    move_model_to_staging(stagged_model_name, stagged_model_version)
    x_test, y_test = test_df_split(df2)
    RMSE_Production = calculate_RMSE(x_test, y_test, "Production")
    logger.info(f"Comparing RMSE Stagged {candidate_model_rmse} to Production {RMSE_Production}")
    if RMSE_Production > candidate_model_rmse:
        logger.info("Staged model to be moved to Production")
        move_staging_to_production(stagged_model_version)