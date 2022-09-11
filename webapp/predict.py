import os
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir("../model")


class PredictInputsLocal(BaseModel):
    HourlyDryBulbTemperature:int
    LastHourDryBulbTemperature:int
    HourlyPrecipitation:float
    HourlyWindSpeed:int

# class ModelInProduction:
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_REGISTER_NAME = os.getenv("MODEL_REGISTER_NAME", "Boston-Temp-Predict")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(f"models:/{MODEL_REGISTER_NAME}/Production")


description = """
API endpoint to provide ML model as a service. 
## Users
You will be able to:
* **Inputs**
    * HourlyDryBulbTemperature:int
    * LastHourDryBulbTemperature:int
    * HourlyPrecipitation:float
    * HourlyWindSpeed:int
    
* Predict Boston's Weather Next Hour Temperature in F
"""

tags_metadata = [
    {
        "name": "weather",
        "description": "Predicts the boston's weather",
    },
]

app = FastAPI(  title="Boston Weather Prediction",
                description=description,)

def prepare_features(input):
    df = pd.DataFrame.from_dict([input])
    try:
        return True, df.drop('HourlyDryBulbTemperature', axis=1)
    except KeyError:
        return False
    

@app.post("/predict", tags=["Temperature"], status_code=status.HTTP_200_OK)
async def predict(request: PredictInputsLocal):
    X, df = prepare_features(request.dict())
    if not X:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Input is missing HourlyDryBulbTemperature key")
    try:
        y_pred = model.predict(df)
        return_v = {"predicted" : int(y_pred[0])}
        return return_v
    except:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Cannot predict the temperature")