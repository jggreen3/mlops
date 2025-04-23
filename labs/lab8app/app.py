from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel
import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:///../data/mlflow.db")

app = FastAPI(
    title="Phone Price Classifier",
    description="Given phone specs, classify the price group of the phone as an integer 0-3",
    version="0.1",
)


# Defining path operation for root endpoint
@app.get("/")
def main():
    return {"message": "This is a model for classifying phone prices"}


class request_body(BaseModel):
    ram: int
    battery_power: int
    px_height: int
    px_width: int


@app.on_event("startup")
def load_artifacts():
    mod_path = f"runs:/9b5f17894a9640bfa4bb336943b125d0/best_model"
    global model
    model = mlflow.pyfunc.load_model(mod_path)


# Defining path operation for /predict endpoint
@app.post("/predict")
def predict(data: request_body):
    X = pd.DataFrame([data.model_dump()])
    predictions = model.predict(X)
    return {"price_range": int(predictions[0])}
