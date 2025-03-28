from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import xgboost as xgb
import joblib

model = xgb.Booster()
model.load_model("models/model.json")

scaler = joblib.load("models/scaler.pkl")

app = FastAPI(title="Prediction API")

class Features(BaseModel):
    feature_values: list[float]

@app.get("/")
def home():
    return {"message": "Bankruptcy Prediction API"}

@app.post("/predict")
def predict_bankruptcy(data: Features):
    try:

        input_data = np.array(data.feature_values).reshape(1, -1)

        expected_features = scaler.n_features_in_
        if input_data.shape[1] < expected_features:
            padding = np.zeros((1, expected_features - input_data.shape[1]))
            input_data = np.hstack((input_data, padding))

        input_scaled = scaler.transform(input_data)

        input_dmatrix = xgb.DMatrix(input_scaled)

        bankruptcy_proba = model.predict(input_dmatrix)[0]

        return {"bankruptcy_probability": float(bankruptcy_proba)}

    except Exception as e:
        return {"error": str(e)}
