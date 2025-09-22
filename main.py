# from fastapi import FastAPI
# from pydantic import BaseModel
# import numpy as np
# import joblib

# from prometheus_fastapi_instrumentator import Instrumentator

# # Load model and scaler
# model = joblib.load("liver_model.pkl")
# _, _, _, _, scaler = joblib.load("data.pkl")  # Unpack tuple, use only scaler

# app = FastAPI()

# #  instrumentor for metrics

# instrumentator = Instrumentator()
# instrumentator.instrument(app).expose(app)

# class PatientData(BaseModel):
#     Age: float
#     Gender: int
#     Total_Bilirubin: float
#     Direct_Bilirubin: float
#     Alkaline_Phosphatase: float
#     Alanine_Aminotransferase: float
#     Aspartate_Aminotransferase: float
#     Total_Proteins: float
#     Albumin: float
#     Albumin_Globulin_Ratio: float

# @app.get("/")
# def read_root():
#     return {"message": "Liver Disease prediction API is up!"}

# @app.post("/predict")
# def predict(data: PatientData):
#     # Prepare the input features as a 2D array
#     features = np.array([[data.Age, data.Gender, data.Total_Bilirubin, data.Direct_Bilirubin,
#                           data.Alkaline_Phosphatase, data.Alanine_Aminotransferase,
#                           data.Aspartate_Aminotransferase, data.Total_Proteins,
#                           data.Albumin, data.Albumin_Globulin_Ratio]])
    
#     # Scale the features using the loaded scaler
#     features = scaler.transform(features)
    
#     # Make prediction using the model
#     prediction = model.predict(features)
    
#     return {"prediction": int(prediction[0])}



import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import joblib
import traceback

# MLflow tracking URI will be provided by env var in k8s
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow = mlflow  # ensure imported
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

# Model URI in model registry
MODEL_URI = os.getenv("MODEL_URI", "models:/MyLiverModel/Production")

from prometheus_fastapi_instrumentator import Instrumentator

# Load model and scaler
model = joblib.load("liver_model.pkl")
_, _, _, _, scaler = joblib.load("data.pkl")  # Unpack tuple, use only scaler

app = FastAPI()

#  instrumentor for metrics

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Try to load model via MLflow registry
model = None
load_error = None
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"[INFO] Loaded model from MLflow URI: {MODEL_URI}")
except Exception as e:
    load_error = str(e)
    print("[WARN] Could not load model from MLflow registry:", load_error)
    # fallback: try to load local model
    try:
        model = joblib.load("liver_model.pkl")
        print("[INFO] Loaded fallback local model at liver_model.pkl")
    except Exception as e2:
        print("[ERROR] Cannot load fallback model:", str(e2))
        model = None

class PatientData(BaseModel):
    Age: float
    Gender: int
    Total_Bilirubin: float
    Direct_Bilirubin: float
    Alkaline_Phosphatase: float
    Alanine_Aminotransferase: float
    Aspartate_Aminotransferase: float
    Total_Proteins: float
    Albumin: float
    Albumin_Globulin_Ratio: float

@app.get("/")
def root():
    return {"message": "Liver Disease Prediction (MLflow model loader)"}

@app.get("/health")
def health():
    return {"model_loaded": model is not None, "mlflow_uri": MLFLOW_TRACKING_URI, "model_uri": MODEL_URI}

@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        df = pd.DataFrame([data.dict()])
        preds = model.predict(df)
        return {"prediction": int(preds[0])}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
