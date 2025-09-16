from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("main/models/liver_model.pkl")
_, _, _, _, scaler = joblib.load("main/models/data.pkl")  # Unpack tuple, use only scaler

app = FastAPI()

class PatientData(BaseModel):
    Age: float
    Total_Bilirubin: float
    Direct_Bilirubin: float
    Alkaline_Phosphatase: float
    Alanine_Aminotransferase: float
    Aspartate_Aminotransferase: float
    Total_Proteins: float
    Albumin: float
    Albumin_Globulin_Ratio: float
    Gender: int

@app.get("/")
def read_root():
    return {"message": "Liver Disease prediction API is up!"}

@app.post("/predict")
def predict(data: PatientData):
    features = np.array([[data.Age, data.Total_Bilirubin, data.Direct_Bilirubin,
                          data.Alkaline_Phosphatase, data.Alanine_Aminotransferase,
                          data.Aspartate_Aminotransferase, data.Total_Proteins,
                          data.Albumin, data.Albumin_Globulin_Ratio, data.Gender]])
    
    # Scale the features using the loaded scaler
    features = scaler.transform(features)
    
    # Make prediction using the model
    prediction = model.predict(features)
    
    return {"prediction": int(prediction[0])}
