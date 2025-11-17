from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# 1. load the trained model
with open(os.path.join(wd, 'model/xgb_best_model.pkl'), 'rb') as f:
    model = joblib.load(f)

with open(os.path.join(wd, 'model/preprocessor.pkl'), 'rb') as f:
    preprocessor = joblib.load(f)

# 2. create FastAPI app
app = FastAPI(title = "Prédiction de la durée d'hospitalisation API")

# 3. Define request model
class PatientData(BaseModel):
    gender: str
    age: int
    alcohol: int
    smoking: int
    bmi: float
    physical_activity: float
    diet_score: float
    glucose: float
    hba1c: float
    medical_condition: str

# 4. Define prediction endpoint
@app.post("/predict")
def predict_hospital_stay(data: PatientData):
    # Convert input data to numpy array
    X_input = pd.DataFrame([{
    "gender": data.gender,
    "age": data.age,
    "alcohol": data.alcohol,
    "smoking": data.smoking,
    "bmi": data.bmi,
    "physical_activity": data.physical_activity,
    "diet_score": data.diet_score,
    "glucose": data.glucose,
    "hba1c": data.hba1c,
    "medical_condition": data.medical_condition
}])
    # Prétraitement (identique à l’entraînement)
    X_processed = preprocessor.transform(X_input)

    # Prédiction
    prediction = model.predict(X_processed)[0]

    return {"predicted_hospital_stay_days": round(float(prediction), 2)}

# 5. Root endpoint
@app.get("/")
def root():
    return {"message": "Bienvenue à l'API de prédiction de la durée d'hospitalisation!"}  