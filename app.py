from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np



# 1. load the trained model
with open('model/xgb_best_model.pkl', 'rb') as f:
    model = joblib.load(f)

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
    hb1ac: float
    medical_condition: str

# 4. Define prediction endpoint
@app.post("/predict")
def predict_hospital_stay(data: PatientData):
    # Convert input data to numpy array
    X_input = np.array([[
        data.gender,
        data.age,
        data.alcohol,
        data.smoking,
        data.bmi,
        data.physical_activity,
        data.diet_score,
        data.glucose,
        data.hb1ac,
        data.medical_condition
    ]])
    #Make prediction
    prediction = model.predict(X_input)[0]

    return {"predicted_hospital_stay_days": round(prediction)}

# 5. Root endpoint
@app.get("/")
def root():
    return {"message": "Bienvenue à l'API de prédiction de la durée d'hospitalisation!"}    
