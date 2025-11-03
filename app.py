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
    Gender: str
    Age: int
    Alcohol: int
    Smoking: int
    BMI: float
    Physical_Activity: float
    Diet_Score: float
    Glucose: float
    HbA1c: float
    Medical_Condition: str

# 4. Define prediction endpoint
@app.post("/predict")
def predict_hospital_stay(data: PatientData):
    # Convert input data to numpy array
    X_input = np.array([[
        data.Gender,
        data.Age,
        data.Alcohol,
        data.Smoking,
        data.BMI,
        data.Physical_Activity,
        data.Diet_Score,
        data.Glucose,
        data.HbA1c,
        data.Medical_Condition
    ]])
    #Make prediction
    prediction = model.predict(X_input)[0]

    return {"predicted_hospital_stay_days": round(prediction)}

# 5. Root endpoint
@app.get("/")
def root():
    return {"message": "Bienvenue à l'API de prédiction de la durée d'hospitalisation!"}    