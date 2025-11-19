import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

#definir working directory
wd = os.path.dirname(os.path.abspath(__file__))

st.title('La durée d\'hospitalisation prédite')

#X_input = pd.DataFrame([{
#    "gender": data.gender,
#    "age": data.age,
#    "alcohol": data.alcohol,
#    "smoking": data.smoking,
#    "bmi": data.bmi,
#    "physical_activity": data.physical_activity,
#    "diet_score": data.diet_score,
#    "glucose": data.glucose,
#    "hba1c": data.hba1c,
#    "medical_condition": data.medical_condition
#}])

genre = ["Homme", "Femme"]
gender = st.segmented_control("Sexe", genre, selection_mode="single")
age = st.number_input("Quel est ton âge ?", value=5)
approve = ["oui", "non"]
alcohol = st.segmented_control("Consomme-tu de l'alcool ?", approve, selection_mode="single")
smoking = st.segmented_control("Es-tu fumeur ?", approve, selection_mode="single")
taille = st.number_input("Quelle est ta taille en cm ?", value=175)
poids = st.number_input("Quel est ton poids en kg ?", value=70)
physical_activity = st.number_input("Nombre d'heures d'activité physique par semaine ?", value=3)
diet_score = st.number_input("Score diététique (0-20) ?", value=10)
glucose = st.number_input("Taux de glucose (mg/dL) ?", value=100)
hba1c = st.number_input("HbA1c (%) ?", value=5)
maladie_options = ['Bonne santé', 'Diabète', 'Asthme', 'Obésité', 'Hypertension', 'Cancer', 'Arthrite', 'Non renseigné']
medical_condition = st.segmented_control(
    "As-tu des antécédents médicaux ?",maladie_options
    , selection_mode="single")

medical_options_map = {
    'Bonne santé': 'healthy',
    'Diabète': 'diabetes',
    'Asthme': 'asthma',
    'Obésité': 'obesity',
    'Hypertension': 'hypertension',
    'Cancer': 'cancer',
    'Arthrite': 'arthritis',
    'Non renseigné': 'Nan'
}

if medical_condition is None:
    medical_condition_code = None
else:
    medical_condition_code = medical_options_map[medical_condition]

# generate BMI
bmi = poids / ((taille/100) ** 2)

# build dataframe correctly
X_input = pd.DataFrame([{
    "gender": "male" if gender == "Homme" else "female",
    "age": np.floor(age),
    "alcohol": 1 if alcohol == "oui" else 0,
    "smoking": 1 if smoking == "oui" else 0,
    "bmi": bmi,
    "physical_activity": physical_activity,
    "diet_score": diet_score,
    "glucose": glucose,
    "hba1c": hba1c,
    "medical_condition": medical_condition_code
}])

print("X_input shape:", X_input.shape)
print(X_input)

# 1. load the trained model
with open(os.path.join(wd, 'model/best_model.pkl'), 'rb') as f:
    model = joblib.load(f)

#2. Load the preprocessor
with open(os.path.join(wd, 'model/preprocessor.pkl'), 'rb') as f:
    preprocessor = joblib.load(f)

if st.button('Prédire la durée d\'hospitalisation'):
    # Prétraitement (identique à l’entraînement)
    X_processed = preprocessor.transform(X_input)

    # Prédiction
    prediction = model.predict(X_processed)[0]

    st.success(f"La durée d'hospitalisation envisagée est de {round(int(prediction))} jours.")   