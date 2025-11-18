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

X_input = pd.DataFrame()

X_input["gender"] = st.selectbox(
    "Quel est ton genre ?", options=["male", "female"]
)

X_input['age'] = st.number_input(
    "Quel est ton âge ?", value=None, placeholder="Type a number..."
)
X_input['age'] = np.floor(X_input['age'])

X_input['alcohol'] = st.selectbox(
    "Es-tu alcoolique ?", options=["oui", "non"]
)

if X_input.loc[0,'alcohol']== "oui":
    X_input.loc[0,'alcohol'] = 1
else:
    X_input.loc[0,'alcohol'] = 0

X_input['smoking'] = st.selectbox(
    "Es-tu fumeur ?", options=["oui", "non"]
)
if X_input['smoking'][0] == "oui":
    X_input['smoking'][0] = 1
else:
    X_input['smoking'][0] = 0

taille = st.number_input(
    "Quelle est ta taille en cm ?", value=None, placeholder="Type a number..."
)

poids = st.number_input(
    "Quel est ton poids en kg ?", value=None, placeholder="Type a number..."
)

X_input['bmi'] = poids / ((taille / 100) ** 2)

X_input['physical_activity'] = st.number_input(
    "Combien d'heures par semaine fais-tu d'activité physique ?", value=None, placeholder="Type a number..."
)

X_input['diet_score'] = st.number_input(
    "Quel est ton score diététique (entre 0 et 20) ?", value=None, placeholder="Type a number..."
)

X_input['glucose'] = st.number_input(
    "Quel est ton taux de glucose sanguin (mg/dL) ?", value=None, placeholder="Type a number..."
)

X_input['hba1c'] = st.number_input(
    "Quel est ton taux d'HbA1c (%) ?", value=None, placeholder="Type a number..."
)

X_input['medical_condition'] = st.selectbox(
    "As-tu des antécédents médicaux ?", options=['diabetes', 'healthy', 'asthma', 'obesity', 'hypertension',
       'cancer', np.nan, 'arthritis']
)

# 1. load the trained model
with open(os.path.join(wd, 'models/xgb_best_model.pkl'), 'rb') as f:
    model = joblib.load(f)

#2. Load the preprocessor
with open(os.path.join(wd, 'models/preprocessor.pkl'), 'rb') as f:
    preprocessor = joblib.load(f)

if st.button('Prédire la durée d\'hospitalisation'):
    # Prétraitement (identique à l’entraînement)
    X_processed = preprocessor.transform(X_input)

    # Prédiction
    prediction = model.predict(X_processed)[0]

    st.success(f"La durée d'hospitalisation prédite est de {round(float(prediction), 2)} jours.")   