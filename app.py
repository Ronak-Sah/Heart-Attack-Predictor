import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("XGB_Model.pkl")

st.title("💓 Heart Attack Prediction App")
st.write("Enter patient details to predict the risk of heart attack.")

# Input fields
age = st.number_input("Age", min_value=20, max_value=100, value=40)
resting_bp = st.number_input("Resting BP", min_value=80, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-2.0, max_value=6.0, value=1.0)

chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
sex = st.selectbox("Sex", ["M", "F"])
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "LVH", "ST"])

# Convert categorical inputs to one-hot (must match training encoding)
input_dict = {
    "Age": age,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": fasting_bs,
    "Oldpeak": oldpeak,
    "ChestPainType_ASY": 1 if chest_pain == "ASY" else 0,
    "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
    "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
    "ChestPainType_TA": 1 if chest_pain == "TA" else 0,
    "Sex_M": 1 if sex == "M" else 0,
    "Sex_F": 1 if sex == "F" else 0,
    "ExerciseAngina_Y": 1 if exercise_angina == "Y" else 0,
    "ExerciseAngina_N": 1 if exercise_angina == "N" else 0,
    "ST_Slope_Up": 1 if st_slope == "Up" else 0,
    "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
    "ST_Slope_Down": 1 if st_slope == "Down" else 0,
    "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
    "RestingECG_LVH": 1 if resting_ecg == "LVH" else 0,
    "RestingECG_ST": 1 if resting_ecg == "ST" else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Predict button
if st.button("Predict"):
    
    feature_names = model.get_booster().feature_names  # For XGBoost model
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    pred = model.predict(input_df)
    prediction=pred[0]
    if prediction == 1:
        st.error("⚠️ High risk of Heart Attack!")
    else:
        st.success("✅ No Heart Attack Risk Detected")
