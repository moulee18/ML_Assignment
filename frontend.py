import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained SVM model
model = joblib.load("svm_model.pkl")

# Streamlit UI
st.title("Kidney Disease Prediction using SVM")

# Define feature inputs (replace with actual feature names)
feature_names = [
    "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar", "Red Blood Cells",
    "Pus Cell", "Bacteria", "Blood Urea",
    "Serum Creatinine", "Sodium", "Potassium", "Hemoglobin", "Packed Cell Volume",
    "White Blood Cell Count", "Red Blood Cell Count"
]

# Create input fields
user_inputs = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}", min_value=0.0, format="%.2f")
    user_inputs.append(value)

# Convert to NumPy array and reshape for model
input_array = np.array(user_inputs).reshape(1, -1)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_array)
    result = "Positive for Kidney Disease" if prediction[0] == 1 else "Negative for Kidney Disease"
    st.success(f"Prediction: {result}")
