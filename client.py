# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and scaler from the pickled files
with open("logreg_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("Wine Quality Prediction")

st.header("Enter the wine features to predict quality:")
wine_type_selection = st.selectbox("Wine Type", ["Red", "White"])
wine_type_encoded = 0 if wine_type_selection == "Red" else 1

# Gather inputs
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=15.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, step=0.01)
citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=2.0, step=0.1)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=15.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, step=0.01)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, step=1.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=250.0, step=1.0)
density = st.number_input("Density", min_value=0.98, max_value=1.2, step=0.001)
pH = st.number_input("pH", min_value=2.5, max_value=4.5, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, step=0.1)
alcohol = st.number_input("Alcohol Content", min_value=0.0, max_value=20.0, step=0.1)

# Create a DataFrame for user input with matching column names
user_input = pd.DataFrame({
    'type': [wine_type_encoded],
    'fixed_acidity': [fixed_acidity],
    'volatile_acidity': [volatile_acidity],
    'citric_acid': [citric_acid],
    'residual_sugar': [residual_sugar],
    'chlorides': [chlorides],
    'free_sulfur_dioxide': [free_sulfur_dioxide],
    'total_sulfur_dioxide': [total_sulfur_dioxide],
    'density': [density],
    'pH': [pH],
    'sulphates': [sulphates],
    'alcohol': [alcohol]
})

# Normalize the user input
user_input_scaled = scaler.transform(user_input)

# Make a prediction
prediction = model.predict(user_input_scaled)
quality_msg = "Good Quality" if prediction[0] == 1 else "Bad Quality"
st.info(f"The wine is of '{quality_msg}'")
