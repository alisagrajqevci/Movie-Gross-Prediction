import streamlit as st
import numpy as np
import joblib


model = joblib.load("gross_rf_model.pkl")
scaler = joblib.load("gross_scaler.pkl")

st.title("🎬 Movie Gross Revenue Predictor")
st.write("Estimate how much a movie will gross based on its budget, IMDB score, votes, and runtime.")

budget = st.number_input("🎥 Budget (USD)", value=10000000, step=1000000)
score = st.slider("⭐ IMDB Score", min_value=1.0, max_value=10.0, step=0.1, value=6.5)
votes = st.number_input("🗳️ Number of Votes", value=50000, step=1000)
runtime = st.number_input("⏱️ Runtime (minutes)", value=100, step=1)

# Predict
if st.button("Predict Gross Revenue"):
    input_data = np.array([[budget, score, votes, runtime]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    st.success(f"💰 Predicted Gross Revenue: **${prediction:,.2f}**")
