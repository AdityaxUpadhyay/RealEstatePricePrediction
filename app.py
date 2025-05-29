
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer

# Load the saved SVR model pipeline
model = joblib.load("svr_house_price_model.joblib")


st.title("Bengaluru House Price Prediction")
st.markdown("Predict the house price using area, number of bathrooms, and BHK (bedrooms).")

# User inputs
min_sqft = st.number_input("Minimum Area (in sqft)", min_value=100, max_value=10000, value=1000, step=50)
max_sqft = st.number_input("Maximum Area (in sqft)", min_value=min_sqft, max_value=15000, value=1200, step=50)
bhk = st.selectbox("BHK (Bedrooms)", list(range(1, 11)))
bath = st.selectbox("Bathrooms", list(range(1, 11)))

# Predict button
if st.button("Predict Price"):
    avg_sqft = (min_sqft + max_sqft) / 2
    input_data = pd.DataFrame([[avg_sqft, bath, bhk]], columns=['total_sqft', 'bath', 'bhk'])
    try:
        prediction = model.predict(input_data)
        st.success(f"Estimated Price: ₹{prediction[0]:,.2f} Lakhs")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
