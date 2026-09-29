import joblib
import pandas as pd
import streamlit as st
import os

@st.cache_resource(show_spinner=False)
def load_car_model():
    model_path = os.path.join(os.path.dirname(__file__), '../models/car_price_model.joblib')
    try:
        return joblib.load(model_path)
    except Exception as e:
        return None

def predict_car_price(model, car_model_str, fuel, location, brand, year, kms):
    try:
        df = pd.DataFrame([[car_model_str, fuel, location, brand, year, kms]], 
                          columns=['Car_Model', 'Car_Fuel', 'Car_Location', 'Brand', 'Car_Year', 'Car_Kms'])
        
        pred = model.predict(df)[0]
        return pred
    except Exception as e:
        return None
