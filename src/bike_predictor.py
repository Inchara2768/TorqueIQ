import joblib
import pandas as pd
import streamlit as st
import os

@st.cache_resource(show_spinner=False)
def load_bike_model():
    model_path = os.path.join(os.path.dirname(__file__), '../models/bike_price_model.joblib')
    try:
        return joblib.load(model_path)
    except Exception as e:
        return None

def predict_bike_price(model, bike_name, city, owner, brand, kms, age, power):
    try:
        df = pd.DataFrame([[bike_name, city, owner, brand, kms, age, power]], 
                          columns=['bike_name', 'city', 'owner', 'brand', 'kms_driven', 'age', 'power'])
        
        pred = model.predict(df)[0]
        return pred
    except Exception as e:
        return None
