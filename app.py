import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from car_predictor import load_car_model, predict_car_price
from bike_predictor import load_bike_model, predict_bike_price

st.set_page_config(
    page_title="TorqueIQ - Smart Vehicle Value Predictor",
    layout="wide"
)

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&display=swap');
        .hero-container {
            padding: 40px;
            background: #121212;
            border-radius: 10px;
            margin-bottom: 30px;
            border-left: 4px solid #E10600;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        }
        .hero-kicker {
            font-size: 0.9rem;
            font-weight: 700;
            color: #2F80ED;
            letter-spacing: 2px;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .hero-wordmark {
            font-family: 'Arial Black', 'Inter', sans-serif;
            font-size: 3rem;
            letter-spacing: 8px;
            margin-bottom: 5px;
            background: linear-gradient(to right, #ffffff, #888888);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-wordmark .iq {
            color: #E10600;
            -webkit-text-fill-color: #E10600;
        }
        .hero-gradient-line {
            height: 3px;
            width: 100px;
            background: linear-gradient(to right, #E10600, #2F80ED);
            margin-top: 10px;
            margin-bottom: 20px;
        }
        .hero-headline {
            font-size: 2.5rem;
            font-weight: 800;
            color: #FFFFFF;
            margin-bottom: 15px;
            line-height: 1.2;
        }
        .hero-supporting {
            font-size: 1.1rem;
            color: #CCCCCC;
            margin-bottom: 25px;
            max-width: 800px;
            line-height: 1.6;
        }
        .hero-brand {
            font-size: 1.2rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 30px;
        }
        .hero-tags {
            font-size: 0.9rem;
            font-weight: 700;
            color: #AAAAAA;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        .hero-tags .dot {
            color: #E10600;
            font-size: 1.2rem;
            line-height: 0;
            padding: 0 5px;
        }
        .hero-footer {
            font-size: 0.8rem;
            font-weight: 800;
            color: #555555;
            letter-spacing: 3px;
            text-transform: uppercase;
        }
        .section-heading {
            margin-top: 10px;
            margin-bottom: 25px;
        }
        .section-title {
            font-family: 'Rajdhani', 'Arial Black', 'Inter', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: #FFFFFF;
            text-transform: uppercase;
            letter-spacing: 4px;
            margin-bottom: 5px;
        }
        .section-subtitle {
            font-size: 0.9rem;
            font-weight: 700;
            color: #888888;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 15px;
        }
        .section-line {
            height: 3px;
            width: 70px;
            background: #E10600;
            margin-bottom: 20px;
        }
        
        .premium-card {
            background: linear-gradient(135deg, #18181b 0%, #101012 100%);
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 25px 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
            position: relative;
            overflow: hidden;
        }
        .premium-card-label {
            font-size: 0.75rem;
            font-weight: 800;
            color: #E10600;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .premium-card.blue-accent .premium-card-label {
            color: #2F80ED;
        }
        .premium-card.white-accent .premium-card-label {
            color: #FFFFFF;
        }
        .premium-card-title {
            font-family: 'Rajdhani', 'Arial Black', 'Inter', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            color: #FFFFFF;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 15px;
        }
        .premium-card-desc {
            font-size: 1.05rem;
            color: #B0B0B0;
            line-height: 1.6;
            margin: 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #18181b 0%, #101012 100%);
            padding: 25px 20px;
            border-radius: 8px;
            border: 1px solid #333333;
            text-align: center;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
            height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .metric-value {
            font-family: 'Rajdhani', 'Inter', sans-serif;
            font-size: 2.8rem;
            font-weight: 700;
            color: #FFFFFF;
            margin: 5px 0 0 0;
            line-height: 1;
        }
        .metric-label {
            color: #888888;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        div[data-testid="stVerticalBlock"]:has(.veh-details-anchor) h5,
        div[data-testid="stVerticalBlock"]:has(.dep-analysis-anchor) h5,
        div[data-testid="stVerticalBlock"]:has(.car-valuation-anchor) h5,
        div[data-testid="stVerticalBlock"]:has(.bike-valuation-anchor) h5 {
            font-family: 'Rajdhani', 'Inter', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: #FFFFFF;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 25px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333333;
        }
        /* Cars Dropdowns */
        div[data-testid="stVerticalBlock"]:has(.car-valuation-anchor) div[data-testid="stSelectbox"] div[role="group"] {
            background: linear-gradient(135deg, #18181b 0%, rgba(225, 6, 0, 0.1) 100%) !important;
            border-color: #333333 !important;
            border-radius: 6px;
        }
        
        /* Bikes Dropdowns */
        div[data-testid="stVerticalBlock"]:has(.bike-valuation-anchor) div[data-testid="stSelectbox"] div[role="group"] {
            background: linear-gradient(135deg, #18181b 0%, rgba(47, 128, 237, 0.1) 100%) !important;
            border-color: #333333 !important;
            border-radius: 6px;
        }

        /* Depreciation Dropdowns */
        div[data-testid="stVerticalBlock"]:has(.veh-details-anchor) div[data-testid="stSelectbox"] div[role="group"],
        div[data-testid="stVerticalBlock"]:has(.dep-analysis-anchor) div[data-testid="stSelectbox"] div[role="group"] {
            background: linear-gradient(135deg, #18181b 0%, #111111 100%) !important;
            border-color: #333333 !important;
            border-radius: 6px;
        }

        /* Ensure dropdown text and icons are white and visible */
        div[data-testid="stSelectbox"] div[role="group"] input {
            color: #FFFFFF !important;
            background-color: transparent !important;
        }
        div[data-testid="stSelectbox"] div[role="group"] svg {
            fill: #FFFFFF !important;
        }

        div[data-testid="stVerticalBlock"]:has(.veh-details-anchor) div[data-testid="stNumberInput"] input,
        div[data-testid="stVerticalBlock"]:has(.dep-analysis-anchor) div[data-testid="stNumberInput"] input,
        div[data-testid="stVerticalBlock"]:has(.car-valuation-anchor) div[data-testid="stNumberInput"] input,
        div[data-testid="stVerticalBlock"]:has(.bike-valuation-anchor) div[data-testid="stNumberInput"] input {
            background-color: #121212 !important;
            border-color: #333333 !important;
            color: #FFFFFF !important;
            padding: 10px 15px !important;
            border-radius: 6px;
        }
        
        /* Input Focus States */
        div[data-testid="stVerticalBlock"]:has(.car-valuation-anchor) div[data-testid="stSelectbox"] div[role="group"]:focus-within,
        div[data-testid="stVerticalBlock"]:has(.car-valuation-anchor) div[data-testid="stNumberInput"] input:focus {
            border-color: #E10600 !important;
            box-shadow: 0 0 0 1px #E10600 !important;
        }
        div[data-testid="stVerticalBlock"]:has(.bike-valuation-anchor) div[data-testid="stSelectbox"] div[role="group"]:focus-within,
        div[data-testid="stVerticalBlock"]:has(.bike-valuation-anchor) div[data-testid="stNumberInput"] input:focus {
            border-color: #2F80ED !important;
            box-shadow: 0 0 0 1px #2F80ED !important;
        }

        /* Increase vertical spacing between input fields */
        div[data-testid="stVerticalBlock"]:has(.veh-details-anchor) div[data-testid="stSelectbox"],
        div[data-testid="stVerticalBlock"]:has(.veh-details-anchor) div[data-testid="stNumberInput"],
        div[data-testid="stVerticalBlock"]:has(.veh-details-anchor) div[data-testid="stSlider"],
        div[data-testid="stVerticalBlock"]:has(.dep-analysis-anchor) div[data-testid="stSelectbox"],
        div[data-testid="stVerticalBlock"]:has(.dep-analysis-anchor) div[data-testid="stNumberInput"],
        div[data-testid="stVerticalBlock"]:has(.dep-analysis-anchor) div[data-testid="stSlider"],
        div[data-testid="stVerticalBlock"]:has(.car-valuation-anchor) div[data-testid="stSelectbox"],
        div[data-testid="stVerticalBlock"]:has(.car-valuation-anchor) div[data-testid="stNumberInput"],
        div[data-testid="stVerticalBlock"]:has(.bike-valuation-anchor) div[data-testid="stSelectbox"],
        div[data-testid="stVerticalBlock"]:has(.bike-valuation-anchor) div[data-testid="stNumberInput"] {
            margin-bottom: 12px;
        }
        
        /* Premium Text Styling for Forms */
        div[data-testid="stVerticalBlock"]:has(.car-valuation-anchor) label p,
        div[data-testid="stVerticalBlock"]:has(.bike-valuation-anchor) label p {
            color: #B0B0B0 !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }
        
        div[data-testid="stVerticalBlock"]:has(.car-valuation-anchor) div[data-testid="stNumberInput"] input,
        div[data-testid="stVerticalBlock"]:has(.bike-valuation-anchor) div[data-testid="stNumberInput"] input,
        div[data-testid="stVerticalBlock"]:has(.car-valuation-anchor) div[data-testid="stSelectbox"] div[role="group"] input,
        div[data-testid="stVerticalBlock"]:has(.bike-valuation-anchor) div[data-testid="stSelectbox"] div[role="group"] input {
            font-size: 1.05rem !important;
            font-weight: 500 !important;
            color: #FFFFFF !important;
        }
        
        /* Shared Button Style */
        div.stButton > button[kind="primary"] {
            color: #FFFFFF;
            border-radius: 6px;
            font-family: 'Rajdhani', 'Inter', sans-serif;
            font-weight: 700;
            font-size: 1.2rem;
            letter-spacing: 2px;
            text-transform: uppercase;
            padding: 1rem;
            margin-top: 15px;
            transition: all 0.3s ease;
            box-shadow: none;
        }
        div.stButton > button[kind="primary"] p {
            font-weight: 700;
            color: #FFFFFF;
        }
        /* Cars Button (Red) */
        div[data-testid="stVerticalBlock"]:has(.car-valuation-anchor) div.stButton > button[kind="primary"] {
            background-color: rgba(225, 6, 0, 0.08);
            border: 1px solid rgba(225, 6, 0, 0.5);
        }
        div[data-testid="stVerticalBlock"]:has(.car-valuation-anchor) div.stButton > button[kind="primary"]:hover {
            background-color: rgba(225, 6, 0, 0.15);
            border: 1px solid rgba(225, 6, 0, 0.8);
        }
        /* Bikes Button (Blue) */
        div[data-testid="stVerticalBlock"]:has(.bike-valuation-anchor) div.stButton > button[kind="primary"] {
            background-color: rgba(47, 128, 237, 0.08);
            border: 1px solid rgba(47, 128, 237, 0.5);
        }
        div[data-testid="stVerticalBlock"]:has(.bike-valuation-anchor) div.stButton > button[kind="primary"]:hover {
            background-color: rgba(47, 128, 237, 0.15);
            border: 1px solid rgba(47, 128, 237, 0.8);
        }
        
        .disclaimer {
            color: #888888;
            font-size: 0.8rem;
            margin-top: 30px;
            text-align: center;
        }
        /* Tabs CSS replaced by button navigation */
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

st.markdown("""
<div class="hero-container">
<div class="hero-kicker">PRECISION. PERFORMANCE. PRICE.</div>
<div class="hero-wordmark">T O R Q U E <span class="iq">I Q</span></div>
<div class="hero-gradient-line"></div>
<div class="hero-headline">KNOW YOUR RIDE'S WORTH. OWN THE DEAL.</div>
<div class="hero-supporting">No guesswork. No second-guessing. Just a sharp, data-driven estimate built around your car.</div>
<div class="hero-brand">YOUR CAR. YOUR NUMBER.</div>

<div class="hero-tags">
<span>CAR VALUATION</span> <span class="dot">&bull;</span> <span>BIKE VALUATION</span> <span class="dot">&bull;</span> <span>MARKET INSIGHTS</span>
</div>

<div class="hero-footer">BUILT TO LEAD. ENGINEERED TO PREDICT.</div>
</div>
""", unsafe_allow_html=True)

if 'active_section' not in st.session_state:
    st.session_state.active_section = 'CARS'

def set_section(section):
    st.session_state.active_section = section

st.markdown("""
<style>
/* Custom Segmented Navigation */
div[data-testid="stVerticalBlock"]:has(.custom-nav-container) {
    background: transparent;
    border: none;
    padding: 0;
    margin-bottom: 30px;
}
div[data-testid="stVerticalBlock"]:has(.custom-nav-container) div[data-testid="stHorizontalBlock"] {
    gap: 20px;
}
div[data-testid="stVerticalBlock"]:has(.custom-nav-container) button {
    height: 45px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    transition: all 0.3s ease;
    box-shadow: none;
    width: 100%;
}
div[data-testid="stVerticalBlock"]:has(.custom-nav-container) button p {
    font-family: 'Rajdhani', 'Inter', sans-serif;
    font-size: 1.15rem;
    font-weight: 500;
    color: #FFFFFF;
    opacity: 0.6;
    letter-spacing: 2px;
    margin: 0;
    text-transform: uppercase;
    transition: all 0.3s ease;
}
div[data-testid="stVerticalBlock"]:has(.custom-nav-container) button:hover {
    background: transparent;
}
div[data-testid="stVerticalBlock"]:has(.custom-nav-container) button:hover p {
    color: #FFFFFF;
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)

active_colors = {
    "CARS": "#E10600",
    "BIKES": "#2F80ED",
    "DEPRECIATION": "rgba(255, 255, 255, 0.4)",
    "MARKET INSIGHTS": "rgba(255, 255, 255, 0.4)"
}

active_idx = {"CARS": 1, "BIKES": 2, "DEPRECIATION": 3, "MARKET INSIGHTS": 4}[st.session_state.active_section]
cur_color = active_colors[st.session_state.active_section]

st.markdown(f"""
<style>
div[data-testid="stVerticalBlock"]:has(.custom-nav-container) div[data-testid="column"]:nth-child({active_idx}) button {{
    background: transparent !important;
    border-bottom: 2px solid {cur_color} !important;
}}
div[data-testid="stVerticalBlock"]:has(.custom-nav-container) div[data-testid="column"]:nth-child({active_idx}) button p {{
    color: #FFFFFF !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}}
</style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='custom-nav-container'></div>", unsafe_allow_html=True)
    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)
    with col_nav1:
        st.button("CARS", on_click=set_section, args=("CARS",), use_container_width=True)
    with col_nav2:
        st.button("BIKES", on_click=set_section, args=("BIKES",), use_container_width=True)
    with col_nav3:
        st.button("DEPRECIATION", on_click=set_section, args=("DEPRECIATION",), use_container_width=True)
    with col_nav4:
        st.button("MARKET INSIGHTS", on_click=set_section, args=("MARKET INSIGHTS",), use_container_width=True)

@st.cache_data(show_spinner=False)
def load_data():
    df_cars = pd.read_csv('data/cleaned_cars_dataset.csv')
    df_bikes = pd.read_csv('data/cleaned_bikes_dataset.csv')
    return df_cars, df_bikes

def format_inr(number):
    s = str(int(number))
    if len(s) <= 3:
        return s
    res = s[-3:]
    s = s[:-3]
    while len(s) > 2:
        res = s[-2:] + "," + res
        s = s[:-2]
    if s:
        res = s + "," + res
    return res

df_cars, df_bikes = load_data()
car_model = load_car_model()
bike_model = load_bike_model()

st.markdown('<div style="height: 1px; width: 100%; background: linear-gradient(90deg, #E10600 0%, #2F80ED 50%, #111111 100%); margin: 20px 0 40px 0;"></div>', unsafe_allow_html=True)

if st.session_state.active_section == "CARS":
    st.markdown("""
<div class="section-heading">
<div class="section-title">CARS</div>
<div class="section-subtitle">CAR PRICE PREDICTION & MARKET INSIGHTS</div>
<div class="section-line"></div>
</div>

<div class="premium-card">
<div class="premium-card-label">CAR VALUATION SYSTEM</div>
<div class="premium-card-title">PRECISION VALUATION ENGINE</div>
<div class="premium-card-desc">Determine the accurate resale value of used cars based on market dynamics.</div>
</div>
""", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='car-valuation-anchor'></div>", unsafe_allow_html=True)
        st.markdown("<h5>CAR VALUATION INPUTS</h5>", unsafe_allow_html=True)
        
        c1, spacer1, c2 = st.columns([1, 0.04, 1])
        with c1:
            car_brand = st.selectbox("Brand", sorted(df_cars['Brand'].unique().tolist()))
        
        brand_models = sorted(df_cars[df_cars['Brand'] == car_brand]['Car_Model'].unique().tolist())
        with spacer1:
            st.empty()
        with c2:
            car_model_in = st.selectbox("Model", brand_models)
            
        c3, spacer2, c4 = st.columns([1, 0.04, 1])
        with c3:
            car_kms = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=40000, step=1000)
        with spacer2:
            st.empty()
        with c4:
            car_year = st.number_input("Manufacturing Year", min_value=1950, max_value=2026, value=2018, step=1)
            
        c5, spacer3, c6 = st.columns([1, 0.04, 1])
        with c5:
            car_fuel = st.selectbox("Fuel Type", sorted(df_cars['Car_Fuel'].unique().tolist()))
        with spacer3:
            st.empty()
        with c6:
            car_loc = st.selectbox("Location", sorted(df_cars['Car_Location'].unique().tolist()))
            
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        if st.button("GET MY VALUATION", key="car_calc", use_container_width=True, type="primary"):
            if car_model:
                pred = predict_car_price(car_model, car_model_in, car_fuel, car_loc, car_brand, car_year, car_kms)
                if pred is not None:
                    st.markdown(f"""
                    <div style="margin-top: 30px; background: linear-gradient(135deg, #18181b 0%, #101012 100%); padding: 30px; border-radius: 8px; border: 1px solid #333333; border-top: 4px solid #E10600; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);">
                        <div style="text-align: center; margin-bottom: 25px;">
                            <div style="color: #888888; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px;">Estimated Resale Value</div>
                            <div style="font-family: 'Rajdhani', sans-serif; font-size: 3.5rem; font-weight: 700; color: #FFFFFF; margin-top: 5px; line-height: 1;">₹{pred:,.0f}</div>
                        </div>
                        <div style="border-top: 1px solid #333333; padding-top: 25px;">
                            <div style="color: #E10600; font-size: 0.8rem; font-weight: 800; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 15px;">Vehicle Summary</div>
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                                <div><span style="color: #888888; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 3px;">Brand</span><span style="color: #FFFFFF; font-weight: 600; font-size: 1.05rem;">{car_brand}</span></div>
                                <div><span style="color: #888888; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 3px;">Model</span><span style="color: #FFFFFF; font-weight: 600; font-size: 1.05rem;">{car_model_in}</span></div>
                                <div><span style="color: #888888; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 3px;">Year</span><span style="color: #FFFFFF; font-weight: 600; font-size: 1.05rem;">{car_year}</span></div>
                                <div><span style="color: #888888; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 3px;">Fuel Type</span><span style="color: #FFFFFF; font-weight: 600; font-size: 1.05rem;">{car_fuel}</span></div>
                                <div><span style="color: #888888; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 3px;">Kilometers</span><span style="color: #FFFFFF; font-weight: 600; font-size: 1.05rem;">{car_kms:,} km</span></div>
                                <div><span style="color: #888888; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 3px;">Location</span><span style="color: #FFFFFF; font-weight: 600; font-size: 1.05rem;">{car_loc}</span></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Prediction failed. Please check inputs and try again.")
            else:
                st.error("Model failed to load.")

elif st.session_state.active_section == "BIKES":
    st.markdown("""
<div class="section-heading">
<div class="section-title">BIKES</div>
<div class="section-subtitle">BIKE PRICE PREDICTION & MARKET INSIGHTS</div>
<div class="section-line" style="background: #2F80ED;"></div>
</div>

<div class="premium-card blue-accent">
<div class="premium-card-label">BIKE VALUATION SYSTEM</div>
<div class="premium-card-title">TWO-WHEELER VALUATION ENGINE</div>
<div class="premium-card-desc">Accurately estimate the value of used motorcycles in the current market.</div>
</div>
""", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='bike-valuation-anchor'></div>", unsafe_allow_html=True)
        st.markdown("<h5>BIKE VALUATION INPUTS</h5>", unsafe_allow_html=True)
        
        b1, spacer_b1, b2 = st.columns([1, 0.04, 1])
        with b1:
            bike_brand = st.selectbox("Brand", sorted(df_bikes['brand'].unique().tolist()))
        
        brand_bikes = sorted(df_bikes[df_bikes['brand'] == bike_brand]['bike_name'].unique().tolist())
        with spacer_b1:
            st.empty()
        with b2:
            bike_name = st.selectbox("Bike Name", brand_bikes)
            
        b3, spacer_b2, b4 = st.columns([1, 0.04, 1])
        with b3:
            bike_kms = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=20000, step=1000)
        with spacer_b2:
            st.empty()
        with b4:
            bike_age = st.number_input("Age (Years)", min_value=0, max_value=50, value=3, step=1)
            
        b5, spacer_b3, b6, spacer_b4, b7 = st.columns([1, 0.04, 1, 0.04, 1])
        with b5:
            bike_power = st.number_input("Engine Power (cc)", min_value=50, max_value=2500, value=150, step=10)
        with spacer_b3:
            st.empty()
        with b6:
            bike_owner = st.selectbox("Owner", sorted(df_bikes['owner'].unique().tolist()))
        with spacer_b4:
            st.empty()
        with b7:
            bike_city = st.selectbox("City", sorted(df_bikes['city'].unique().tolist()))
            
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        if st.button("GET MY VALUATION", key="bike_calc", use_container_width=True, type="primary"):
            if bike_model:
                pred = predict_bike_price(bike_model, bike_name, bike_city, bike_owner, bike_brand, bike_kms, bike_age, bike_power)
                if pred is not None:
                    st.markdown(f"""
                    <div style="margin-top: 30px; background: linear-gradient(135deg, #18181b 0%, #101012 100%); padding: 30px; border-radius: 8px; border: 1px solid #333333; border-top: 4px solid #2F80ED; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);">
                        <div style="text-align: center; margin-bottom: 25px;">
                            <div style="color: #888888; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px;">Estimated Resale Value</div>
                            <div style="font-family: 'Rajdhani', sans-serif; font-size: 3.5rem; font-weight: 700; color: #FFFFFF; margin-top: 5px; line-height: 1;">₹{pred:,.0f}</div>
                        </div>
                        <div style="border-top: 1px solid #333333; padding-top: 25px;">
                            <div style="color: #2F80ED; font-size: 0.8rem; font-weight: 800; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 15px;">Vehicle Summary</div>
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                                <div><span style="color: #888888; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 3px;">Brand</span><span style="color: #FFFFFF; font-weight: 600; font-size: 1.05rem;">{bike_brand}</span></div>
                                <div><span style="color: #888888; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 3px;">Model</span><span style="color: #FFFFFF; font-weight: 600; font-size: 1.05rem;">{bike_name}</span></div>
                                <div><span style="color: #888888; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 3px;">Age</span><span style="color: #FFFFFF; font-weight: 600; font-size: 1.05rem;">{bike_age} Years</span></div>
                                <div><span style="color: #888888; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 3px;">Engine Power</span><span style="color: #FFFFFF; font-weight: 600; font-size: 1.05rem;">{bike_power} cc</span></div>
                                <div><span style="color: #888888; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 3px;">Kilometers</span><span style="color: #FFFFFF; font-weight: 600; font-size: 1.05rem;">{bike_kms:,} km</span></div>
                                <div><span style="color: #888888; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 3px;">Ownership</span><span style="color: #FFFFFF; font-weight: 600; font-size: 1.05rem;">{bike_owner}</span></div>
                                <div><span style="color: #888888; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 3px;">City</span><span style="color: #FFFFFF; font-weight: 600; font-size: 1.05rem;">{bike_city}</span></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Prediction failed. Please check inputs and try again.")
            else:
                st.error("Model failed to load.")

elif st.session_state.active_section == "DEPRECIATION":
    st.markdown("""
<div class="section-heading">
<div class="section-title">DEPRECIATION</div>
<div class="section-subtitle">VEHICLE DEPRECIATION CALCULATOR</div>
<div class="section-line"></div>
</div>

<div class="premium-card">
<div class="premium-card-label">DEPRECIATION ENGINE</div>
<div class="premium-card-title">VALUE RETENTION CALCULATOR</div>
<div class="premium-card-desc">Calculate the estimated current value of a vehicle based on standard depreciation models.</div>
</div>
""", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='veh-details-anchor'></div>", unsafe_allow_html=True)
        st.markdown("<h5>VEHICLE DETAILS</h5>", unsafe_allow_html=True)
        
        r1_col1, r1_col2, r1_col3 = st.columns(3)
        with r1_col1: vehicle_type = st.selectbox("Vehicle Type", ["Cars", "Bikes"])
        with r1_col2: purchase_price = st.number_input("Purchase Price (₹)", min_value=100, max_value=10000000, value=30000, step=1000)
        with r1_col3: depreciation_rate = st.slider("Annual Depreciation Rate (%)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
        
        r2_col1, r2_col2, r2_col3 = st.columns(3)
        current_year_default = 2026
        with r2_col1: purchase_year = st.number_input("Purchase Year", min_value=1950, max_value=current_year_default, value=2021, step=1)
        with r2_col2: current_year = st.number_input("Current Year", min_value=purchase_year, max_value=2100, value=current_year_default, step=1)
        
    years_owned = current_year - purchase_year
    if years_owned == 0:
        current_value = purchase_price
    else:
        current_value = purchase_price * (1 - depreciation_rate / 100) ** years_owned
        
    value_lost = purchase_price - current_value
    percent_depreciated = (value_lost / purchase_price) * 100 if purchase_price > 0 else 0
    
    with st.container():
        st.markdown("<div class='dep-analysis-anchor'></div>", unsafe_allow_html=True)
        st.markdown("<h5>DEPRECIATION ANALYSIS</h5>", unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid #E10600;">
                <div class="metric-label">Estimated Value</div>
                <div class="metric-value">₹{format_inr(current_value)}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid #2F80ED;">
                <div class="metric-label">Value Lost</div>
                <div class="metric-value">₹{format_inr(value_lost)}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid #FFFFFF;">
                <div class="metric-label">Depreciation</div>
                <div class="metric-value">{percent_depreciated:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
        # Chart placed below metric cards in a natural flow
        
        years_list = list(range(purchase_year, current_year + 1))
        values_list = [purchase_price * (1 - depreciation_rate / 100) ** (y - purchase_year) for y in years_list]
        
        df_chart = pd.DataFrame({
            "Year": years_list,
            "Estimated Value (₹)": values_list
        })
        
        fig = px.line(df_chart, x="Year", y="Estimated Value (₹)", markers=True,
                      title=f"{vehicle_type.upper()} VALUE OVER TIME")
        
        line_color = '#E10600' if vehicle_type == 'Cars' else '#2F80ED'
        
        fig.update_traces(line_color=line_color, marker=dict(size=8, color=line_color))
        fig.update_layout(
            height=370,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#FFFFFF',
            title_font_family='Rajdhani',
            title_font_size=24,
            xaxis=dict(showgrid=True, gridcolor='#333333', dtick=1),
            yaxis=dict(showgrid=True, gridcolor='#333333', tickprefix='₹'),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="disclaimer">
        <b>Disclaimer:</b> This calculator provides rough estimates based on a simplified fixed-rate depreciation model. 
        Actual vehicle resale values depend on mileage, condition, local market demand, modifications, and macroeconomic factors. 
        Do not use this as a guaranteed resale price.
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.active_section == "MARKET INSIGHTS":
    st.markdown("""
<div class="section-heading">
<div class="section-title">MARKET INSIGHTS</div>
<div class="section-subtitle">DATA-DRIVEN TRENDS & ANALYSIS</div>
<div class="section-line" style="background: #FFFFFF;"></div>
</div>
""", unsafe_allow_html=True)

    with st.container():
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            insight_cat = st.selectbox("Vehicle Category", ["Cars", "Bikes"])
            
        if insight_cat == "Cars":
            df_plot = df_cars.copy()
            brand_col = "Brand"
            city_col = "Car_Location"
            year_col = "Car_Year"
            kms_col = "Car_Kms"
            price_col = "Car_Price"
            theme_color = "#E10600"
        else:
            df_plot = df_bikes.copy()
            brand_col = "brand"
            city_col = "city"
            year_col = "age" 
            kms_col = "kms_driven"
            price_col = "price"
            theme_color = "#2F80ED"
            
        with m2:
            f_brand = st.selectbox("Filter by Brand", ["All"] + sorted(df_plot[brand_col].unique().tolist()))
        with m3:
            f_city = st.selectbox("Filter by City", ["All"] + sorted(df_plot[city_col].unique().tolist()))
            
        # Age/Year filter
        if insight_cat == "Cars":
            min_y, max_y = int(df_plot[year_col].min()), int(df_plot[year_col].max())
            with m4:
                f_year = st.slider("Filter by Year", min_y, max_y, (min_y, max_y))
        else:
            min_a, max_a = int(df_plot[year_col].min()), int(df_plot[year_col].max())
            with m4:
                f_age = st.slider("Filter by Age", min_a, max_a, (min_a, max_a))
                
        # Apply filters
        if f_brand != "All":
            df_plot = df_plot[df_plot[brand_col] == f_brand]
        if f_city != "All":
            df_plot = df_plot[df_plot[city_col] == f_city]
            
        if insight_cat == "Cars":
            df_plot = df_plot[(df_plot[year_col] >= f_year[0]) & (df_plot[year_col] <= f_year[1])]
        else:
            df_plot = df_plot[(df_plot[year_col] >= f_age[0]) & (df_plot[year_col] <= f_age[1])]
            
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown(f"<h5>SUMMARY METRICS</h5>", unsafe_allow_html=True)
        sm1, sm2, sm3, sm4 = st.columns(4)
        
        num_listings = len(df_plot)
        with sm1:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {theme_color};">
                <div class="metric-label">Total Listings</div>
                <div class="metric-value">{num_listings:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        med_price = df_plot[price_col].median() if num_listings > 0 else 0
        with sm2:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {theme_color};">
                <div class="metric-label">Median Price</div>
                <div class="metric-value">₹{med_price:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        avg_price = df_plot[price_col].mean() if num_listings > 0 else 0
        with sm3:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {theme_color};">
                <div class="metric-label">Average Price</div>
                <div class="metric-value">₹{avg_price:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        med_age = df_plot[year_col].median() if num_listings > 0 else 0
        age_label = "Median Year" if insight_cat == "Cars" else "Median Age"
        age_val = f"{int(med_age)}" if insight_cat == "Cars" else f"{med_age:,.1f} Yrs"
        with sm4:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {theme_color};">
                <div class="metric-label">{age_label}</div>
                <div class="metric-value">{age_val if num_listings > 0 else '-'}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        
        if num_listings > 0:
            ch1, ch2 = st.columns(2)
            
            with ch1:
                fig_dist = px.histogram(df_plot, x=price_col, title="Price Distribution", color_discrete_sequence=[theme_color], nbins=30)
                fig_dist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#FFFFFF', title_font_family='Rajdhani')
                fig_dist.update_xaxes(title="Listed Price (INR)", showgrid=True, gridcolor='#333333')
                fig_dist.update_yaxes(title="Count", showgrid=True, gridcolor='#333333')
                st.plotly_chart(fig_dist, use_container_width=True)
                
            with ch2:
                if f_brand == "All":
                    brand_med = df_plot.groupby(brand_col)[price_col].median().reset_index().sort_values(by=price_col, ascending=False).head(15)
                    fig_brand = px.bar(brand_med, x=brand_col, y=price_col, title="Top 15 Brands by Median Price", color_discrete_sequence=["#FFFFFF"])
                    fig_brand.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#FFFFFF', title_font_family='Rajdhani')
                    fig_brand.update_xaxes(title="Brand", showgrid=False)
                    fig_brand.update_yaxes(title="Median Price (INR)", showgrid=True, gridcolor='#333333')
                    st.plotly_chart(fig_brand, use_container_width=True)
                else:
                    model_col = "Car_Model" if insight_cat == "Cars" else "bike_name"
                    model_med = df_plot.groupby(model_col)[price_col].median().reset_index().sort_values(by=price_col, ascending=False).head(15)
                    fig_brand = px.bar(model_med, x=model_col, y=price_col, title=f"Top 15 Models by Median Price ({f_brand})", color_discrete_sequence=["#FFFFFF"])
                    fig_brand.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#FFFFFF', title_font_family='Rajdhani')
                    fig_brand.update_xaxes(title="Model", showgrid=False)
                    fig_brand.update_yaxes(title="Median Price (INR)", showgrid=True, gridcolor='#333333')
                    st.plotly_chart(fig_brand, use_container_width=True)
            
            ch3, ch4 = st.columns(2)
            
            with ch3:
                fig_age = px.scatter(df_plot, x=year_col, y=price_col, title=f"{age_label.replace('Median ', '')} vs Price", color_discrete_sequence=["#FFFFFF"], opacity=0.5)
                fig_age.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#FFFFFF', title_font_family='Rajdhani')
                fig_age.update_xaxes(title=age_label.replace('Median ', ''), showgrid=True, gridcolor='#333333')
                fig_age.update_yaxes(title="Listed Price (INR)", showgrid=True, gridcolor='#333333')
                st.plotly_chart(fig_age, use_container_width=True)
                
            with ch4:
                fig_kms = px.scatter(df_plot, x=kms_col, y=price_col, title="Kilometers Driven vs Price", color_discrete_sequence=[theme_color], opacity=0.5)
                fig_kms.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#FFFFFF', title_font_family='Rajdhani')
                fig_kms.update_xaxes(title="Kilometers Driven", showgrid=True, gridcolor='#333333')
                fig_kms.update_yaxes(title="Listed Price (INR)", showgrid=True, gridcolor='#333333')
                st.plotly_chart(fig_kms, use_container_width=True)
                
            ch5, ch6 = st.columns(2)
            
            with ch5:
                city_med = df_plot.groupby(city_col)[price_col].median().reset_index().sort_values(by=price_col, ascending=False).head(15)
                fig_city = px.bar(city_med, x=city_col, y=price_col, title="Top 15 Cities by Median Price", color_discrete_sequence=[theme_color])
                fig_city.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#FFFFFF', title_font_family='Rajdhani')
                fig_city.update_xaxes(title="City", showgrid=False)
                fig_city.update_yaxes(title="Median Price (INR)", showgrid=True, gridcolor='#333333')
                st.plotly_chart(fig_city, use_container_width=True)
                
            if insight_cat == "Cars":
                with ch6:
                    fuel_dist = df_plot['Car_Fuel'].value_counts().reset_index()
                    fuel_dist.columns = ['Fuel Type', 'Count']
                    fig_fuel = px.pie(fuel_dist, values='Count', names='Fuel Type', title="Fuel Type Distribution", color_discrete_sequence=["#E10600", "#FFFFFF", "#2F80ED"])
                    fig_fuel.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#FFFFFF', title_font_family='Rajdhani')
                    st.plotly_chart(fig_fuel, use_container_width=True)
        else:
            st.info("No listings found for the selected filters.")


st.markdown("---")
st.markdown("<p style='text-align: center; color: #888888; font-size: 0.8rem;'>© 2026 TorqueIQ Analytics. All rights reserved.</p>", unsafe_allow_html=True)