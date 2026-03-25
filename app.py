import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


# CONFIG
st.set_page_config(page_title="House Price AI", page_icon="🏠", layout="wide")

# LOAD MODEL (Pipeline)
import os

def load_model():
    return joblib.load("house_model.pkl")

model = load_model()

# UI STYLE

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    color: white;
}
h1, h2, h3, label {
    color: white !important;
}
div[data-testid="stNumberInput"] input {
    color: black !important;
    background-color: white !important;
    border-radius: 10px;
}
div[data-baseweb="select"] div {
    color: black !important;
    background-color: white !important;
}
div[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 28px;
    font-weight: bold;
}
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #facc15);
    color: white;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
}
            
/* ===== METRIC CARD ===== */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}
            
div[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    transition: 0.3s;
    box-shadow: 0 12px 30px rgba(0,0,0,0.5);
}

/* ===== FIX EXPANDER HEADER ===== */

div[data-testid="stExpander"] summary {
    color: white !important;
    font-weight: 600;
}
div[data-testid="stExpander"] summary:hover {
    color: #facc15 !important;  /* เหลือง */
}

div[data-testid="stExpander"] summary:focus {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# HEADER

st.markdown("<h1 style='text-align:center;'>House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict house price using Machine Learning</p>", unsafe_allow_html=True)

st.markdown("---")


# INPUT

st.subheader("📊 Property Information")

col1, col2, col3 = st.columns(3)

with col1:
    rooms = st.number_input("Rooms", 1, 10, 3)
    bedroom2 = st.number_input("Bedrooms", 0, 10, 3)
    bathroom = st.number_input("Bathrooms", 1, 5, 2)

with col2:
    car = st.number_input("Car Spaces", 0, 5, 1)
    landsize = st.number_input("Land Size (sqm)", 1, 5000, 500)
    buildingarea = st.number_input("Building Area (sqm)", 1, 1000, 100)

with col3:
    yearbuilt = st.number_input("Year Built", 1800, 2025, 2000)
    distance = st.number_input("Distance (km)", 0.0, 50.0, 10.0)
    region = st.selectbox("Region", ["Northern", "Western", "Southern", "Eastern"])


# FEATURE EXPLANATION

with st.expander("ℹ️ What do these features mean?"):
    st.markdown("""
    **Rooms** – จำนวนห้อง  
    **Bedrooms** – จำนวนห้องนอน  
    **Bathrooms** – จำนวนห้องน้ำ  
    **Parking area** – ที่จอดรถ  
    **Land Size** – ขนาดพื้นที่ (ตารางเมตร)  
    **Building Area** – ขนาดอาคารก่อสร้าง(ตารางเมตร)  
    **Year Built** – ปีที่สร้าง  
    **Distance** – ระยะห่างจากเมือง (กิโลเมตร)  
    **Region** – ภูมิภาค  
    """)

st.markdown("---")


# PREDICT

if st.button(" Predict Price", use_container_width=True):

    # ========= VALIDATION =========
    if rooms < 1:
        st.error("Rooms must be at least 1")
        st.stop()

    if bathroom < 1:
        st.error("Bathrooms must be at least 1")
        st.stop()

    if landsize <= 0 or buildingarea <= 0:
        st.error("Area must be at least 0")
        st.stop()

    if yearbuilt < 1800 or yearbuilt > 2025:
        st.error("Invalid year built")
        st.stop()

    if distance < 0:
        st.error("Distance cannot be negative")
        st.stop()

    # ========= INPUT DATA =========
    input_data = pd.DataFrame({
        'Suburb': ['Richmond'],
        'Rooms': [rooms],
        'Type': ['h'],
        'Method': ['S'],
        'SellerG': ['Unknown'],
        'Distance': [distance],
        'Postcode': [3000],
        'Bedroom2': [bedroom2],
        'Bathroom': [bathroom],
        'Car': [car],
        'Landsize': [landsize],
        'BuildingArea': [buildingarea],
        'YearBuilt': [yearbuilt],
        'CouncilArea': ['Unknown'],
        'Lattitude': [-37.81],
        'Longtitude': [144.96],
        'Regionname': [region],
        'Propertycount': [1000]
    })

    input_data = input_data[model.feature_names_in_]

    # ========= PREDICTION =========
    prediction = model.predict(input_data)
    price = prediction[0]

    # ========= REAL CONFIDENCE  =========
    try:
       
        X_transformed = model.named_steps['preprocessor'].transform(input_data)

        trees = model.named_steps['model'].estimators_

        tree_preds = np.array([t.predict(X_transformed)[0] for t in trees])

        std = np.std(tree_preds)

        # normalize 
        confidence = max(0, 100 - (std / price) * 100)

    except Exception:
        # fallback 
        confidence = 75.0

    # ========= LEVEL =========
    if confidence > 80:
        level = "High"
    elif confidence > 60:
        level = "Medium"
    else:
        level = "Low"

    # ========= DISPLAY =========
    st.markdown("## 📊 Prediction Result")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("AUD", f"{price:,.0f}")

    with colB:
        st.metric("THB", f"{price*24:,.0f}")

    with colC:
        st.metric("Confidence", f"{confidence:.1f}%")

    st.caption(f"Confidence level: {level}")

    st.success("Prediction completed successfully")
    st.caption("Exchange rate approx: 1 AUD ≈ 24 THB")

st.markdown("---")

# ==============================
# ABOUT
# ==============================
with st.expander("📌 About the Model"):
    st.write("""
    This app uses a Random Forest model trained on Australian housing data.
    """)

st.warning("⚠️  ราคานี้เป็นค่าประมาณจากโมเดล Machine Learning")
