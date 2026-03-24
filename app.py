import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="House Price AI", page_icon="🏠", layout="wide")

# LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load("house_model.pkl")

model = load_model()

# HEADER
st.title("🏠 House Price Prediction AI")
st.caption("Predict house price using Machine Learning")

st.markdown("---")

# INPUT UI
st.subheader("📊 Property Information")

col1, col2, col3 = st.columns(3)

with col1:
    rooms = st.number_input("Rooms", 1, 10, 3)
    bedroom2 = st.number_input("Bedrooms", 0, 10, 3)
    bathroom = st.number_input("Bathrooms", 1, 5, 2)

with col2:
    car = st.number_input("Car Spaces", 0, 5, 1)
    landsize = st.number_input("Land Size", 1, 5000, 500)
    buildingarea = st.number_input("Building Area", 1, 1000, 100)

with col3:
    yearbuilt = st.number_input("Year Built", 1800, 2025, 2000)
    distance = st.number_input("Distance", 0.0, 50.0, 10.0)
    region = st.selectbox("Region", ["Northern", "Western", "Southern", "Eastern"])

st.markdown("---")

# PREDICT
if st.button("🚀 Predict Price", use_container_width=True):

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

    # กัน error feature mismatch
    try:
        input_data = input_data[model.feature_names_in_]
    except:
        pass

    # ===== PREDICT =====
    prediction = model.predict(input_data)[0]

    # ===== CONFIDENCE =====
    try:
        X_transformed = model.named_steps['preprocessor'].transform(input_data)
        trees = model.named_steps['model'].estimators_
        tree_preds = np.array([t.predict(X_transformed)[0] for t in trees])
        std = np.std(tree_preds)
        confidence = max(0, 100 - (std / prediction) * 100)
    except:
        confidence = 75

    # ===== RESULT =====
    st.subheader("📊 Prediction Result")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Price (AUD)", f"{prediction:,.0f}")

    with colB:
        st.metric("Price (THB)", f"{prediction*24:,.0f}")

    with colC:
        st.metric("Confidence", f"{confidence:.1f}%")

    # ===== INSIGHT (โบนัส) =====
    st.markdown("### 🧠 Model Insight")

    if distance > 20:
        st.info("📍 บ้านอยู่ไกลเมือง → ราคามักลดลง")
    if buildingarea > 200:
        st.info("🏠 พื้นที่บ้านใหญ่ → ราคามักสูงขึ้น")
    if rooms >= 5:
        st.info("🛏️ ห้องเยอะ → ราคามักสูงขึ้น")

    # ===== FEATURE IMPORTANCE (โบนัสโคตรสำคัญ) =====
    st.markdown("### 🔍 Feature Importance")

    try:
        importances = model.named_steps["model"].feature_importances_
        features = model.named_steps["preprocessor"].get_feature_names_out()

        feat_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(feat_df.set_index("Feature"))
    except:
        st.warning("Feature importance not available")

    st.success("Prediction completed!")

st.markdown("---")

st.caption("⚠️ This is an estimated price using Machine Learning")
