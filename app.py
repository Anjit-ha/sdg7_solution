import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved scaler and model
scaler = joblib.load('scaler.pkl')
rf_model = joblib.load('random_forest_model.pkl')

# Page Configuration
st.set_page_config(page_title="Clean Energy Price Predictor", page_icon="🌿", layout="centered")

# Title and description
st.title("🌿 Clean Energy Price Predictor")
st.markdown("""
Welcome to the **Clean Energy Price Predictor** — a tool that supports **Sustainable Development Goal 7 (SDG7)** by promoting access to **affordable, reliable, and modern energy**.

This application predicts the **purchasing price (dollar/kWh)** based on energy demand and fuel cost factors, helping different types of users make informed, energy-smart decisions.
""")

st.divider()

# Step 1: User type
st.subheader("1. Who are you?")
user_type = st.selectbox("Choose your user type:", ["Household User", "Energy Manager", "Policy Planner"])

st.divider()

# Step 2: Input features
st.subheader("2. Enter Energy and Price Information")

col1, col2 = st.columns(2)
with col1:
    unmet = st.number_input('🔌 Unmet Energy Demand (kWh)', min_value=0.0, value=2698.0)
    hour = st.slider('🕒 Hour of Day', 0, 23, 0)
with col2:
    load = st.number_input('⚡ Load (kWh)', min_value=0.0, value=2698.0)
    natural_gas_price = st.number_input('⛽ Natural Gas Price ($/M Btu)', min_value=0.0, value=2.97)

# Step 3: Prepare data and predict
input_data = pd.DataFrame({
    'Unmet(kWh)': [unmet],
    'Load (kWh)': [load],
    'Hour': [hour],
    'Natural Gas Price ($/M Btu)': [natural_gas_price]
})

st.divider()

# Step 4: Prediction button
if st.button('📈 Predict Purchasing Price'):
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    predicted_price = rf_model.predict(input_scaled)[0]

    st.success(f"✅ Predicted Purchasing Price: **${predicted_price:.4f} per kWh**")


    # Tailored advice
    st.subheader("💡 How This Helps You:")
    if user_type == "Household User":
        advice = (
            "Lower predicted prices indicate a good time to use energy-intensive appliances. "
            "Smart scheduling can help you save money and reduce your carbon footprint."
        )
    elif user_type == "Energy Manager":
        advice = (
            "This forecast supports load optimization and can help you decide when to shift usage or store energy. "
            "It also assists in maximizing the value of integrating renewables."
        )
    elif user_type == "Policy Planner":
        advice = (
            "Use this insight to design tariff policies, plan subsidies, and build long-term energy strategies. "
            "Accurate predictions support wider access and infrastructure planning."
        )

    st.info(advice)

st.divider()

# Footer
st.markdown("""
📘 **Note:** This model aligns with **SDG7: Affordable and Clean Energy**  
By enabling better energy cost prediction, we empower users to make energy-efficient decisions that reduce emissions, optimize usage, and lower costs.
""")
