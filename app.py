import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the exact column order it was trained on
model = joblib.load('insurance_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("🏥 Insurance Charge Predictor")
st.write("Enter patient details to estimate hospital bill charges.")

# --- Collect input from the user ---
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", ["Female", "Male"])
smoker = st.selectbox("Smoker", ["No", "Yes"])
region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

if st.button("Predict Charges"):
    # --- Recreate the SAME encoding get_dummies() did during training ---
    input_dict = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if sex == "Male" else 0,
        'smoker_yes': 1 if smoker == "Yes" else 0,
        'region_northwest': 1 if region == "Northwest" else 0,
        'region_southeast': 1 if region == "Southeast" else 0,
        'region_southwest': 1 if region == "Southwest" else 0,
    }

    input_df = pd.DataFrame([input_dict])

    # Reorder columns to EXACTLY match what the model was trained on
    input_df = input_df[model_columns]

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated hospital bill: ₹{prediction:,.2f}")
