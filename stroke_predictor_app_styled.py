
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb


# Load models and transformers
log_model = joblib.load("log_model.pkl")
scaler = joblib.load("scaler.pkl")
preprocessor = joblib.load("preprocessor.pkl")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgb_model.json")



# Page configuration
st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🧠", layout="centered")

# 🌗 Theme toggle
if "theme" not in st.session_state:
    st.session_state.theme = "light"

theme = st.radio("🌗 Toggle Theme", options=["light", "dark"], index=0 if st.session_state.theme == "light" else 1)
st.session_state.theme = theme

# 🎨 Dynamic CSS
light_css = '''
<style>
    .stApp {
        background-color: #f5f7fa;
        color: #000;
    }
    h1, h4, h2 {
        color: #003366;
    }
    .stButton>button {
        background-color: #003366;
        color: white;
        border-radius: 6px;
    }
    .stButton>button:hover {
        background-color: #00509e;
        color: white;
    }
</style>
'''

dark_css = '''
<style>
    .stApp {
        background-color: #1e1e1e;
        color: #f1f1f1;
    }
    h1, h4, h2 {
        color: #66ccff;
    }
    .stButton>button {
        background-color: #66ccff;
        color: #000;
        border-radius: 6px;
    }
    .stButton>button:hover {
        background-color: #3399cc;
        color: white;
    }
</style>
'''

st.markdown(light_css if st.session_state.theme == "light" else dark_css, unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>🧠 Stroke Risk Prediction Tool</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Predict stroke risk using patient health data</h4>", unsafe_allow_html=True)
st.markdown("---")

# Input Form
with st.form("input_form"):
    st.subheader("🔍 Enter Patient Information")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 0, 100, 45)
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Other"])
    with col2:
        Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=105.0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=26.5)
        smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
        hypertension = st.selectbox("Hypertension", [0, 1], help="1 if patient has hypertension, 0 otherwise")
        heart_disease = st.selectbox("Heart Disease", [0, 1], help="1 if patient has any heart disease, 0 otherwise")


    submitted = st.form_submit_button("🔎 Predict Risk")

    if submitted:
        # Feature engineering
        is_obese = int(bmi > 30)
        is_underweight = int(bmi < 18.5)
        is_diabetic = int(avg_glucose_level > 126)

        age_group = pd.cut([age], bins=[0, 40, 60, 80, 120], labels=["<40", "40-60", "60-80", "80+"])[0]
        glucose_bin = pd.cut([avg_glucose_level], bins=[0, 90, 126, 200, 300],
                             labels=["normal", "pre-diabetic", "diabetic", "high-risk"])[0]
        bmi_bin = pd.cut([bmi], bins=[0, 18.5, 25, 30, 50],
                         labels=["underweight", "normal", "overweight", "obese"])[0]

                # Ask for hypertension and heart_disease in the form
        hypertension = st.selectbox("Hypertension", [0, 1], help="1 if patient has hypertension, 0 otherwise", key="hypertension_select")
        heart_disease = st.selectbox("Heart Disease", [0, 1], help="1 if patient has any heart disease, 0 otherwise", key="heart_disease_select")

        input_dict = {
            "gender": [gender],
            "ever_married": [ever_married],
            "work_type": [work_type],
            "Residence_type": [Residence_type],
            "smoking_status": [smoking_status],
            "age_group": [age_group],
            "glucose_bin": [glucose_bin],
            "bmi_bin": [bmi_bin],
            "is_obese": [is_obese],
            "is_underweight": [is_underweight],
            "is_diabetic": [is_diabetic],
            "age": [age],
            "avg_glucose_level": [avg_glucose_level],
            "bmi": [bmi],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease]
        }

        input_df = pd.DataFrame(input_dict)
        processed_input = preprocessor.transform(input_df)
        scaled_input = scaler.transform(processed_input)

        xgb_pred_proba = xgb_model.predict_proba(processed_input)[0][1]
        log_pred_proba = log_model.predict_proba(scaled_input)[0][1]

        # Output Metrics
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        col1, col2 = st.columns(2)
        col1.metric("XGBoost Stroke Risk (%)", f"{xgb_pred_proba * 100:.2f}")
        col2.metric("Logistic Regression Risk (%)", f"{log_pred_proba * 100:.2f}")

        risk_level = "🟢 Low Risk" if xgb_pred_proba < 0.3 else ("🟠 Moderate Risk" if xgb_pred_proba < 0.7 else "🔴 High Risk")
        st.markdown(f"<h4 style='color:#003366;'>Risk Category (XGBoost): {risk_level}</h4>", unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Built using machine learning models trained on the Kaggle Stroke Prediction dataset.")
