# 🧠 Stroke Predictor Streamlit App

This is a Streamlit web app that predicts the likelihood of stroke in patients based on their clinical and demographic data.

## 🚀 Features
- 🔍 Uses Logistic Regression and XGBoost classifiers
- ⚖️ Handles class imbalance using SMOTE
- 🧠 Explains XGBoost predictions with SHAP
- 📊 Clean, modern UI with light/dark mode toggle
- 📁 Models loaded from `.pkl` files

## 📂 Files
- `stroke_predictor_app_styled.py` – main Streamlit app
- `xgb_model.pkl`, `log_model.pkl` – pre-trained models
- `scaler.pkl`, `encoder.pkl` or `preprocessor.pkl` – preprocessing pipeline
- `healthcare-dataset-stroke-data.csv` – original dataset

## 📦 Installation
```bash
pip install -r requirements.txt
