import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

@st.cache_resource
def load_model_and_columns():
    model = joblib.load("model.pkl")
    columns = joblib.load("columns.pkl")
    return model, columns

def render_sidebar():
    st.sidebar.title("🧾 Patient Information")
    st.sidebar.markdown("Adjust the patient's clinical measurements:")

    if st.sidebar.button("\U0001F504 Reset All Inputs"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()

    age = st.sidebar.slider("Age", 20, 80, 50, key="age")
    sex = st.sidebar.radio("Sex", ["Male", "Female"], key="sex")
    chest_pain = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY"], key="chest_pain")
    cholesterol = st.sidebar.slider("Cholesterol", 100, 600, 200, key="cholesterol")
    fasting_bs = st.sidebar.radio("Fasting Blood Sugar", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="fasting_bs")
    max_hr = st.sidebar.slider("❤ Max Heart Rate", 60, 220, 150, key="max_hr")
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1, key="oldpeak")
    exercise_angina = st.sidebar.radio(" Exercise Induced Angina", ["Yes", "No"], key="exercise_angina")
    st_slope = st.sidebar.selectbox(" ST Slope", ["Flat", "Up", "Down"], key="st_slope")
    resting_ecg = st.sidebar.selectbox(" Resting ECG", ["Normal", "ST", "LVH"], key="resting_ecg")

    return {
        "age": age,
        "sex": sex,
        "chest_pain": chest_pain,
        "cholesterol": cholesterol,
        "fasting_bs": fasting_bs,
        "max_hr": max_hr,
        "oldpeak": oldpeak,
        "exercise_angina": exercise_angina,
        "st_slope": st_slope,
        "resting_ecg": resting_ecg
    }

def build_input_dataframe(inputs):
    data = {
        "age": inputs["age"],
        "sex_m": 1 if inputs["sex"] == "Male" else 0,
        "cholesterol": inputs["cholesterol"],
        "fastingbs": int(inputs["fasting_bs"]),
        "maxhr": inputs["max_hr"],
        "oldpeak": inputs["oldpeak"],
        "exerciseangina_y": 1 if inputs["exercise_angina"] == "Yes" else 0,
        f"chestpaintype_{inputs['chest_pain'].lower()}": 1,
        f"restingecg_{inputs['resting_ecg'].lower()}": 1,
        f"st_slope_{inputs['st_slope'].lower()}": 1
    }

    expected_dummies = [
        "chestpaintype_asy", "chestpaintype_nap",
        "restingecg_st", "restingecg_lvh",
        "st_slope_flat", "st_slope_down"
    ]
    for col in expected_dummies:
        if col not in data:
            data[col] = 0

    df = pd.DataFrame([data])
    radar_data = {
        "Age": inputs["age"],
        "Cholesterol": inputs["cholesterol"],
        "MaxHR": inputs["max_hr"],
        "Oldpeak": inputs["oldpeak"]
    }

    return df, radar_data

def make_prediction(model, input_df, expected_columns):
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return prediction, probability

def display_radar_chart(data):
    labels = list(data.keys())
    values = list(data.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='crimson', linewidth=2)
    ax.fill(angles, values, color='crimson', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Input Feature Profile")
    st.pyplot(fig)

def display_result(prediction, probability):
    st.subheader("Cell Cluster Prediction")
    label = "Positive" if prediction == 1 else "Negative"
    st.markdown(f"**Model Prediction:** {label}")

    benign = 1 - probability
    malignant = probability

    if malignant >= 0.5:
        st.error(f"\U0001F6A8 High risk of heart disease.\n\n**Probability of high risk:** {malignant:.2f}")
    else:
        st.success(f"\u2705 Low risk of heart disease.\n\n**Probability of low risk:** {benign:.2f}")

    st.info(
        "⚠️ This app can assist medical professionals in evaluating risk, "
        "but should not be used as a substitute for professional diagnosis."
    )

def main():
    st.set_page_config(page_title="Heart Disease Estimator", layout="wide")
    st.title("🫀 Heart Disease Diagnosis")

    st.markdown(
        "This app predicts the likelihood of heart disease using a trained machine learning model based on patient medical data. "
        "Adjust the values in the sidebar to update the prediction and radar chart in real time."
    )

    model, expected_columns = load_model_and_columns()
    user_inputs = render_sidebar()
    input_df, radar_data = build_input_dataframe(user_inputs)
    prediction, probability = make_prediction(model, input_df, expected_columns)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Feature Profile Radar")
        display_radar_chart(radar_data)

    with col2:
        display_result(prediction, probability)

if __name__ == "__main__":
    main()
