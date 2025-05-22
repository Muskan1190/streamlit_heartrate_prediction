import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ----------------------------------
# Load model and column structure
# ----------------------------------
@st.cache_resource
def load_model_and_columns():
    model = joblib.load("model.pkl")
    columns = joblib.load("columns.pkl")
    return model, columns

# ----------------------------------
# Handle sidebar inputs + reset
# ----------------------------------
def render_sidebar():
    st.sidebar.title("Patient Information")

    if st.sidebar.button("🔄 Reset All Inputs"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()

    age = st.sidebar.slider("Age", 20, 80, 50, key="age")
    sex = st.sidebar.radio("Sex", ["Male", "Female"], key="sex")
    chest_pain = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY"], key="chest_pain")
    cholesterol = st.sidebar.slider("Cholesterol Level", 100, 600, 200, key="cholesterol")
    fasting_bs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"], key="fasting_bs")
    resting_ecg = st.sidebar.selectbox("Resting ECG Result", ["LVH", "ST"], key="resting_ecg")
    maxhr = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150, key="maxhr")
    exercise_angina = st.sidebar.radio("Exercise-Induced Angina", ["Yes", "No"], key="exercise_angina")
    oldpeak = st.sidebar.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1, key="oldpeak")
    st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"], key="st_slope")

    return {
        "age": age,
        "sex": sex,
        "chest_pain": chest_pain,
        "cholesterol": cholesterol,
        "fasting_bs": fasting_bs,
        "resting_ecg": resting_ecg,
        "max_hr": max_hr,
        "exercise_angina": exercise_angina,
        "oldpeak": oldpeak,
        "st_slope": st_slope
    }

# ----------------------------------
# Prepare input dataframe
# ----------------------------------
def build_input_dataframe(inputs):
    data = {
        "age": inputs["age"],
        "sex": 1 if inputs["sex"] == "Male" else 0,
        "cholesterol": inputs["cholesterol"],
        "fastingbs": 1 if inputs["fasting_bs"] == "Yes" else 0,
        "maxhr": inputs["max_hr"],
        "oldpeak": inputs["oldpeak"],
        "exerciseangina": 1 if inputs["exercise_angina"] == "Yes" else 0,
        f"chestpaintype_{inputs['chest_pain'].lower()}": 1,
        f"restingecg_{inputs['resting_ecg'].lower()}": 1,
        f"st_slope_{inputs['st_slope'].lower()}": 1
    }

    # Fill missing dummy variables with 0
    for col in [
        "chestpaintype_asy", "chestpaintype_ata", "chestpaintype_nap",
        "restingecg_lvh", "restingecg_st",
        "st_slope_flat", "st_slope_up"
    ]:
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

# ----------------------------------
# Prediction logic
# ----------------------------------
def make_prediction(model, input_df, expected_columns):
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return prediction, probability

# ----------------------------------
# Radar chart display
# ----------------------------------
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

# ----------------------------------
# Display result to user
# ----------------------------------
def display_result(prediction, probability):
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"High risk of heart disease.\n\nEstimated probability: **{probability:.2f}**")
    else:
        st.success(f"Low risk of heart disease.\n\nEstimated probability: **{1 - probability:.2f}**")

# ----------------------------------
# Main app function
# ----------------------------------
def main():
    st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
    st.title("Heart Disease Risk Estimator")
    st.markdown("This tool predicts the likelihood of heart disease based on your medical inputs using a trained logistic regression model.")

    model, expected_columns = load_model_and_columns()
    user_inputs = render_sidebar()
    input_df, radar_data = build_input_dataframe(user_inputs)
    prediction, probability = make_prediction(model, input_df, expected_columns)

    display_result(prediction, probability)
    st.subheader("Summary of Your Input")
    st.write("Radar Data Preview:", radar_data)
    display_radar_chart(radar_data)

# ----------------------------------
# Run the app
# ----------------------------------
if __name__ == "__main__":
    main()
