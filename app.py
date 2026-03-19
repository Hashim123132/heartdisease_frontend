import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Heart Stroke Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0f0f0f;
    color: #ffffff;
}
h1 {color: #ff4b4b; text-align: center; font-family: 'Arial Black', sans-serif;}
.stMarkdown p {color: #c0c0c0; font-size: 18px;}
div.stButton > button:first-child {
    background-color: #1f77b4; color: white; font-size: 20px;
    height: 50px; width: 220px; border-radius: 12px; border: 2px solid #ffffff;
}
</style>
""", unsafe_allow_html=True)

# --- TITLE AND DESCRIPTION ---
st.markdown("<h1>❤️ Heart Stroke Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p>Provide the following details to check your heart stroke risk:</p>", unsafe_allow_html=True)

# --- INPUT FORM WITH EXPLANATIONS ---
with st.container():
    col1, col2 = st.columns(2)

    age = col1.slider("Age", 18, 100, 40)
    sex = col2.selectbox("Sex", ["M", "F"])

    chest_pain = col1.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    
    resting_bp = col2.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    
    cholesterol = col1.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    
    fasting_bs = col2.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1],
        help="Shows if blood sugar is high after fasting (>120 mg/dL). High sugar increases heart risk."
    )

    resting_ecg = col1.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"],
        help="Measures heart’s electrical activity at rest to detect irregular rhythms or strain."
    )

    max_hr = col2.slider("Max Heart Rate", 60, 220, 150)
    
    exercise_angina = col1.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"],
        help="Indicates if chest pain occurs during physical activity, showing stress-related heart issues."
    )

    oldpeak = col2.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    
    st_slope = col1.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"],
        help="Slope of ST segment on ECG during exercise; flat/down slopes suggest heart problems."
    )

# --- PREDICTION BUTTON ---
if st.button("Predict"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.markdown("---")
    if prediction == 1:
        st.markdown("<h2 style='color:#ff4b4b; text-align:center;'>⚠️ High Risk of Heart Disease</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color:#2ecc71; text-align:center;'>✅ Low Risk of Heart Disease</h2>", unsafe_allow_html=True)