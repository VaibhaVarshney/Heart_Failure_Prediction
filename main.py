import streamlit as st
import pickle


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Failure Prediction",
    layout="wide"
)


# --------------------------------------------------
# Load model & scaler
# --------------------------------------------------
model_randomforest = pickle.load(open('randomforest_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_heart_failure(
    age, sex, chest_pain, resting_bp, cholesterol,
    fasting_bs, rest_ecg, max_hr, exercise_angina,
    oldpeak, st_slope
):
    sex = 1 if sex == 'M' else 0
    chest_pain = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}[chest_pain]
    rest_ecg = {'Normal': 0, 'ST': 1, 'LVH': 2}[rest_ecg]
    exercise_angina = 1 if exercise_angina == 'Y' else 0
    st_slope = {'Up': 0, 'Flat': 1, 'Down': 2}[st_slope]

    data = [[
        age, sex, chest_pain, resting_bp, cholesterol,
        fasting_bs, rest_ecg, max_hr, exercise_angina,
        oldpeak, st_slope
    ]]

    data = scaler.transform(data)
    result = model_randomforest.predict(data)

    return (
        "Person Having Heart Disease"
        if result[0] == 1
        else "Person Not Having Heart Disease"
    )


# --------------------------------------------------
# Sidebar navigation (SAFE)
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "",
    ["Home", "Documentation", "Contact", "About Us"]
)


# --------------------------------------------------
# HOME
# --------------------------------------------------
if page == "Home":
    st.title("Heart Failure Prediction")

    age = st.number_input("Age", 0, 120, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure", 0, 200, 120)
    cholesterol = st.number_input("Cholesterol", 0, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Maximum Heart Rate", 0, 220, 150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    if st.button("Predict"):
        result = predict_heart_failure(
            age, sex, chest_pain, resting_bp, cholesterol,
            fasting_bs, rest_ecg, max_hr,
            exercise_angina, oldpeak, st_slope
        )

        with st.expander("Prediction Result", expanded=True):
            st.subheader(result)
            st.write("⚠️ This is a machine learning prediction. Consult a physician.")

        if result == "Person Not Having Heart Disease":
            st.balloons()
        else:
            st.error("Please consult a physician immediately.")


# --------------------------------------------------
# DOCUMENTATION
# --------------------------------------------------
elif page == "Documentation":
    st.title("Documentation")

    st.markdown("""
    ### Heart Failure Prediction System

    This system uses a **Random Forest Classifier** trained on clinical health data.

    **Features used**
    - Age
    - Sex
    - Chest Pain Type
    - Blood Pressure
    - Cholesterol
    - Fasting Blood Sugar
    - ECG
    - Heart Rate
    - Exercise Angina
    - Oldpeak
    - ST Slope
    """)


# --------------------------------------------------
# ABOUT US
# --------------------------------------------------
elif page == "About Us":
    st.title("About Us")

    st.markdown("""
    We are focused on applying machine learning to healthcare.

    **Team**
    - Vaibhav Varshney — Data Scientist

    📧 varshney2vaibhav@gmail.com
    """)


# --------------------------------------------------
# CONTACT
# --------------------------------------------------
elif page == "Contact":
    st.title("Contact")

    st.markdown("""
    📧 **Email:** varshney2vaibhav@gmail.com  
    📞 **Phone:** +91 8287907911  
    📍 **Location:** Delhi 110072
    """)
