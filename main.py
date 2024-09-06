import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
from streamlit_navigation_bar import st_navbar

# Load the trained model and scaler
model_randomforest = pickle.load(open('randomforest_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Function to make a prediction
def predict_heart_failure(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, rest_ecg, max_hr, exercise_angina, oldpeak, st_slope):
    # Encode input data
    sex = 1 if sex == 'M' else 0
    chest_pain = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}[chest_pain]
    rest_ecg = {'Normal': 0, 'ST': 1, 'LVH': 2}[rest_ecg]
    exercise_angina = 1 if exercise_angina == 'Y' else 0
    st_slope = {'Up': 0, 'Flat': 1, 'Down': 2}[st_slope]
    
    # Prepare input data for prediction
    data = [[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, rest_ecg, max_hr, exercise_angina, oldpeak, st_slope]]
    data = scaler.transform(data)
    
    # Make prediction
    result = model_randomforest.predict(data)
    return 'Person Having Heart Disease' if result == [1] else 'Person Not Having Heart Disease'

styles = {
    "nav": {
        "background-color": "rgb(123, 209, 146)",
        
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(49, 51, 63)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
    
}

# Navigation
page = st_navbar(["Home", "Documentation", "Contact", "About Us"], styles=styles)

# Home Page
if page == "Home":
    st.title('Heart Failure Prediction')

    # Input form
    age = st.number_input('Age', min_value=0, max_value=120, value=40)
    sex = st.selectbox('Sex', ['M', 'F'])
    chest_pain = st.selectbox('Chest Pain Type', ['TA', 'ATA', 'NAP', 'ASY'])
    resting_bp = st.number_input('Resting Blood Pressure', min_value=0, max_value=200, value=120)
    cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200)
    fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    rest_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
    max_hr = st.number_input('Maximum Heart Rate', min_value=0, max_value=220, value=150)
    exercise_angina = st.selectbox('Exercise Induced Angina', ['Y', 'N'])
    oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

    # Prediction button
    if st.button('Predict'):
        result = predict_heart_failure(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, rest_ecg, max_hr, exercise_angina, oldpeak, st_slope)
        
        # Displaying the result in an expanded text section
        with st.expander("Prediction Result"):
            st.write(f"### Result: {result}")
            st.write(f"""
            **Summary of Inputs:**
            - **Age**: {age}
            - **Sex**: {sex}
            - **Chest Pain Type**: {chest_pain}
            - **Resting Blood Pressure**: {resting_bp}
            - **Cholesterol**: {cholesterol}
            - **Fasting Blood Sugar**: {fasting_bs}
            - **Resting ECG**: {rest_ecg}
            - **Maximum Heart Rate**: {max_hr}
            - **Exercise Induced Angina**: {exercise_angina}
            - **Oldpeak**: {oldpeak}
            - **ST Slope**: {st_slope}

            
            
            **Note:** Please consult a physician as this is a trained model which may be wrong.
            """)
        
        # Animation based on the result
        if result == 'Person Not Having Heart Disease':
            st.balloons()
        else:
            st.error("Person Having Heart Disease - Please consult a physician immediately.")
#github
# Documentation Page
elif page == "Documentation":
    st.title('Documentation')
    st.write("""
    # Heart Failure Prediction Project

    This project is designed to predict heart failure based on several input parameters. The model used is a RandomForestClassifier which has been trained on a dataset containing various health metrics. The following parameters are used for prediction:

    - **Age**: Age of the person
    - **Sex**: Gender of the person (M for Male, F for Female)
    - **Chest Pain Type**: Types of chest pain experienced (TA, ATA, NAP, ASY)
    - **Resting Blood Pressure**: Resting blood pressure (in mm Hg)
    - **Cholesterol**: Serum cholesterol in mg/dl
    - **Fasting Blood Sugar**: Whether fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    - **Resting ECG**: Resting electrocardiographic results (Normal, ST, LVH)
    - **Maximum Heart Rate**: Maximum heart rate achieved
    - **Exercise Induced Angina**: Exercise induced angina (Y = Yes, N = No)
    - **Oldpeak**: ST depression induced by exercise relative to rest
    - **ST Slope**: The slope of the peak exercise ST segment (Up, Flat, Down)

    ## How to Use the Application

    1. Navigate to the 'Home' page.
    2. Enter the required input parameters in the form provided.
    3. Click on the 'Predict' button to get the prediction result.
    
    ## About the Model

    The RandomForestClassifier model used in this project is a powerful ensemble learning algorithm that combines the predictions of multiple decision trees to improve accuracy and robustness. The model has been trained and tested on a dataset to ensure reliable predictions.
    """)

# About Us Page
elif page == "About Us":
    st.title('About Us')
    st.write("""
    # About Us

    We are a team of data scientists and healthcare professionals dedicated to leveraging machine learning to improve healthcare outcomes. Our mission is to provide accurate and reliable predictive models that assist healthcare providers in making informed decisions.

    Our team consists of:
    - **Vaibhav Varshney**: Data Scientist
    
    Contact us at [varshney2vaibhav@gmail.com](mailto:varshney2vaibhav@gmail.com) for more information.
    """)

# Contact Page
elif page == "Contact":
    st.title('Contact')
    st.write("""
    # Contact Us

    If you have any questions or need further information, please feel free to reach out to us.

    - **Email**: [varshney2vaibhav@gmail.com](mailto:varshney2vaibhav@gmail.com)
    - **Phone**: +91 8287907911
    - **Address**: Delhi 110072
    """)
