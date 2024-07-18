
import streamlit as st
import pickle
import numpy as np

# Load the trained RandomForest model and scaler
with open('diabetes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('sc.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Create the web app
st.title('Diabetes Prediction App')

# Input fields
Pregnancies = st.number_input('Pregnancies', min_value=0.0, max_value=20.0, value=0.0, step=1.0)
Glucose = st.number_input('Glucose', min_value=0.0, max_value=400.0, value=30.0, step=1.0)
BloodPressure = st.number_input('BloodPressure', min_value=0.0, max_value=200.0, value=0.0, step=1.0)
SkinThickness = st.number_input('SkinThickness', min_value=0.0, max_value=200.0, value=0.0, step=1.0)
Insulin = st.number_input('Insulin', min_value=0.0, max_value=700.0, value=0.0, step=1.0)
DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=2.0, value=0.0, step=0.01)
BMI = st.number_input('BMI', min_value=0.0, max_value=150.0, value=20.0, step=0.1)
Age = st.number_input('Age', min_value=15.0, max_value=100.0, value=25.0, step=1.0)

# Prepare the feature vector
features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]], dtype=np.float64)

# Scale the features
features_scaled = scaler.transform(features)

# Predict Diabetes
predicted_Diabetes = model.predict(features_scaled)

prediction_label = "Yes" if predicted_Diabetes[0] == 1 else "No"
st.write(f'Predicted Diabetes: {prediction_label}')
