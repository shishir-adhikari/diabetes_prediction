# diabestes_predictor.py
import pickle
import streamlit as st 
from streamlit_option_menu import option_menu

# loading the saved model
diabetes_model = pickle.load(open('./diabetes_model.sav', 'rb'))

# Set the title of the app
st.title("Diabetes Prediction App")

# Page content

Pregnancies = st.text_input('Number of Pregnancies')
Glucose = st.text_input('Glucose Level')
BloodPressure = st.text_input('Blood Pressure Value')
SkinThickness = st.text_input('Skin Thickness Value')
Insulin = st.text_input('Insulin Level')
BMI = st.text_input('BMI')
DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
Age = st.text_input('Age')

# Code for Prediction

diab_diagnosis = ''

# creating a button for prediction

if st.button('Get Prediction Result'):
    diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    if (diab_prediction[0] == 1): 
        diab_diagnosis = 'The person is Diabetic'

    else:
        diab_diagnosis = 'The person is NOT Diabetic'

st.success(diab_diagnosis)


