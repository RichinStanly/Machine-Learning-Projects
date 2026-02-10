import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("💳 Credit Card Approval Predictor")
st.write("Enter the applicant's details below to check if they should get a credit card.")

@st.cache_data
def get_model():
    data = pd.read_csv('Credit_Card_Applications.csv')

    fraud = data[data['Class'] == 1]
    normal = data[data['Class'] == 0]
    normal_sample = normal.sample(n=len(fraud), random_state=42)
    new_data = pd.concat([normal_sample, fraud], axis=0)
    
    X = new_data.drop(['Class', 'CustomerID'], axis=1)
    y = new_data['Class']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

try:
    model = get_model()
    st.success("Model loaded successfully! ✅")
except Exception as e:
    st.error(f"Error loading data: {e}")

st.sidebar.header("Applicant Details")

def user_input_features():
    # We create sliders/inputs for A1 to A15 based on your dataset ranges
    A1 = st.sidebar.selectbox("Gender (A1)", (0, 1))
    A2 = st.sidebar.number_input("Age (A2)", min_value=13.0, max_value=80.0, value=30.0)
    A3 = st.sidebar.number_input("Debt (A3)", min_value=0.0, max_value=28.0, value=5.0)
    A4 = st.sidebar.selectbox("Marital Status (A4)", (1, 2, 3))
    A5 = st.sidebar.selectbox("Bank Customer (A5)", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
    A6 = st.sidebar.selectbox("Education Level (A6)", (1, 2, 3, 4, 5, 6, 7, 8, 9))
    A7 = st.sidebar.number_input("Years Employed (A7)", min_value=0.0, max_value=30.0, value=2.0)
    A8 = st.sidebar.selectbox("Prior Default (A8)", (0, 1))
    A9 = st.sidebar.selectbox("Employed (A9)", (0, 1))
    A10 = st.sidebar.number_input("Credit Score (A10)", min_value=0, max_value=70, value=5)
    A11 = st.sidebar.selectbox("Drivers License (A11)", (0, 1))
    A12 = st.sidebar.selectbox("Citizen (A12)", (1, 2, 3))
    A13 = st.sidebar.number_input("Zip Code (A13)", min_value=0, max_value=2000, value=100)
    A14 = st.sidebar.number_input("Income (A14)", min_value=0, max_value=100000, value=500)
    
    data = {'A1': A1, 'A2': A2, 'A3': A3, 'A4': A4, 'A5': A5, 'A6': A6, 'A7': A7, 
            'A8': A8, 'A9': A9, 'A10': A10, 'A11': A11, 'A12': A12, 'A13': A13, 'A14': A14}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("User Input parameters")
st.write(input_df)


if st.button("Predict Approval"):
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.success("🎉 Application APPROVED!")
    else:
        st.error("❌ Application REJECTED")