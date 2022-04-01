# model deployment using all in one method

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import load_model
savedModel=load_model('model_keras.h5')

with open("preprocessor.pkl", "rb") as model_file:
    preprocessor = pickle.load(model_file)

columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents','tenure','PhoneService','MultipleLines'
        ,'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV'
        ,'StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']


st.title("Churn Prediction")
gender = st.selectbox("gender", ['Male', 'Female'])
SeniorCitizen = st.selectbox("SeniorCitizen", ['No', 'Yes'])
SeniorCitizen = 'no' if SeniorCitizen == 'No' else 'yes'
Partner = st.selectbox("Partner", ['No', 'Yes'])
Dependents = st.selectbox("Dependents", ['No', 'Yes',])
tenure = st.number_input("Berapa lama Anda bersama kita dalam bulan?")
PhoneService = st.selectbox("PhoneService", ['Yes', 'No','No phone service'])
MultipleLines = st.selectbox("MultipleLines", ['No', 'Yes'])
InternetService = st.selectbox("InternetService", ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("OnlineSecurity", ['No', 'Yes','No internet service'])
OnlineBackup = st.selectbox("OnlineBackup", ['No', 'Yes','No internet service'])
DeviceProtection = st.selectbox("DeviceProtection", ['No', 'Yes','No internet service'])
TechSupport = st.selectbox("TechSupport", ['No', 'Yes','No internet service'])
StreamingTV = st.selectbox("StreamingTV", ['No', 'Yes','No internet service'])
StreamingMovies = st.selectbox("StreamingMovies", ['No', 'Yes','No internet service'])
Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox("PaperlessBilling", ['No', 'Yes'])
PaymentMethod = st.selectbox("PaymentMethod", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
MonthlyCharges = st.number_input("Tagihan Bulanan")
TotalCharges = st.number_input("Tagihan Total")


# inference
new_data = [gender, SeniorCitizen, Partner, Dependents,tenure,PhoneService,MultipleLines
        ,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV
        ,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges]
new_data = pd.DataFrame([new_data], columns=columns)
new_data = preprocessor.transform(new_data)
st.dataframe(new_data)
res = savedModel.predict(new_data).item()
res = np.where(res < 0.5, 'No Churn','Churn')
st.title(res)