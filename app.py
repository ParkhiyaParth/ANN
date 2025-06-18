import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

model=load_model('model.h5')

with open('label_encoder.pkl','rb') as file:
    label_encoder=pickle.load(file)
    
with open('one_hot_encoder.pkl','rb') as file:
    one_hot_encoder=pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)    


st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn based on their account information.")
st.write("Please enter the following details:")
# Input fields

credit_score = st.number_input("Credit Score")    
Geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
Gender=st.selectbox("Gender",["Male","Female"])
Age = st.slider("Age", min_value=18, max_value=100, value=30)
Tenure = st.slider("Tenure (in years)", min_value=0, max_value=10, value=1)
Balance = st.number_input("Balance", min_value=0.0, max_value=100000.0, value=5000.0)
NumOfProducts = st.slider("Number of Products", min_value=1, max_value=4, value=1)
HasCrCard = st.selectbox("Has Credit Card", [1, 0], index=0)
IsActiveMember = st.selectbox("Is Active Member", [1, 0], index=0)
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, max_value=1000000.0, value=50000.0)

input_data=pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':label_encoder.transform([Gender])[0],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary],
})

geo_data=pd.DataFrame(one_hot_encoder.transform([[Geography]]).toarray(),columns=one_hot_encoder.get_feature_names_out(['Geography']))
input_data=pd.concat([input_data,geo_data],axis=1)

scaled_data=scaler.transform(input_data)

prediction=model.predict(scaled_data)


# Display the prediction result
if st.button("Predict"):
    if prediction[0][0] > 0.5:
        st.success("The customer is likely to churn.")
        
        st.write("Predicted Probability of Churn:", prediction[0][0])
    else:
        st.success("The customer is likely to stay.")
        
        st.write("Predicted Probability of Churn:", prediction[0][0])
