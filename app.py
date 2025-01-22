import pandas as pd
import numpy as np 
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import streamlit as st

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoder and scaler
with open('labelencoder_gender.pkl', 'rb') as file:
    label_encoder_gen = pickle.load(file)

with open('OneHotEncoder_Geography', 'rb') as file:
    OneHot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# streamlit app
st.title('Customer churn prediction')

# user input
geography = st.selectbox('Geography', OneHot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gen.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure')
num_of_products = st.slider('Number  of Products', 1, 4)
has_cr_card = st.selectbox('Has credit card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the Input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gen.transform([gender])[0]],
    'Age': [age],
    'Tenure':[tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-Hot encode Geography
geo_encoded = OneHot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=OneHot_encoder_geo.get_feature_names_out(['Geography']))

# combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# scale the input data
input_data_scaled = scaler.transform(input_data)

# predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'churn probability: {prediction_proba: .2f}')

if prediction_proba > 0.5:
    st.write('the customer is likely to churn')

else:
    st.write('The customer is not likely to churn')