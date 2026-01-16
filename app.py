import streamlit as st
import joblib 
import numpy as np

model=joblib.load('house_price_model.pkl')

st.title("🏠 House Price Prediction App Made by Pujara Ridham")
features=model.feature_names_in_

st.set_page_config(page_title="House Price Predictor")
st.write("Enter house details below:""Enter house details below:")
input_values=[]
for feature in features:
    val=st.number_input( 
        feature,
        value=0.0,
        key=f"input_{feature}",
        )
    
    input_values.append(val)
if st.button("Predict Price"):
    input_array=np.array(input_values).reshape(1, -1)
    prediction=model.predict(input_array)
    st.success(f"💰 Predicted House Price: {prediction[0]:.2f}")