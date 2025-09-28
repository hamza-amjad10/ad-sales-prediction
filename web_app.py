import numpy as np
import streamlit as st
import joblib


model=joblib.load("ad_sales_model.pkl")
scaler=joblib.load("ad_sales_scaler.pkl")

st.title("Ad Sales Prediction Web App")

tv=st.text_input("Enter the amount spent on TV Ads")
radio=st.text_input("Enter the amount spent on Radio Ads")
newspaper=st.text_input("Enter the amount spent on Newspaper Ads")


if st.button("Predict Sales"):
    features=np.array([[tv,radio,newspaper]])
    features_scaled=scaler.transform(features)
    predicted_value=model.predict(features_scaled)
    st.write("The predicted sales is: ", round(predicted_value[0],2))
