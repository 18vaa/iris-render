# app.py
import streamlit as st
import pandas as pd
import joblib
import logging

# Load model
model = joblib.load('iris_model.pkl')

st.title("🌸 Iris Flower Classifier")

# Inputs
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

# Predict
features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                        columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
prediction = model.predict(features)[0]

# Class labels
labels = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
st.write(f"### Prediction: {labels[prediction]}")




logging.basicConfig(filename="logs.txt", level=logging.INFO)
logging.info(f"Input: {features.values.tolist()}, Prediction: {labels[prediction]}")

