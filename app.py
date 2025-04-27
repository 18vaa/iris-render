import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import logging

# Setup logging
logging.basicConfig(filename="logs.txt", level=logging.INFO)

# Load dataset and train model
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

model = RandomForestClassifier()
model.fit(X, y)

st.title("🌸 Iris Flower Classifier")

# Inputs
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

# Predict
features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                        columns=iris.feature_names)
prediction = model.predict(features)[0]

# Class labels
labels = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
st.write(f"### Prediction: {labels[prediction]}")

# Log prediction
logging.info(f"Input: {features.values.tolist()}, Prediction: {labels[prediction]}")
