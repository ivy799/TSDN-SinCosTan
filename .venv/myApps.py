import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pyreadstat
from joblib import load

# Load dataset
data, meta = pyreadstat.read_sav('Dataset Final.sav')

# Load pre-trained XGBoost model
xgb_model = load('xgb_model.joblib')

# EDA
st.title("Obesity Prediction App")
st.write("## Exploratory Data Analysis")

st.write("### Dataset Overview")
st.write(data.head())

st.write("### Dataset Description")
st.write(data.describe())

# Handle missing values if any
if data.isnull().sum().sum() > 0:
    st.warning("The dataset contains null values. These will be dropped.")
    data = data.dropna()

# Drop unnecessary columns
columns_to_drop = ['Pulau', 'B1R1', 'weight_final', 'filter_$']
data = data.drop(columns=columns_to_drop, errors='ignore')

# Define features and target
X = data.drop(columns=['Y'])
y = data['Y']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prediction using loaded XGBoost model
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"### Model Accuracy (XGBoost): {accuracy:.2f}")

# User input
st.write("## Input Features")
input_data = []
for i in range(X.shape[1]):
    input_data.append(st.number_input(f"Feature X{i+1}", min_value=0.0, max_value=100.0, step=0.1))

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    prediction = xgb_model.predict(input_df)
    st.write(f"### Prediction: {'Obese' if prediction[0] == 1 else 'Not Obese'}")
