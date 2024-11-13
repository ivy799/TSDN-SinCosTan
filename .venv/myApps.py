import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pyreadstat
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
# Cache the model loading
@st.cache(allow_output_mutation=True)
def load_model():
    return load('xgb_model.joblib')

# Contoh HTML
html_content = """
<div style='background-color:lightblue; padding:10px;'>
    <h2>Selamat Datang di Aplikasi Saya</h2>
    <p>Ini adalah contoh teks HTML yang disisipkan dalam aplikasi Streamlit.</p>
</div>
"""

# Menampilkan HTML di Streamlit
st.markdown(html_content, unsafe_allow_html=True)

xgb_model = load_model()

# Cache the dataset loading and preprocessing
@st.cache
def load_and_preprocess_data():
    data, meta = pyreadstat.read_sav('Dataset Final.sav')
    if data.isnull().sum().sum() > 0:
        data = data.dropna()
    columns_to_drop = ['Pulau', 'B1R1', 'weight_final', 'filter_$']
    data = data.drop(columns=columns_to_drop, errors='ignore')
    X = data.drop(columns=['Y'])
    y = data['Y']
    return train_test_split(X, y, test_size=0.2, random_state=42), X, data

(X_train, X_test, y_train, y_test), X, data = load_and_preprocess_data()

# EDA
st.title("Obesity Prediction App")
st.write("## Exploratory Data Analysis")

# Dataset Overview
st.write("### Dataset Overview")
st.write(data.head())

# Dataset Description
st.write("### Dataset Description")
st.write(data.describe())

# Visualizations
st.write("## Data Visualizations")

# Histogram
st.write("### Histogram of Features")
fig, ax = plt.subplots(figsize=(10, 6))
data.hist(ax=ax)
st.pyplot(fig)

# Box Plot
st.write("### Box Plot of Features")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=data, ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

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
