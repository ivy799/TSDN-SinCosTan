import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pyreadstat
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
import base64
from io import BytesIO

# Cache the model loading
@st.cache_resource
def load_model():
    return load('xgboost_optimized_model.pkl')

# # Contoh HTML
# html_content = """
# <div style='background-color:lightblue; padding:10px;'>
#     <h2>Selamat Datang di Aplikasi Saya</h2>
#     <p>Ini adalah contoh teks HTML yang disisipkan dalam aplikasi Streamlit.</p>
# </div>
# """

# Menampilkan HTML di Streamlit
# st.markdown(html_content, unsafe_allow_html=True)

xgb_model = load_model()

# Cache the dataset loading and preprocessing
@st.cache_data
def load_and_preprocess_data():
    data, meta = pyreadstat.read_sav('Dataset Final.sav')
    if data.isnull().sum().sum() > 0:
        data = data.dropna()
    columns_to_drop = ['B1R1', 'weight_final', 'filter_$']  # 'Pulau' column is not dropped here
    data = data.drop(columns=columns_to_drop, errors='ignore')
    X = data.drop(columns=['Y', 'Pulau'])  # 'Pulau' column is excluded from prediction data
    y = data['Y']
    return train_test_split(X, y, test_size=0.2, random_state=42), X, data, meta.column_names_to_labels

(X_train, X_test, y_train, y_test), X, data, column_names_to_labels = load_and_preprocess_data()

# Rename columns in data for display purposes
data_display = data.rename(columns=column_names_to_labels)

# EDA
st.title("Aplikasi Prediksi Obesitas")
st.write("""
Aplikasi ini memprediksi kemungkinan obesitas berdasarkan fitur input pengguna.
Ini menggunakan model XGBoost yang telah dilatih untuk membuat prediksi dan memberikan wawasan
tentang faktor risiko utama yang berkontribusi terhadap obesitas. Aplikasi ini juga mencakup analisis data eksploratif (EDA)
dan visualisasi untuk membantu memahami dataset dengan lebih baik.

Fitur:
- Memprediksi obesitas menggunakan model XGBoost yang telah dilatih
- Menampilkan peta Indonesia dengan diagram lingkaran yang menunjukkan faktor risiko untuk setiap pulau
- Memungkinkan input pengguna untuk prediksi dan menampilkan faktor risiko utama serta pengaruhnya
""")

# Histogram
# st.write("### Histogram dari Fitur")
# fig, ax = plt.subplots(figsize=(10, 6))
# data.hist(ax=ax, bins=20)
# st.pyplot(fig)

# Box Plot
# st.write("### Box Plot dari Fitur")
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.boxplot(data=data, ax=ax)
# st.pyplot(fig)

# # Correlation Heatmap
# st.write("### Heatmap Korelasi")
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
# st.pyplot(fig)

# # Prediction using loaded XGBoost model
# y_pred = xgb_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# st.write(f"### Akurasi Model (XGBoost): {accuracy:.2f}")

# Map of Indonesia
st.write("## Faktor Risiko Obesitas di Setiap Pulau di Indonesia")

# Create a map centered around Indonesia
m = folium.Map(location=[-2.548926, 118.0148634], zoom_start=4)

# Add markers for each island with a popup for statistics
islands = {
    1 : [-0.789275, 100.619385],
    2 : [-7.614529, 110.712246],
    3 : [-1.681487, 113.382354],
    4 : [-1.430421, 121.445617],
    5 : [-4.269928, 138.080353],
    6: [-8.409518, 115.188919],
    7: [-3.23846, 130.14527]
}

def create_pie_chart(data, island):
    risk_factors = data[data['Pulau'] == island].mean().sort_values(ascending=False).head(5)
    fig, ax = plt.subplots()
    ax.pie(risk_factors, labels=risk_factors.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

for island, coords in islands.items():
    pie_chart = create_pie_chart(data, island)
    popup_html = f"""
    <b>{island}</b><br>
    <img src="data:image/png;base64,{pie_chart}" alt="Pie chart">
    """
    folium.Marker(
        location=coords,
        popup=popup_html,
        tooltip=island
    ).add_to(m)

# Display the map
folium_static(m)

# User input
st.write("# Prediksi Obesitas")
st.write("## Input Data")
input_data = {}
cols = st.columns(2)  # Create two columns for better layout
for i, col_name in enumerate(X.columns):
    label = column_names_to_labels.get(col_name, col_name)
    col = cols[i % 2]  # Alternate between columns
    input_data[col_name] = col.number_input(label, min_value=0.0, max_value=100.0, step=0.1, key=f"input_{i}")

# Prediction
if st.button("Prediksi"):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    prediction = xgb_model.predict(input_df)
    if prediction[0] == 1:
        st.write("### Prediksi: Risiko Tinggi Obesitas")
        
        # Display pie chart of risk factors
        risk_factors = input_df.mean().sort_values(ascending=False).head(5)
        risk_factors_labels = [column_names_to_labels.get(factor, factor) for factor in risk_factors.index]
        fig, ax = plt.subplots()
        ax.pie(risk_factors, labels=risk_factors_labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
        
        # Display statistical report
        st.write("#### Laporan Statistik Faktor Risiko Utama:")
        for factor in risk_factors.index:
            label = column_names_to_labels.get(factor, factor)
            st.write(f"**{label}**: {input_df[factor].values[0]}%")
        
    else:
        st.write("### Prediksi: Risiko Rendah Obesitas")
        st.write("#### Alasan Tidak Obesitas:")
        st.write("- Diet seimbang")
        st.write("- Aktivitas fisik teratur")
        st.write("- Pilihan gaya hidup sehat")
        st.write("- Kesehatan metabolik yang baik")