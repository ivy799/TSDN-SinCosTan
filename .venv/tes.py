import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
import pyreadstat
import matplotlib.pyplot as plt

# Cache the model loading
@st.cache_resource
def load_model():
    return load('xgb_model.joblib')

xgb_model = load_model()

# Cache the dataset loading and preprocessing
@st.cache_data
def load_and_preprocess_data():
    data, meta = pyreadstat.read_sav('Dataset Final.sav')
    if data.isnull().sum().sum() > 0:
        data = data.dropna()
    columns_to_drop = ['B1R1', 'weight_final', 'filter_$']
    data = data.drop(columns=columns_to_drop, errors='ignore')
    X = data.drop(columns=['Y', 'Pulau'])
    y = data['Y']
    return train_test_split(X, y, test_size=0.2, random_state=42), X, data, meta.column_names_to_labels

(X_train, X_test, y_train, y_test), X, data, column_names_to_labels = load_and_preprocess_data()

# Label mappings
label_mappings = {
    'X1': {1.0: 'Perkotaan', 2.0: 'Pedesaan'},
    'X2': {1.0: 'Laki-laki', 2.0: 'Perempuan'},
    'X3': {1.0: 'Belum Kawin', 2.0: 'Kawin', 3.0: 'Cerai hidup', 4.0: 'Cerai mati'},
    'X4': {1.0: '15 - 24 tahun', 2.0: '25 - 34 tahun', 3.0: '35 - 44 tahun', 4.0: '45 - 54 tahun', 
                5.0: '55 - 64 tahun', 6.0: '65 - 74 tahun', 7.0: '> 74 tahun'},
    'X5': {1.0: 'Tidak/ belum pernah sekolah', 2.0: 'Tidak tamat SD/MI', 3.0: 'Tamat SD/MI', 
                4.0: 'Tamat SMP/MTS', 5.0: 'Tamat SMA/MA', 6.0: 'Tamat D1/D2/D3', 7.0: 'Tamat PT'},
    'X6': {1.0: 'Tidak bekerja', 2.0: 'Sekolah', 3.0: 'PNS/ TNI/ Polri/ BUMN/ BUMD', 
                4.0: 'Pegawai swasta', 5.0: 'Wiraswasta', 6.0: 'Petani', 7.0: 'Nelayan', 
                8.0: 'Buruh/ sopir/ pembantu ruta', 9.0: 'Lainnya'},
    'X7': {1.0: 'Ya, rutin', 2.0: 'Ya, kadang-kadang', 3.0: 'Tidak'},
    'X8': {1.0: '>1 kali per hari', 2.0: '1 kali per hari', 3.0: '3-6 kali per minggu', 
                 4.0: '1-2 kali per minggu', 5.0: '< 3 kali per bulan', 6.0: 'Tidak pernah'},
    'X9': {1.0: '>1 kali per hari', 2.0: '1 kali per hari', 3.0: '3-6 kali per minggu', 
                 4.0: '1-2 kali per minggu', 5.0: '< 3 kali per bulan', 6.0: 'Tidak pernah'},
    'X10': {1.0: '>1 kali per hari', 2.0: '1 kali per hari', 3.0: '3-6 kali per minggu', 
                 4.0: '1-2 kali per minggu', 5.0: '< 3 kali per bulan', 6.0: 'Tidak pernah'},
    'X11': {1.0: '>1 kali per hari', 2.0: '1 kali per hari', 3.0: '3-6 kali per minggu', 
                 4.0: '1-2 kali per minggu', 5.0: '< 3 kali per bulan', 6.0: 'Tidak pernah'},
    'X12': {1.0: '>1 kali per hari', 2.0: '1 kali per hari', 3.0: '3-6 kali per minggu', 
                 4.0: '1-2 kali per minggu', 5.0: '< 3 kali per bulan', 6.0: 'Tidak pernah'},
    'X13': {1.0: '>1 kali per hari', 2.0: '1 kali per hari', 3.0: '3-6 kali per minggu', 
                 4.0: '1-2 kali per minggu', 5.0: '< 3 kali per bulan', 6.0: 'Tidak pernah'},
    'X14': {1.0: '>1 kali per hari', 2.0: '1 kali per hari', 3.0: '3-6 kali per minggu', 
                 4.0: '1-2 kali per minggu', 5.0: '< 3 kali per bulan', 6.0: 'Tidak pernah'},
    'X15': {1.0: '>1 kali per hari', 2.0: '1 kali per hari', 3.0: '3-6 kali per minggu', 
                 4.0: '1-2 kali per minggu', 5.0: '< 3 kali per bulan', 6.0: 'Tidak pernah'},
    'X16': {1.0: '>1 kali per hari', 2.0: '1 kali per hari', 3.0: '3-6 kali per minggu', 
                 4.0: '1-2 kali per minggu', 5.0: '< 3 kali per bulan', 6.0: 'Tidak pernah'},
    'X17': {1.0: 'Tidak', 2.0: 'Berhenti Merokok', 3.0: 'Ya'},
    'X18': {1.0: 'Cukup Aktif', 2.0: 'Kurang Aktif'},
    'X19': {1.0: 'Ya', 2.0: 'Tidak'},
    'X20': {1.0: 'Normal', 2.0: 'Prehypertension', 3.0: 'Hypertension stage 1', 
                 4.0: 'Hypertension stage 2'},
}

# Title and description
st.title("Aplikasi Prediksi Obesitas")
st.write("""
Aplikasi ini memprediksi kemungkinan obesitas berdasarkan input pengguna.
""")

# User input section
st.write("## Input Data")
input_data = {}
cols = st.columns(2)

for i, col_name in enumerate([col for col in column_names_to_labels.keys() if col in X.columns]):
    col = cols[i % 2]
    label = column_names_to_labels.get(col_name, col_name)
    options = list(label_mappings[col_name].values())
    selected_label = col.selectbox(
        label,
        options=options,
        key=f"input_{i}"
    )
    numeric_value = [k for k, v in label_mappings[col_name].items() if v == selected_label][0]
    input_data[col_name] = numeric_value

# Prediction section
if st.button("Prediksi"):
    input_df = pd.DataFrame([input_data], columns=X.columns)

    prediction = xgb_model.predict(input_df)
    
    if prediction[0] == 1:
        st.write("### Prediksi: Risiko Tinggi Obesitas")
        
        # Display risk factors
        risk_factors = input_df.mean().sort_values(ascending=False).head(5)
        risk_factors_labels = [column_names_to_labels.get(factor, factor) for factor in risk_factors.index]
        
        fig, ax = plt.subplots()
        ax.pie(risk_factors, labels=risk_factors_labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
        
        st.write("#### Laporan Statistik Faktor Risiko Utama:")
        for factor in risk_factors.index:
            label = column_names_to_labels.get(factor, factor)
            st.write(f"**{label}**: {input_df[factor].values[0]}")
    else:
        st.write("### Prediksi: Risiko Rendah Obesitas")
        st.write("#### Faktor Protektif:")
        st.write("- Diet seimbang")
        st.write("- Aktivitas fisik teratur")
        st.write("- Pilihan gaya hidup sehat")
        st.write("- Kesehatan metabolik yang baik")