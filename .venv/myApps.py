import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pyreadstat
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import base64
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_PALETTE = {
    'primary': '#6C63FF',      
    'secondary': '#2EC4B6',    
    'accent': '#FF6B6B',       
    'background': '#1A1A2E',   
    'text': '#000000',         
    'gradient': ['#6C63FF', '#2EC4B6']  
}


pages = ["Home", "Analisis Data", "Prediksi", "Peta Risiko"]
page = "Home"  

st.markdown(
    f"""
<style>
    .main {{
        background-color: {COLOR_PALETTE['background']};
        font-family: 'Helvetica Neue', sans-serif;
        color: {COLOR_PALETTE['text']};
    }}
    .stButton>button {{
        background-color: {COLOR_PALETTE['primary']};
        color: {COLOR_PALETTE['text']};
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s;
    }}
    .stButton>button:hover {{
        background-color: {COLOR_PALETTE['secondary']};
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(108, 99, 255, 0.2);
    }}
    .chart-container {{
        background: {COLOR_PALETTE['gradient']};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    .metric-container {{
        background: linear-gradient(135deg, {COLOR_PALETTE['gradient'][0]} 0%, {COLOR_PALETTE['gradient'][1]} 100%);
        color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.2);
    }}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <style>
        /* Base styling */
        .main {{
            background-color: {COLOR_PALETTE['background']};
            font-family: 'Helvetica Neue', sans-serif;
            color: {COLOR_PALETTE['text']};
            padding: 1rem 2rem;
        }}
        
        /* Navigation styling */
        .nav-container {{
            background: {COLOR_PALETTE['gradient']};
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }}
        
        .nav-title {{
            color: {COLOR_PALETTE['text']};
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid {COLOR_PALETTE['primary']};
        }}
        
        .section-container {{
            padding: 2rem;
            margin: 1.5rem 0;
        }}
        
        .content-padding {{
            padding: 0 1rem;
        }}
        
        .stButton>button {{
            background: linear-gradient(135deg, {COLOR_PALETTE['primary']} 0%, {COLOR_PALETTE['secondary']} 100%);
            color: {COLOR_PALETTE['text']};
            padding: 0.8rem 2rem;
            border-radius: 8px;
            border: none;
            transition: all 0.3s;
            margin: 1rem 0;
            width: 100%;
        }}
        
        .chart-container {{
            background: {COLOR_PALETTE['gradient']};
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1.5rem 0;
        }}
        
        .metric-container {{
            padding: 1.5rem;
            margin: 1.5rem 0;
        }}

        .sidebar .sidebar-content {{
            background: {COLOR_PALETTE['background']};
            padding: 2rem 1rem;
        }}
        
        .nav-section {{
            margin-bottom: 2rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

label_mappings = {
    'X1': {0.0: 'Perkotaan', 1.0: 'Pedesaan'},
    'X2': {0.0: 'Laki-laki', 1.0: 'Perempuan'},
    'X3': {1.0: 'Belum Kawin', 2.0: 'Kawin', 3.0: 'Cerai hidup', 4.0: 'Cerai mati'},
    'X4': {1.0: '15 - 24 tahun', 2.0: '25 - 34 tahun', 3.0: '35 - 44 tahun', 4.0: '45 - 54 tahun', 
           5.0: '55 - 64 tahun', 6.0: '65 - 74 tahun', 7.0: '> 74 tahun'},
    'X5': {1.0: 'Tidak/ belum pernah sekolah', 2.0: 'Tidak tamat SD/MI', 3.0: 'Tamat SD/MI', 
           4.0: 'Tamat SMP/MTS', 5.0: 'Tamat SMA/MA', 6.0: 'Tamat D1/D2/D3', 7.0: 'Tamat PT'},
    'X6': {1.0: 'Tidak bekerja', 2.0: 'Sekolah', 3.0: 'PNS/ TNI/ Polri/ BUMN/ BUMD', 
           4.0: 'Pegawai swasta', 5.0: 'Wiraswasta', 6.0: 'Petani', 7.0: 'Nelayan', 
           8.0: 'Buruh/ sopir/ pembantu ruta', 9.0: 'Lainnya'},
    'X7': {0.0: 'Ya, rutin', 1.0: 'Ya, kadang-kadang', 2.0: 'Tidak'},
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
    'X16': {0.0: '>1 kali per hari', 1.0: '1 kali per hari', 2.0: '3-6 kali per minggu', 
            4.0: '1-2 kali per minggu', 5.0: '< 3 kali per bulan', 6.0: 'Tidak pernah'},
    'X17': {0.0: 'Tidak', 1.0: 'Berhenti Merokok', 2.0: 'Ya'},
    'X18': {0.0: 'Cukup Aktif', 1.0: 'Kurang Aktif'},
    'X19': {1.0: 'Ya', 2.0: 'Tidak'},
    'X20': {0.0: 'Normal', 1.0: 'Prehypertension', 2.0: 'Hypertension stage 1', 
            3.0: 'Hypertension stage 2'},
}

@st.cache_resource
def load_model():
    try:
        return load("xgboost_optimized_model.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


xgb_model = load_model()

@st.cache_data
def load_and_preprocess_data():
    try:
        data, meta = pyreadstat.read_sav("Dataset Final.sav")
        if data.isnull().sum().sum() > 0:
            data = data.dropna()
        columns_to_drop = [
            "B1R1",
            "weight_final",
            "filter_$",
        ]  
        data = data.drop(columns=columns_to_drop, errors="ignore")
        X = data.drop(
            columns=["Y", "Pulau"]
        ) 
        y = data["Y"]
        return (
            train_test_split(X, y, test_size=0.2, random_state=42),
            X,
            data,
            meta.column_names_to_labels,
        )
    except Exception as e:
        st.error(f"Error loading and preprocessing data: {e}")
        return None, None, None, None


(X_train, X_test, y_train, y_test), X, data, column_names_to_labels = (
    load_and_preprocess_data()
)

if X_train is None or xgb_model is None:
    st.error("Error: Data atau model tidak dapat dimuat.")
    st.stop()

if 'data' not in locals() or data is None:
    st.error("Error: Dataset tidak dapat dimuat.")
    st.stop()

data_display = data.rename(columns=column_names_to_labels)

st.sidebar.markdown(
    f"""
    <div class="nav-container">
        <div class="nav-title">Navigasi</div>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigasi",  
    pages,
    label_visibility="collapsed"  
)

st.sidebar.markdown(
    f"""
    <div class="nav-container" style="margin-top: 2rem;">
        <div class="nav-title">Tentang Aplikasi</div>
        <p style="color: {COLOR_PALETTE['text']}; font-size: 0.9em; margin-top: 1rem;">
            Versi 1.0.0<br>
            Dikembangkan dengan Streamlit
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="content-padding">', unsafe_allow_html=True)

if page == "Home":
    st.markdown(
    f"""
    <div class="metric-container">
        <h1 style="text-color: #FFFFFFF; text-align: center; font-size: 2.2em; margin-bottom: 10px;">Aplikasi Prediksi Obesitas</h1>
        <p style="text-align: center; font-size: 1.2em;">Analisis & Prediksi Risiko Obesitas menggunakan Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
    )

    st.write(
        """
    Aplikasi ini dirancang untuk memprediksi kemungkinan obesitas berdasarkan fitur input pengguna. Dengan memanfaatkan model XGBoost yang telah dilatih, aplikasi ini mampu memberikan prediksi serta mengidentifikasi faktor risiko utama yang berkontribusi terhadap obesitas. Selain itu, aplikasi ini juga menyediakan analisis data eksploratif (EDA) dan visualisasi untuk membantu pengguna memahami dataset secara lebih mendalam.
        """
    )

    st.header("Fitur Aplikasi:")
    st.write(
        """
        - Visualisasi distribusi tiap fitur dalam dataset.
        - Prediksi risiko obesitas berdasarkan input pengguna.
        - Visualisasi faktor risiko berdasarkan pulau di Indonesia
        """
    )

elif page == "Analisis Data":
    st.header("Analisis Data Eksploratif (EDA)")
    
    label_to_column = {column_names_to_labels.get(col, col): col for col in X.columns}
    
    selected_label = st.selectbox("Pilih Fitur untuk Analisis:", list(label_to_column.keys()))
    selected_feature = label_to_column[selected_label]
    
    plot_data = data.copy()
    if selected_feature in label_mappings:
        try:
            value_counts = plot_data[selected_feature].value_counts().sort_index()
            
            categories = []
            for val in value_counts.index:
                try:
                    lookup_val = float(val) if isinstance(val, (int, float)) else val
                    category = label_mappings[selected_feature].get(lookup_val, f"Value {val}")
                    categories.append(category)
                except (ValueError, TypeError):
                    categories.append(f"Value {val}")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=categories,
                y=value_counts.values,
                marker_color=COLOR_PALETTE['primary']
            ))
            
            fig.update_layout(
                title=f"Analisis {selected_label}",
                template="plotly_white",
                showlegend=False,
                font=dict(size=12),
                hoverlabel=dict(bgcolor="white"),
                plot_bgcolor="rgba(0,0,0,0)",
                height=500,
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis_title="Frekuensi",
                xaxis=dict(
                    tickangle=45,
                    title="Kategori"
                ),
                bargap=0.1
            )
            
            st.plotly_chart(fig, use_container_width=True, key="categorical_plot")
            
        except Exception as e:
            st.error(f"Error creating categorical plot: {e}")
            st.write("Detailed error information:", str(e))
            
    else:
        fig = px.histogram(
            plot_data,
            x=selected_feature,
            title=f"Analisis {selected_label}",
            template="plotly_white",
            color_discrete_sequence=[COLOR_PALETTE['primary']],
            marginal="box"
        )
        fig.update_layout(
            showlegend=False,
            font=dict(size=12),
            hoverlabel=dict(bgcolor="white"),
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis_title="Frekuensi",
            bargap=0.1
        )
    
        st.plotly_chart(fig, use_container_width=True, key="histogram_plot")

    st.markdown("Statistik Deskriptif")
    
    col1, col2 = st.columns(2)
    if selected_feature in label_mappings:
        value_counts = plot_data[selected_feature].value_counts()
        most_common = label_mappings[selected_feature].get(value_counts.index[0], value_counts.index[0])
        least_common = label_mappings[selected_feature].get(value_counts.index[-1], value_counts.index[-1])
        with col1:
            st.metric("Kategori Terbanyak", f"{most_common}")
            st.metric("Total Kategori", f"{len(value_counts)}")
        with col2:
            st.metric("Kategori Tersedikit", f"{least_common}")
            st.metric("Total Data", f"{len(plot_data[selected_feature]):,}")
    else:
        with col1:
            st.metric("Rata-rata", f"{data[selected_feature].mean():.2f}")
            st.metric("Median", f"{data[selected_feature].median()::.2f}")
        with col2:
            st.metric("Standar Deviasi", f"{data[selected_feature].std()::.2f}")
            st.metric("Total Data", f"{len(data[selected_feature]):,}")

elif page == "Prediksi":
    st.header("Prediksi Risiko Obesitas")
    st.write("##### Masukkan data yang sesuai untuk mendapatkan prediksi risiko obesitas berdasarkan faktor sosio-demografis dan pola hidup Anda.")
    input_data = {}
    
    st.write("### Sosio-Demografis")
    demo_col1, demo_col2 = st.columns(2)
    demographic_vars = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
    
    for i, col_name in enumerate(demographic_vars):
        col = demo_col1 if i % 2 == 0 else demo_col2
        with col:
            label = column_names_to_labels.get(col_name, col_name)
            options = list(label_mappings[col_name].values())
            selected_label = st.selectbox(
                label,
                options=options,
                key=f"demo_{i}",
                help=f"Pilih {label.lower()} Anda"
            )
            numeric_value = [k for k, v in label_mappings[col_name].items() if v == selected_label][0]
            input_data[col_name] = numeric_value

    st.write("### Pola Hidup")
    life_col1, life_col2 = st.columns(2)
    lifestyle_vars = ['X7', 'X17', 'X18', 'X19']
    
    for i, col_name in enumerate(lifestyle_vars):
        col = life_col1 if i % 2 == 0 else life_col2
        with col:
            label = column_names_to_labels.get(col_name, col_name)
            options = list(label_mappings[col_name].values())
            selected_label = st.selectbox(
                label,
                options=options,
                key=f"life_{i}",
                help=f"Pilih {label.lower()} Anda"
            )
            numeric_value = [k for k, v in label_mappings[col_name].items() if v == selected_label][0]
            input_data[col_name] = numeric_value

    st.write("### Pola Makan")
    diet_cols = st.columns(3)
    dietary_vars = ['X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16']
    
    for i, col_name in enumerate(dietary_vars):
        with diet_cols[i % 3]:
            label = column_names_to_labels.get(col_name, col_name)
            options = list(label_mappings[col_name].values())
            selected_label = st.selectbox(
                label,
                options=options,
                key=f"diet_{i}",
                help=f"Pilih frekuensi konsumsi {label.lower()}"
            )
            numeric_value = [k for k, v in label_mappings[col_name].items() if v == selected_label][0]
            input_data[col_name] = numeric_value

    st.write("### Status Kesehatan")
    col_name = 'X20'
    label = column_names_to_labels.get(col_name, col_name)
    options = list(label_mappings[col_name].values())

    selected_label = st.selectbox(
        label,
        options=options,
        key="health_0",
        help=f"Pilih {label.lower()} Anda"
    )
    numeric_value = [k for k, v in label_mappings[col_name].items() if v == selected_label][0]
    input_data[col_name] = numeric_value

    st.markdown("<br>", unsafe_allow_html=True)

    predict_button = st.button(
        "Prediksi Sekarang",
        help="Klik untuk melihat hasil prediksi",
        key="predict_button"
    )

    if predict_button:
        try:
            input_df = pd.DataFrame([input_data], columns=X.columns)
            prediction = xgb_model.predict(input_df)

            if prediction[0] == 1:
                st.markdown(
                    f"""
                    <div class='metric-container'>
                        <h2 style='text-align: center;'>Prediksi: Risiko Tinggi Obesitas</h2>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

                risk_factors = input_df.mean().sort_values(ascending=False).head(5)
                fig = go.Figure(
                    go.Bar(
                        x=risk_factors.values,
                        y=[
                            column_names_to_labels.get(factor, factor)
                            for factor in risk_factors.index
                        ],
                        orientation="h",
                        marker=dict(
                            color=[
                                "rgba(0, 123, 255, 0.8)",
                                "rgba(0, 123, 255, 0.6)",
                                "rgba(0, 123, 255, 0.4)",
                                "rgba(0, 123, 255, 0.3)",
                                "rgba(0, 123, 255, 0.2)",
                            ]
                        ),
                    )
                )
                fig.update_layout(
                    title="Top 5 Faktor Risiko",
                    template="plotly_white",
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig, use_container_width=True, key="risk_factors_plot")

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
        except Exception as e:
            st.error(f"Error during prediction: {e}")

elif page == "Peta Risiko":
    st.title("Faktor Risiko Obesitas di Setiap Pulau di Indonesia")

    islands = {
        "Sumatera": [-0.789275, 100.619385],
        "Jawa": [-7.614529, 110.712246],
        "Kalimantan": [-1.681487, 113.382354],
        "Sulawesi": [-1.430421, 121.445617],
        "Papua": [-4.269928, 138.080353],
        "Bali_Nusa": [-8.409518, 115.188919],
        "Maluku": [-3.23846, 130.14527],
    }

    from streamlit_folium import folium_static
    island_name_to_int = {
        "Sumatera": 1,
        "Jawa": 2,
        "Kalimantan": 3,
        "Sulawesi": 4,
        "Papua": 5,
        "Bali_Nusa": 6,
        "Maluku": 7,
    }

    m = folium.Map(
        location=[-2.548926, 118.0148634], zoom_start=4, tiles="cartodb positron"
    )

    for island, coords in islands.items():
        island_int = island_name_to_int[island]
        island_data = data[data["Pulau"] == island_int]
        obesity_risk = island_data["Y"].mean() * 100

        popup_content = f"""
        <div style='width: 200px'>
            <h4>{island}</h4>
            <p>Risiko Obesitas: {obesity_risk:.1f}%</p>
            <p>Jumlah Sampel: {len(island_data)}</p>
        </div>
        """

        folium.Marker(
            location=coords,
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=island,
            icon=folium.Icon(
                color="red" if obesity_risk > 50 else "blue", icon="info-sign"
            ),
        ).add_to(m)

    folium.LayerControl().add_to(m)

    try:
        folium_static(m)
    except Exception as e:
        st.error(f"Error displaying map: {e}")

    st.write("## Detail Faktor Risiko per Pulau")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        selected_island = st.selectbox(
            "Pilih Pulau untuk melihat detail:",
            list(islands.keys()),
            key="island_selector",
        )

    if selected_island:
        st.markdown(
            f"""
            <div class='metric-container'>
                <h3 style='text-align: center;'>Analisis Faktor Risiko: {selected_island}</h3>
            </div>
        """,
            unsafe_allow_html=True,
        )

        stat_col, chart_col = st.columns([1, 2])

        with stat_col:
            island_int = island_name_to_int[selected_island]
            island_data = data[data["Pulau"] == island_int]

            metrics_container = st.container()
            with metrics_container:
                st.markdown(
                    """
                    <style>
                        .metric-box {
                            background: white;
                            padding: 15px;
                            border-radius: 8px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            margin: 10px 0;
                        }
                        .metric-label {
                            color: #666;
                            font-size: 0.9em;
                        }
                        .metric-value {
                            color: #007bff;
                            font-size: 1.8em;
                            font-weight: bold;
                        }
                    </style>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Jumlah Sampel</div>
                        <div class="metric-value">{len(island_data):,}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Risiko Obesitas</div>
                        <div class="metric-value">{(island_data['Y'].mean()*100):.1f}%</div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

        with chart_col:
            X_island = island_data.drop(columns=["Y", "Pulau"])
            y_island = island_data["Y"]
            xgb_model.fit(X_island, y_island)
            feature_importances = (
                pd.Series(xgb_model.feature_importances_, index=X_island.columns)
                .sort_values(ascending=False)
                .head(5)
            )

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=feature_importances.values,
                    y=[
                        column_names_to_labels.get(x, x)
                        for x in feature_importances.index
                    ],
                    orientation="h",
                    marker=dict(
                        color=[
                            "rgba(0, 123, 255, 0.8)",
                            "rgba(0, 123, 255, 0.6)",
                            "rgba(0, 123, 255, 0.4)",
                            "rgba(0, 123, 255, 0.3)",
                            "rgba(0, 123, 255, 0.2)",
                        ],
                        line=dict(width=1, color="#333"),
                    ),
                )
            )

            fig.update_layout(
                title=f"Top 5 Faktor Risiko di {selected_island}",
                template="plotly_white",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis_title="Skor Pentingnya Faktor",
                yaxis_title="Faktor",
                plot_bgcolor="rgba(0,0,0,0)",
                hoverlabel=dict(bgcolor="white"),
                font=dict(family="Helvetica Neue, Arial", size=12),
            )

            st.plotly_chart(fig, use_container_width=True, key="island_risk_factors")

            risk_dist = pd.DataFrame(
                {
                    "Status": ["Obesitas", "Tidak Obesitas"],
                    "Persentase": [
                        (island_data["Y"] == 1).mean() * 100,
                        (island_data["Y"] == 0).mean() * 100,
                    ],
                }
            )

            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=risk_dist["Status"],
                        values=risk_dist["Persentase"],
                        hole=0.3,
                        marker=dict(colors=["#007bff", "#28a745"]),
                    )
                ]
            )

            fig_pie.update_layout(
                title="Distribusi Status Obesitas",
                template="plotly_white",
                height=300,
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
            )

            st.plotly_chart(fig_pie, use_container_width=True, key="island_risk_distribution")