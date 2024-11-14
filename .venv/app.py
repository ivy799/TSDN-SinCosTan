from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
import pyreadstat
from joblib import load
import matplotlib.pyplot as plt
import folium
import base64
from io import BytesIO

app = Flask(__name__)

# Load model
def load_model():
    return load('xgb_model.joblib')

xgb_model = load_model()

# Load and preprocess data
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
data_display = data.rename(columns=column_names_to_labels)

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

@app.route('/')
def home():
    # Create map
    m = folium.Map(location=[-2.548926, 118.0148634], zoom_start=5)
    
    islands = {
        1: [-0.789275, 100.619385],
        2: [-7.614529, 110.712246],
        3: [-1.681487, 113.382354],
        4: [-1.430421, 121.445617],
        5: [-4.269928, 138.080353]
    }

    for island, coords in islands.items():
        pie_chart = create_pie_chart(data, island)
        popup_html = f"""
        <b>{island}</b><br>
        <img src="data:image/png;base64,{pie_chart}" alt="Pie chart">
        """
        folium.Marker(
            location=coords,
            popup=popup_html,
            tooltip=str(island)
        ).add_to(m)

    return render_template('index.html',
                         table_head=data_display.head().to_html(),
                         table_describe=data_display.describe().to_html(),
                         map=m._repr_html_(),
                         features=X.columns,
                         labels=column_names_to_labels)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {}
    for col in X.columns:
        input_data[col] = float(request.form.get(col, 0))
    
    input_df = pd.DataFrame([input_data], columns=X.columns)
    prediction = xgb_model.predict(input_df)
    result = 'Obese' if prediction[0] == 1 else 'Not Obese'
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)