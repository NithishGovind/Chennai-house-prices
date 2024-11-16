# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load the model
model = joblib.load('Chennai_house_price.pkl')

# Data processing function
def na_remove(data):
    data.replace(9, 0.5, inplace=True)

def data_processing(data):
    data.replace(9, 0.5, inplace=True)
    K = np.log(data['Price'] / data['Area'])
    data['Location'] = K
    house_features = data.drop(['Price'], axis=1)
    
    pipeline = Pipeline([
        ('remove', na_remove(house_features)),
        ('scaler', StandardScaler())
    ])
    
    return pipeline.fit_transform(house_features)

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON input
    test_data = pd.DataFrame(data)
    
    # Process the input and make predictions
    processed_data = data_processing(test_data)
    predictions = np.exp(model.predict(processed_data))  # Transform log scale predictions
    return jsonify({'predicted_prices': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
