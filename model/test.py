# test.py

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('Chennai_house_price.pkl')

# Function to process data for prediction
def na_remove(data):
    data.replace(9, 0.5, inplace=True)

def data_processing(data):
    # Assuming `data` contains all necessary columns except 'Price'
    data.replace(9, 0.5, inplace=True)
    K = np.log(data['Price'] / data['Area'])
    data['Location'] = K
    house_features = data.drop(['Price'], axis=1)
    
    # Create and apply the pipeline
    my_pipeline = Pipeline([
        ('remove', na_remove(house_features)),
        ('scaler', StandardScaler())
    ])
    
    return my_pipeline.fit_transform(house_features)

# Load new data for testing (you may update this part with actual test data)
test_data = pd.DataFrame({
    'Area': [1200, 1500],  # Example test data, replace with actual values
    'Location': [1, 2],
    'No. of Bedrooms': [3, 2],
    'Resale': [1, 1],
    'MaintenanceStaff': [1, 0],
    'Gymnasium': [1, 0],
    'SwimmingPool': [1, 0],
    'LandscapedGardens': [1, 0],
    'JoggingTrack': [1, 0],
    'RainWaterHarvesting': [1, 0],
    'IndoorGames': [1, 0],
    'ShoppingMall': [1, 0],
    'Intercom': [1, 0],
    'SportsFacility': [1, 0],
    'ATM': [1, 0],
    'ClubHouse': [1, 0],
    'School': [1, 0],
    '24X7Security': [1, 0],
    'PowerBackup': [1, 0],
    'CarParking': [1, 0],
    'StaffQuarter': [1, 0],
    'Cafeteria': [1, 0],
    'MultipurposeRoom': [1, 0],
    'Hospital': [1, 0],
    'WashingMachine': [1, 0],
    'Gasconnection': [1, 0],
    'AC': [1, 0],
    'Wifi': [1, 0],
    "Children'splayarea": [1, 0],
    'LiftAvailable': [1, 0],
    'BED': [1, 0],
    'VaastuCompliant': [1, 0],
    'Microwave': [1, 0],
    'GolfCourse': [1, 0],
    'TV': [1, 0],
    'DiningTable': [1, 0],
    'Sofa': [1, 0],
    'Wardrobe': [1, 0],
    'Refrigerator': [1, 0],
    'Price': [1200000, 1500000]  # Example prices for processing
})

# Process the test data
processed_data = data_processing(test_data)

# Make predictions
predicted_prices = model.predict(processed_data)
predicted_prices = np.exp(predicted_prices)  # Convert log scale predictions back to normal scale

# Print the results
print("Predicted Prices:", predicted_prices)
