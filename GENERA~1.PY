import pandas as pd
import numpy as np
import os

def generate_crop_data(data_dir):
    print("Generating crop_recommendation.csv...")
    crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
             'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
             'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
             'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
    
    # Define centroids for each crop to create distinct clusters
    crop_profiles = {
        'rice': {'N': 80, 'P': 40, 'K': 40, 'temp': 23, 'hum': 82, 'ph': 6.5, 'rain': 230},
        'maize': {'N': 100, 'P': 45, 'K': 20, 'temp': 22, 'hum': 65, 'ph': 6.2, 'rain': 100},
        'chickpea': {'N': 40, 'P': 60, 'K': 80, 'temp': 18, 'hum': 16, 'ph': 7.3, 'rain': 80},
        'kidneybeans': {'N': 20, 'P': 55, 'K': 20, 'temp': 20, 'hum': 21, 'ph': 5.7, 'rain': 100},
        'pigeonpeas': {'N': 20, 'P': 67, 'K': 20, 'temp': 27, 'hum': 48, 'ph': 5.7, 'rain': 150},
        'mothbeans': {'N': 20, 'P': 40, 'K': 20, 'temp': 28, 'hum': 53, 'ph': 6.8, 'rain': 50},
        'mungbean': {'N': 20, 'P': 40, 'K': 20, 'temp': 28, 'hum': 85, 'ph': 6.7, 'rain': 48},
        'blackgram': {'N': 40, 'P': 67, 'K': 20, 'temp': 29, 'hum': 65, 'ph': 7.1, 'rain': 67},
        'lentil': {'N': 20, 'P': 67, 'K': 20, 'temp': 24, 'hum': 64, 'ph': 6.9, 'rain': 45},
        'pomegranate': {'N': 20, 'P': 10, 'K': 40, 'temp': 22, 'hum': 90, 'ph': 6.4, 'rain': 107},
        'banana': {'N': 100, 'P': 80, 'K': 50, 'temp': 27, 'hum': 80, 'ph': 5.9, 'rain': 100},
        'mango': {'N': 20, 'P': 27, 'K': 30, 'temp': 31, 'hum': 50, 'ph': 5.7, 'rain': 95},
        'grapes': {'N': 23, 'P': 130, 'K': 200, 'temp': 23, 'hum': 81, 'ph': 6.0, 'rain': 70},
        'watermelon': {'N': 100, 'P': 100, 'K': 50, 'temp': 25, 'hum': 85, 'ph': 6.5, 'rain': 50},
        'muskmelon': {'N': 100, 'P': 10, 'K': 50, 'temp': 28, 'hum': 92, 'ph': 6.3, 'rain': 24},
        'apple': {'N': 20, 'P': 130, 'K': 200, 'temp': 22, 'hum': 92, 'ph': 5.9, 'rain': 112},
        'orange': {'N': 20, 'P': 10, 'K': 10, 'temp': 25, 'hum': 92, 'ph': 7.1, 'rain': 110},
        'papaya': {'N': 50, 'P': 50, 'K': 50, 'temp': 33, 'hum': 92, 'ph': 6.7, 'rain': 150},
        'coconut': {'N': 20, 'P': 10, 'K': 30, 'temp': 27, 'hum': 95, 'ph': 5.9, 'rain': 175},
        'cotton': {'N': 120, 'P': 40, 'K': 20, 'temp': 23, 'hum': 79, 'ph': 6.9, 'rain': 80},
        'jute': {'N': 80, 'P': 40, 'K': 40, 'temp': 24, 'hum': 79, 'ph': 6.7, 'rain': 175},
        'coffee': {'N': 100, 'P': 20, 'K': 30, 'temp': 26, 'hum': 57, 'ph': 6.8, 'rain': 150}
    }
    
    data = []
    for crop, profile in crop_profiles.items():
        for _ in range(1200): # Increase samples per crop to 1200
            row = {
                'N': max(0, int(profile['N'] + np.random.normal(0, 3))), # Reduced noise (5 -> 3)
                'P': max(0, int(profile['P'] + np.random.normal(0, 3))), # Reduced noise
                'K': max(0, int(profile['K'] + np.random.normal(0, 3))), # Reduced noise
                'temperature': profile['temp'] + np.random.normal(0, 0.5), # Reduced noise (1 -> 0.5)
                'humidity': max(0, min(100, profile['hum'] + np.random.normal(0, 1))), # Reduced noise (2 -> 1)
                'ph': max(0, min(14, profile['ph'] + np.random.normal(0, 0.1))), # Reduced noise (0.2 -> 0.1)
                'rainfall': max(0, profile['rain'] + np.random.normal(0, 5)), # Reduced noise (10 -> 5)
                'label': crop
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(data_dir, "crop_recommendation.csv"), index=False)

def generate_yield_data(data_dir):
    print("Generating yield_forecast.csv...")
    years = range(1990, 2024)
    data = []
    for year in years:
        row = {
            'Year': year,
            'average_rain': np.random.uniform(500, 2500),
            'pesticides_tonnes': np.random.uniform(10, 500),
            'avg_temp': np.random.uniform(15, 35),
            'area': np.random.uniform(100, 5000),
            'yield_amount': np.random.uniform(1000, 50000)
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(data_dir, "yield_forecast.csv"), index=False)

def generate_rainfall_data(data_dir):
    print("Generating rainfall_prediction.csv...")
    data = []
    for _ in range(500):
        row = {
            'Temperature': np.random.uniform(10, 50),
            'Humidity': np.random.uniform(10, 100),
            'WindSpeed': np.random.uniform(0, 30),
            'Pressure': np.random.uniform(980, 1030),
            'Rainfall': np.random.uniform(0, 500)
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(data_dir, "rainfall_prediction.csv"), index=False)

if __name__ == "__main__":
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    generate_crop_data(data_dir)
    generate_yield_data(data_dir)
    generate_rainfall_data(data_dir)
    print("Datasets created successfully in 'data/' directory.")
