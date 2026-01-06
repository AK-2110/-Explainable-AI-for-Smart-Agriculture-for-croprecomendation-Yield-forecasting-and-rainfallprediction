import numpy as np
import pandas as pd
import os

def generate_crop_data(n_samples=2000): # Increased samples
    """
    Generates synthetic data for Crop Recommendation with distinct clusters.
    """
    np.random.seed(42)
    
    # Specific Clusters to ensure model learns distinctions
    crops_clusters = {
        'Rice':         {'N': (60, 90), 'P': (35, 60), 'K': (35, 45), 'temp': (20, 28), 'hum': (80, 90), 'ph': (6.0, 7.0), 'rain': (180, 300)}, # Wet
        'Maize':        {'N': (60, 100), 'P': (40, 60), 'K': (15, 25), 'temp': (18, 28), 'hum': (50, 70), 'ph': (5.5, 7.0), 'rain': (60, 110)},  # Moderate
        'Chickpea':     {'N': (20, 60), 'P': (55, 80), 'K': (75, 85), 'temp': (17, 22), 'hum': (15, 20), 'ph': (6.0, 8.0), 'rain': (30, 60)},    # Dry/Cool
        'Kidneybeans':  {'N': (10, 40), 'P': (55, 80), 'K': (15, 25), 'temp': (15, 25), 'hum': (20, 30), 'ph': (5.5, 6.0), 'rain': (60, 150)},
        'Pigeonpeas':   {'N': (10, 40), 'P': (55, 80), 'K': (15, 25), 'temp': (25, 35), 'hum': (40, 70), 'ph': (5.0, 7.0), 'rain': (90, 160)},
        'Mothbeans':    {'N': (10, 40), 'P': (35, 60), 'K': (15, 25), 'temp': (24, 32), 'hum': (40, 65), 'ph': (4.0, 8.0), 'rain': (30, 80)},  # Hardy
        'Mungbean':     {'N': (10, 40), 'P': (35, 60), 'K': (15, 25), 'temp': (25, 30), 'hum': (55, 70), 'ph': (6.0, 7.0), 'rain': (40, 70)},
        'Blackgram':    {'N': (20, 60), 'P': (55, 80), 'K': (15, 25), 'temp': (25, 35), 'hum': (60, 70), 'ph': (6.5, 7.5), 'rain': (60, 75)},
        'Lentil':       {'N': (10, 30), 'P': (55, 80), 'K': (15, 25), 'temp': (18, 29), 'hum': (60, 70), 'ph': (6.0, 7.5), 'rain': (35, 55)},
        'Pomegranate':  {'N': (10, 40), 'P': (10, 30), 'K': (35, 45), 'temp': (18, 25), 'hum': (85, 95), 'ph': (5.5, 7.0), 'rain': (100, 120)},
        'Banana':       {'N': (80, 120), 'P': (70, 95), 'K': (45, 55), 'temp': (25, 30), 'hum': (75, 85), 'ph': (5.5, 6.5), 'rain': (90, 120)},
        'Mango':        {'N': (10, 40), 'P': (15, 30), 'K': (25, 35), 'temp': (27, 35), 'hum': (45, 55), 'ph': (5.0, 7.0), 'rain': (85, 100)},
        'Grapes':       {'N': (10, 40), 'P': (120, 145), 'K': (195, 205), 'temp': (10, 40), 'hum': (80, 85), 'ph': (5.5, 6.5), 'rain': (60, 80)}, # High K
        'Watermelon':   {'N': (80, 120), 'P': (10, 30), 'K': (45, 55), 'temp': (24, 28), 'hum': (80, 90), 'ph': (6.0, 7.0), 'rain': (40, 60)},
        'Muskmelon':    {'N': (80, 120), 'P': (10, 30), 'K': (45, 55), 'temp': (27, 30), 'hum': (90, 95), 'ph': (6.0, 6.8), 'rain': (20, 30)},
        'Apple':        {'N': (10, 40), 'P': (120, 145), 'K': (195, 205), 'temp': (21, 24), 'hum': (90, 95), 'ph': (5.5, 6.5), 'rain': (100, 120)}, # Cool
        'Orange':       {'N': (10, 40), 'P': (10, 30), 'K': (10, 15), 'temp': (10, 35), 'hum': (90, 95), 'ph': (6.0, 7.5), 'rain': (100, 120)},
        'Papaya':       {'N': (30, 70), 'P': (45, 70), 'K': (45, 60), 'temp': (23, 44), 'hum': (90, 95), 'ph': (6.5, 7.0), 'rain': (40, 250)},
        'Coconut':      {'N': (10, 40), 'P': (10, 30), 'K': (25, 35), 'temp': (25, 29), 'hum': (90, 95), 'ph': (5.5, 6.5), 'rain': (150, 250)}, # Coastal
        'Cotton':       {'N': (100, 140), 'P': (35, 60), 'K': (15, 25), 'temp': (22, 26), 'hum': (30, 60), 'ph': (6.0, 8.0), 'rain': (60, 90)}, # Dry/Hot
        'Jute':         {'N': (60, 90), 'P': (35, 60), 'K': (35, 45), 'temp': (23, 26), 'hum': (70, 90), 'ph': (6.0, 7.5), 'rain': (150, 200)},
        'Coffee':       {'N': (80, 120), 'P': (15, 30), 'K': (25, 35), 'temp': (23, 27), 'hum': (50, 70), 'ph': (6.0, 7.5), 'rain': (120, 180)}
    }
    
    data = []
    
    for crop, limit in crops_clusters.items():
        n_count = n_samples // len(crops_clusters)
        
        # Determine ranges
        N = np.random.uniform(limit['N'][0], limit['N'][1], n_count)
        P = np.random.uniform(limit['P'][0], limit['P'][1], n_count)
        K = np.random.uniform(limit['K'][0], limit['K'][1], n_count)
        temp = np.random.uniform(limit['temp'][0], limit['temp'][1], n_count)
        hum = np.random.uniform(limit['hum'][0], limit['hum'][1], n_count)
        ph = np.random.uniform(limit['ph'][0], limit['ph'][1], n_count)
        rain = np.random.uniform(limit['rain'][0], limit['rain'][1], n_count)
        
        # Add noise
        N += np.random.normal(0, 5, n_count)
        temp += np.random.normal(0, 1, n_count)
        
        crop_data = pd.DataFrame({
            'N': N, 'P': P, 'K': K, 
            'temperature': temp, 'humidity': hum, 
            'ph': ph, 'rainfall': rain, 
            'label': crop
        })
        data.append(crop_data)
        
    final_df = pd.concat(data).sample(frac=1).reset_index(drop=True)
    
    # Clip negative values
    cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    final_df[cols] = final_df[cols].applymap(lambda x: max(0, x))
    
    return final_df

def generate_yield_data(n_samples=1000):
    """
    Generates synthetic data for Yield Forecasting.
    Features: Year, Average_Rainfall, Pesticides_Tonnes, Avg_Temp, Area
    Target: Yield (hg/ha)
    """
    np.random.seed(42)
    
    years = np.random.randint(2000, 2025, n_samples)
    rain = np.random.normal(1000, 300, n_samples)
    pest = np.random.normal(50, 20, n_samples)
    temp = np.random.normal(20, 5, n_samples)
    area = np.random.normal(5000, 2000, n_samples)
    
    # Synthetic yield formula
    yield_val = (0.5 * rain) + (20 * pest) + (100 * temp) + (0.01 * area) + np.random.normal(0, 500, n_samples)
    yield_val = np.maximum(yield_val, 0)
    
    df = pd.DataFrame({
        'Year': years,
        'average_rain': rain,
        'pesticides_tonnes': pest,
        'avg_temp': temp,
        'area': area,
        'yield_amount': yield_val
    })
    return df

def generate_rainfall_data(n_samples=2000):
    """
    Generates time-series data for Rainfall Prediction.
    Features: Date, Temperature, Humidity, WindSpeed, Pressure
    Target: Rainfall
    """
    np.random.seed(42)
    dates = pd.date_range(start='2015-01-01', periods=n_samples, freq='D')
    
    # Create seasonal patterns
    t = np.arange(n_samples)
    seasonality = 100 * np.sin(2 * np.pi * t / 365) # Yearly cycle
    
    temp = 25 + 10 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 2, n_samples)
    humidity = 60 + 20 * np.cos(2 * np.pi * t / 365) + np.random.normal(0, 5, n_samples)
    wind = 10 + 5 * np.sin(4 * np.pi * t / 365) + np.random.normal(0, 2, n_samples)
    pressure = 1013 + 5 * np.cos(2 * np.pi * t / 365) + np.random.normal(0, 2, n_samples)
    
    # Rainfall depends on seasonality and humidity
    rainfall = np.maximum(0, seasonality + 0.5 * humidity + np.random.normal(0, 20, n_samples))
    # Add some zero rainfall days (dry season simulation)
    rainfall[rainfall < 15] = 0
    
    df = pd.DataFrame({
        'Date': dates,
        'Temperature': temp,
        'Humidity': humidity,
        'WindSpeed': wind,
        'Pressure': pressure,
        'Rainfall': rainfall
    })
    return df

if __name__ == "__main__":
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Generating Crop Recommendation Data...")
    crop_df = generate_crop_data()
    crop_df.to_csv(os.path.join(out_dir, "crop_recommendation.csv"), index=False)
    
    print("Generating Yield Forecast Data...")
    yield_df = generate_yield_data()
    yield_df.to_csv(os.path.join(out_dir, "yield_forecast.csv"), index=False)
    
    print("Generating Rainfall Prediction Data...")
    rain_df = generate_rainfall_data()
    rain_df.to_csv(os.path.join(out_dir, "rainfall_prediction.csv"), index=False)
    
    print("Data generation complete.")
