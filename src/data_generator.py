import numpy as np
import pandas as pd
import os

def generate_crop_data(n_samples=1000):
    """
    Generates synthetic data for Crop Recommendation.
    Features: N, P, K, temperature, humidity, ph, rainfall
    Target: label (Rice, Maize, Chickpea, Kidneybeans, Pigeonpeas, Mothbeans, Mungbean, Blackgram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee)
    """
    np.random.seed(42)
    
    crops = ['Rice', 'Maize', 'Chickpea', 'Kidneybeans', 'Pigeonpeas', 'Mothbeans', 'Mungbean', 'Blackgram', 'Lentil', 'Pomegranate', 'Banana', 'Mango', 'Grapes', 'Watermelon', 'Muskmelon', 'Apple', 'Orange', 'Papaya', 'Coconut', 'Cotton', 'Jute', 'Coffee']
    
    data = []
    
    for crop in crops:
        # Create varied distributions for different crops to make them distinguishable
        samples_per_crop = n_samples // len(crops)
        
        if crop in ['Rice', 'Jute', 'Coconut']: # High rainfall, high humidity
            N = np.random.normal(80, 20, samples_per_crop)
            P = np.random.normal(40, 10, samples_per_crop)
            K = np.random.normal(40, 10, samples_per_crop)
            temp = np.random.normal(26, 5, samples_per_crop)
            hum = np.random.normal(80, 10, samples_per_crop)
            ph = np.random.normal(6.5, 1, samples_per_crop)
            rain = np.random.normal(200, 50, samples_per_crop)
        elif crop in ['Cotton', 'Maize']: # Moderate
            N = np.random.normal(100, 20, samples_per_crop)
            P = np.random.normal(50, 10, samples_per_crop)
            K = np.random.normal(20, 5, samples_per_crop)
            temp = np.random.normal(30, 5, samples_per_crop)
            hum = np.random.normal(60, 10, samples_per_crop)
            ph = np.random.normal(7, 1, samples_per_crop)
            rain = np.random.normal(80, 20, samples_per_crop)
        else: # Generic/Dry
            N = np.random.normal(40, 20, samples_per_crop)
            P = np.random.normal(60, 20, samples_per_crop)
            K = np.random.normal(20, 10, samples_per_crop)
            temp = np.random.normal(25, 7, samples_per_crop)
            hum = np.random.normal(50, 15, samples_per_crop)
            ph = np.random.normal(6, 1.5, samples_per_crop)
            rain = np.random.normal(50, 30, samples_per_crop)
            
        crop_data = pd.DataFrame({
            'N': N, 'P': P, 'K': K,
            'temperature': temp, 'humidity': hum, 'ph': ph, 'rainfall': rain,
            'label': crop
        })
        data.append(crop_data)
        
    final_df = pd.concat(data).sample(frac=1).reset_index(drop=True)
    
    # Ensure no negative values
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
