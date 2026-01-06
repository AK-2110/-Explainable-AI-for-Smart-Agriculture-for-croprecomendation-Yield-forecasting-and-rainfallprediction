import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from src.feature_extraction import XLNetFeatureExtractor
from src.feature_selection import EBMOFeatureSelection
from src.preprocessing import DataPreprocessor

# Set page config
st.set_page_config(page_title="Smart Agri XAI", layout="wide")

# Paths
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

@st.cache_resource
def load_models():
    """Load all trained models and artifacts."""
    if not os.path.exists(MODELS_DIR):
        return None, None, None, None, None
        
    # Load Scikit-Learn/Joblib models
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    crop_svm = joblib.load(os.path.join(MODELS_DIR, "crop_svm.joblib"))
    ebmo = joblib.load(os.path.join(MODELS_DIR, "ebmo_selector.pkl"))
    
    # Load Keras models
    # Note: Transformer has custom layers, so we might need custom_objects if saved with save_model
    # But usually .keras format or h5 handles standard layers well. 
    # If custom layer issues arise, we'll need to import the class.
    yield_lstm = tf.keras.models.load_model(os.path.join(MODELS_DIR, "yield_lstm.keras"))
    rainfall_transformer = tf.keras.models.load_model(os.path.join(MODELS_DIR, "rainfall_transformer.keras"))
    
    return preprocessor, crop_svm, ebmo, yield_lstm, rainfall_transformer

def main():
    st.title("🌾 Explainable AI for Smart Agriculture")
    st.markdown("""
    This application uses **XLNet, EBMO, and Hybrid Models** to provide:
    1. **Crop Recommendation** (with Explainability)
    2. **Yield Forecasting**
    3. **Rainfall Prediction**
    """)
    
    preprocessor, crop_svm, ebmo, yield_lstm, rainfall_transformer = load_models()
    
    if not preprocessor:
        st.error("Models not found! Please run `python main.py` first to train and save models.")
        return

    # Sidebar Navigation
    app_mode = st.sidebar.selectbox("Choose Application", 
        ["Crop Recommendation", "Yield Forecasting", "Rainfall Prediction"])

    if app_mode == "Crop Recommendation":
        run_crop_recommendation(preprocessor, crop_svm, ebmo)
        
    elif app_mode == "Yield Forecasting":
        run_yield_forecasting(preprocessor, yield_lstm)
        
    elif app_mode == "Rainfall Prediction":
        run_rainfall_prediction(preprocessor, rainfall_transformer)

import random

def get_live_weather(location):
    """
    Simulate fetching live data based on location.
    Uses hashing to ensure the same location always yields the same 'live' data
    (unless the user changes parameters manually).
    """
    loc_lower = location.lower().strip()
    
    # 1. Known Profiles for Realism (Mock Database)
    weather_db = {
        'delhi': {'temp': 32.0, 'hum': 40.0, 'rain': 60.0, 'wind': 15.0, 'press': 1005.0},
        'new delhi': {'temp': 33.0, 'hum': 38.0, 'rain': 55.0, 'wind': 14.0, 'press': 1004.0},
        'mumbai': {'temp': 28.0, 'hum': 85.0, 'rain': 300.0, 'wind': 20.0, 'press': 1008.0},
        'pune': {'temp': 27.0, 'hum': 60.0, 'rain': 100.0, 'wind': 12.0, 'press': 1010.0},
        'nagpur': {'temp': 35.0, 'hum': 45.0, 'rain': 110.0, 'wind': 10.0, 'press': 1002.0},
        'chennai': {'temp': 30.0, 'hum': 80.0, 'rain': 200.0, 'wind': 25.0, 'press': 1006.0},
        'kolkata': {'temp': 29.0, 'hum': 82.0, 'rain': 250.0, 'wind': 18.0, 'press': 1000.0},
        'bangalore': {'temp': 24.0, 'hum': 65.0, 'rain': 120.0, 'wind': 15.0, 'press': 1012.0},
        'hyderabad': {'temp': 30.0, 'hum': 55.0, 'rain': 80.0, 'wind': 12.0, 'press': 1009.0},
        'jaipur': {'temp': 36.0, 'hum': 30.0, 'rain': 40.0, 'wind': 12.0, 'press': 1003.0},
    }
    
    if loc_lower in weather_db:
        base = weather_db[loc_lower]
        # Add tiny jitter so it feels "live" but stays consistent to region
        random.seed(42) # Fixed jitter for demo stability
        return {
            'temp': base['temp'] + round(random.uniform(-1, 1), 1),
            'hum': base['hum'] + round(random.uniform(-2, 2), 1),
            'rain': base['rain'] + round(random.uniform(-5, 5), 1),
            'wind': base['wind'] + round(random.uniform(-1, 1), 1),
            'press': base['press'] + round(random.uniform(-1, 1), 1)
        }
    
    # 2. Unknown Location: Deterministic Random based on Name
    # This ensures "Srinagar" always gives Cold, "Jaisalmer" gives Hot (if lucky with hash)
    # But mainly it ensures consistency.
    else:
        # Create a seed from the city name
        seed_val = sum(ord(c) for c in loc_lower)
        random.seed(seed_val)
        
        return {
            'temp': round(random.uniform(20.0, 35.0), 1),
            'hum': round(random.uniform(40.0, 90.0), 1),
            'rain': round(random.uniform(50.0, 250.0), 1),
            'wind': round(random.uniform(5.0, 25.0), 1),
            'press': round(random.uniform(995.0, 1020.0), 1)
        }

def display_preventive_measures(rain, temp, hum):
    """Displays risk warnings and preventive measures based on climatic conditions."""
    st.markdown("### 🛡️ Climate Risk & Preventive Measures")
    
    risks_found = False
    
    # 1. Drought / Low Water
    if rain < 300: # Threshold for concern
        st.warning("⚠️ **Risk: Low Rainfall / Drought Conditions**")
        st.markdown("""
        *   **Irrigation**: Adopt drip or sprinkler irrigation to maximize water efficiency.
        *   **Mulching**: Cover soil with organic mulch to reduce evaporation.
        *   **Variety Selection**: Choose drought-resistant crop varieties.
        """)
        risks_found = True
        
    # 2. Flood / Excess Water
    elif rain > 2000:
        st.warning("⚠️ **Risk: Heavy Rainfall / Flood Prone**")
        st.markdown("""
        *   **Drainage**: Ensure proper field drainage channels to prevent waterlogging.
        *   **Raised Beds**: Plant crops on raised beds to keep roots aerated.
        *   **Harvesting**: If crop is mature, harvest immediately to prevent spoilage.
        """)
        risks_found = True

    # 3. Heat Stress
    if temp > 35:
        st.warning("⚠️ **Risk: High Temperature / Heat Stress**")
        st.markdown("""
        *   **Irrigation**: Frequent light irrigation during cooler hours (evening/early morning).
        *   **Shade**: Use shade nets or intercropping with taller plants to protect sensitive crops.
        """)
        risks_found = True
        
    # 4. Cold Stress
    elif temp < 10:
        st.warning("⚠️ **Risk: Low Temperature / Frost Damage**")
        st.markdown("""
        *   **Protective Cover**: Use row covers or tunnels to retain heat.
        *   **Irrigation**: Irrigate late in the evening; moist soil holds heat better than dry soil.
        *   **Smoke**: Controlled smoking around the field can prevent frost formation.
        """)
        risks_found = True

    # 5. Disease Risk (High Humidity)
    if hum > 85:
        st.warning("⚠️ **Risk: High Humidity (Fungal Disease Risk)**")
        st.markdown("""
        *   **Spacing**: Ensure wider plant spacing for air circulation.
        *   **Monitoring**: Check daily for fungal spots or mold. Apply fungicides preventively if needed.
        """)
        risks_found = True
        
    if not risks_found:
        st.success("✅ **Climate Conditions are Favorable.** No extreme warnings detected.")

def run_crop_recommendation(preprocessor, model, ebmo):
    st.header("🌱 Crop Recommendation System")
    
    # Session State
    if 'c_temp' not in st.session_state: st.session_state['c_temp'] = 25.0
    if 'c_hum' not in st.session_state: st.session_state['c_hum'] = 80.0
    if 'c_rain' not in st.session_state: st.session_state['c_rain'] = 200.0
    
    c1, c2 = st.columns([3, 1])
    with c1:
        loc = st.text_input("📍 Farm Location", "Nagpur, India")
    with c2:
        st.write("")
        st.write("")
        if st.button("🔄 Auto-Detect"):
            w = get_live_weather(loc)
            st.session_state['c_temp'] = w['temp']
            st.session_state['c_hum'] = w['hum']
            st.session_state['c_rain'] = w['rain']
            st.toast("Weather Updated!")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Soil Parameters")
        n = st.number_input("Nitrogen (N)", 0, 140, 90)
        p = st.number_input("Phosphorous (P)", 0, 145, 40)
        k = st.number_input("Potassium (K)", 0, 205, 40)
        ph = st.number_input("pH Level", 0.0, 14.0, 6.5)
        
    with col2:
        st.subheader("Weather Parameters")
        temp = st.number_input("Temperature (°C)", 0.0, 50.0, key='c_temp')
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, key='c_hum')
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, key='c_rain')
        
    # User Request: Use previous year crop data
    st.subheader("Farming History")
    crop_options = ['None', 'Rice', 'Maize', 'Chickpea', 'Kidneybeans', 'Pigeonpeas', 'Mothbeans', 'Mungbean', 'Blackgram', 'Lentil', 'Pomegranate', 'Banana', 'Mango', 'Grapes', 'Watermelon', 'Muskmelon', 'Apple', 'Orange', 'Papaya', 'Coconut', 'Cotton', 'Jute', 'Coffee']
    prev_crop = st.selectbox("Previous Year Crop", crop_options)
        
    if st.button("Recommend Crop"):
        with st.spinner("Analyzing Soil, Weather & History..."):
            # Prepare Input
            input_data = pd.DataFrame({
                'N': [n], 'P': [p], 'K': [k],
                'temperature': [temp], 'humidity': [humidity], 
                'ph': [ph], 'rainfall': [rainfall],
                'label': ['dummy'] # Placeholder
            })
            
            # CRITICAL FIX: Scale the input using the same scaler as training
            numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            if 'tabular_num' in preprocessor.scalers:
                scaler = preprocessor.scalers['tabular_num']
                # Transform only the numeric columns
                input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
            else:
                st.warning("⚠️ Scaler not found: Predictions might be inaccurate (using raw values).")

            # 1. Feature Extraction (XLNet)
            extractor = XLNetFeatureExtractor()
            texts = extractor.tabular_to_text(input_data, numeric_cols)
            features = extractor.extract_features(texts) # Shape (1, 768)
            
            # 2. Feature Selection (EBMO)
            # ebmo is the object. transform() should work if it stores the mask.
            features_selected = ebmo.transform(features)
            
            # 3. Prediction & Profit Analysis
            # Get Probabilities to find top suitable candidates
            if hasattr(model, 'predict_proba') or hasattr(model, 'model'): # Check if wrapped or raw
                # The CropClassifier class wraps the model in .model
                # But here 'model' might be the raw SVC object loaded via joblib
                # Let's check how main.py saved it. 
                # main.py: joblib.dump(svm_model.model, ...) -> So it is the raw SVC
                probs = model.predict_proba(features_selected)[0]
                
                # Get Top 3 suitable crops
                top_3_idx = np.argsort(probs)[-3:][::-1]
                top_3_crops = []
                if 'label' in preprocessor.encoders:
                    top_3_crops = preprocessor.encoders['label'].inverse_transform(top_3_idx)
                
                # Simulated Profit Data (INR per Hectare - Hypothetical)
                CROP_PROFIT = {
                    'Rice': 60000, 'Maize': 45000, 'Chickpea': 40000, 'Kidneybeans': 42000,
                    'Pigeonpeas': 38000, 'Mothbeans': 35000, 'Mungbean': 36000, 'Blackgram': 37000,
                    'Lentil': 39000, 'Pomegranate': 250000, 'Banana': 150000, 'Mango': 200000,
                    'Grapes': 300000, 'Watermelon': 120000, 'Muskmelon': 130000, 'Apple': 400000,
                    'Orange': 180000, 'Papaya': 160000, 'Coconut': 140000, 'Cotton': 90000,
                    'Jute': 50000, 'Coffee': 220000
                }
                
                # Filter Top Suitable Crops (Prob > 10% to be considered "Safe")
                suitable_crops = [crop for i, crop in enumerate(top_3_crops) if probs[top_3_idx[i]] > 0.1]
                if not suitable_crops: suitable_crops = [top_3_crops[0]] # Fallback
                
                # Find the most profitable among suitable
                best_profit_crop = max(suitable_crops, key=lambda c: CROP_PROFIT.get(c, 0))
                best_agronomic_crop = top_3_crops[0]
                
                st.write("---")
                st.markdown("### 💰 Profitability & Selection")
                
                cA, cB = st.columns(2)
                
                with cA:
                    st.success(f"🌱 **Agronomic Best Choice**: **{best_agronomic_crop}**")
                    st.caption(f"Best match for soil/weather (Confidence: {probs[top_3_idx[0]]:.2%})")
                    
                with cB:
                    st.info(f"💵 **Economic Best Choice**: **{best_profit_crop}**")
                    profit = CROP_PROFIT.get(best_profit_crop, 0)
                    st.caption(f"Highest profit among suitable crops (~₹{profit}/ha)")
                
                if best_profit_crop != best_agronomic_crop:
                    st.warning(f"**Insight**: While **{best_agronomic_crop}** grows best, **{best_profit_crop}** is also suitable and offers higher profit. **We recommend {best_profit_crop} for maximum returns.**")
                    crop_name = best_profit_crop
                else:
                    st.success(f"**Insight**: **{best_agronomic_crop}** is both the best grower AND the best profit option!")
                    crop_name = best_agronomic_crop
                    
            else:
                # Fallback if no probability
                pred_idx = model.predict(features_selected)[0]
                crop_name = preprocessor.encoders['label'].inverse_transform([pred_idx])[0]
                st.success(f"Recommended Crop: **{crop_name}**")

            # --- Profit Comparison with Previous Year ---
            if prev_crop != 'None':
                prev_profit = CROP_PROFIT.get(prev_crop, 0)
                curr_profit = CROP_PROFIT.get(crop_name, 0)
                diff = curr_profit - prev_profit
                
                st.markdown("#### 📊 Profit Trend Analysis")
                st.write(f"Previous Year ({prev_crop}): **₹{prev_profit}** | This Year ({crop_name}): **₹{curr_profit}**")
                
                if diff > 0:
                    st.success(f"📈 **Profit Increase**: Expected +₹{diff} per hectare compared to last year.")
                elif diff < 0:
                    st.error(f"📉 **Profit Warning**: This crop yields -₹{abs(diff)} less than last year (but might be necessary for soil health).")
            
            # --- Crop Rotation Logic (Updated) ---
            st.markdown("### 🔄 Crop Rotation Analysis")
            if prev_crop == crop_name:
                st.warning(f"⚠️ **Monoculture Risk**: You grew **{prev_crop}** last year. Planting it again depletes specific soil nutrients and increases disease risk.")
            elif prev_crop in ['Chickpea', 'Kidneybeans', 'Pigeonpeas', 'Mothbeans', 'Mungbean', 'Blackgram', 'Lentil'] and crop_name in ['Rice', 'Maize', 'Cotton']:
                st.success(f"✅ **Excellent Rotation**: Previous crop **{prev_crop}** is a legume which fixed Nitrogen in the soil. This benefits **{crop_name}**.")
            elif prev_crop != 'None':
                st.info(f"ℹ️ **Rotation Check**: Rotating from **{prev_crop}** to **{crop_name}** is generally acceptable.")
            
            # 4. Explainability (Dynamic & Static)
            st.subheader("Explainability (XAI)")
            
            # A. Rule-Based / Feature Contribution (Dynamic)
            st.markdown("#### 🧠 Why this prediction?")
            st.write(f"Analyzing key factors for **{crop_name}**:")
            
            # Simple profile rules (derived from data_generator.py logic)
            # Rice/Jute/Coconut: High Rain, High Hum
            # Cotton/Maize: Moderate
            # Others: Dry
            
            reasons = []
            if crop_name in ['Rice', 'Jute', 'Coconut']:
                if rainfall > 150: reasons.append(f"✅ **High Rainfall ({rainfall}mm)** matches requirement (>150mm).")
                else: reasons.append(f"⚠️ **Rainfall ({rainfall}mm)** is lower than ideal, but other factors compensate.")
                if humidity > 70: reasons.append(f"✅ **High Humidity ({humidity}%)** is optimal.")
            elif crop_name in ['Cotton', 'Maize']:
                 if 25 < temp < 35: reasons.append(f"✅ **Temperature ({temp}°C)** is perfect for growing phase.")
                 if 50 < rainfall < 120: reasons.append(f"✅ **Rainfall ({rainfall}mm)** is within moderate range.")
            else: # Dry crops
                 if rainfall < 100: reasons.append(f"✅ **Low Rainfall ({rainfall}mm)** suits this dry-land crop.")
                 if temp > 20: reasons.append(f"✅ **Warm Climate ({temp}°C)** is beneficial.")
            
            # NPK checks
            if n > 100: reasons.append(f"✅ High Nitrogen ({n}) availability.")
            
            if reasons:
                for r in reasons: st.write(r)
            else:
                st.write("✅ Complex combination of soil nutrients and weather patterns favors this crop.")

            # B. Dynamic Visualization (User vs Ideal)
            st.info("Visualizing your farm conditions against crop requirements:")
            
            # Define Ideal Profiles (Centroids from data_generator.py)
            ideal_profiles = {
                'Rice': {'N': 80, 'P': 40, 'K': 40, 'Temp': 26, 'Hum': 80, 'Rain': 200},
                'Jute': {'N': 80, 'P': 40, 'K': 40, 'Temp': 26, 'Hum': 80, 'Rain': 200},
                'Coconut': {'N': 80, 'P': 40, 'K': 40, 'Temp': 26, 'Hum': 80, 'Rain': 200},
                
                'Cotton': {'N': 100, 'P': 50, 'K': 20, 'Temp': 30, 'Hum': 60, 'Rain': 80},
                'Maize': {'N': 100, 'P': 50, 'K': 20, 'Temp': 30, 'Hum': 60, 'Rain': 80},
                
                'Default': {'N': 40, 'P': 60, 'K': 20, 'Temp': 25, 'Hum': 50, 'Rain': 50}
            }
            
            profile = ideal_profiles.get(crop_name, ideal_profiles['Default'])
            
            # Prepare Data for Plotting
            features = ['N', 'P', 'K', 'Temp', 'Hum', 'Rain']
            user_values = [n, p, k, temp, humidity, rainfall]
            ideal_values = [profile['N'], profile['P'], profile['K'], profile['Temp'], profile['Hum'], profile['Rain']]
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(features))
            width = 0.35
            
            rects1 = ax.bar(x - width/2, user_values, width, label='Your Farm', color='#1f77b4')
            rects2 = ax.bar(x + width/2, ideal_values, width, label=f'Ideal for {crop_name}', color='#2ca02c')
            
            ax.set_ylabel('Value')
            ax.set_title(f'Feature Comparison: Your Conditions vs {crop_name} Needs')
            ax.set_xticks(x)
            ax.set_xticklabels(features)
            ax.legend()
            
            # Normalize Y axis for better view if rain is huge
            # (Optional: Log scale if rain dominates, but linear is easier to understand)
            
            st.pyplot(fig)
            plt.close(fig)
            
            # C. Preventive Measures
            st.write("---")
            display_preventive_measures(rainfall, temp, humidity)

def run_yield_forecasting(preprocessor, model):
    st.header("📈 Yield Forecasting")
    st.write("Predict future yield based on environmental factors.")
    
    # Yield Forecasting Auto-Detect
    if 'y_rain' not in st.session_state: st.session_state['y_rain'] = 1200.0
    if 'y_temp' not in st.session_state: st.session_state['y_temp'] = 28.0
    
    c1, c2 = st.columns([3, 1])
    with c1: loc = st.text_input("📍 Region", "Punjab, India")
    with c2:
        st.write("")
        st.write("")
        if st.button("🔄 Get Weather"):
            w = get_live_weather(loc)
            st.session_state['y_rain'] = w['rain'] * 10 
            st.session_state['y_temp'] = w['temp']
            st.toast("Fetch Complete!")

    rain = st.number_input("Average Rainfall (mm/year)", 0.0, 3000.0, key='y_rain')
    pest = st.number_input("Pesticides (tonnes)", 0.0, 1000.0, 50.0)
    temp = st.number_input("Average Temp (°C)", 0.0, 50.0, key='y_temp')
    area_acres = st.number_input("Area (Acres)", 0.0, 100000.0, 100.0)
    
    if st.button("Forecast Yield"):
        # Sequence creation: repeat input x 3 timestamps
        # Shape must be (1, 3, 5) because model was trained on 5 features (including target)
        scaler = preprocessor.scalers.get('yield_amount')
        
        # Convert Acres to Hectares for Model Compatibility (1 Acre = 0.404686 Hectare)
        area_ha = area_acres * 0.404686
        
        if scaler:
            # Input order: ['average_rain', 'pesticides_tonnes', 'avg_temp', 'area', 'yield_amount']
            # We assume the user's input represents the current state. 
            # For the 'yield_amount' feature (autoregressive), we can use a "Previous Yield" if available,
            # or use the current estimate/dummy. 
            # Better UI: Ask user for "Previous Yield" or just use 0 (or mean).
            # For this fix, let's assume 0 as an initial seed or provide an input.
            
            # Using 0 might bias it if data was not centered at 0. 
            # But the scaler handles normalization.
            prev_yield = st.number_input("Prior Year Yield (HG/HA)", 0.0, 100000.0, 20000.0)
            
            raw_input = np.array([[rain, pest, temp, area_ha, prev_yield]])
            scaled_input = scaler.transform(raw_input)
            
            # Keep ALL 5 features
            scaled_features = scaled_input[0] 
            
            # Replicate 3 times
            input_seq = np.tile(scaled_features, (3, 1)).reshape(1, 3, 5)
            
            # Predict
            pred_scaled = model.predict(input_seq)
            
            # Inverse Transform
            idx = 4 # Target index
            min_val = scaler.data_min_[idx]
            max_val = scaler.data_max_[idx]
            
            pred_inv = pred_scaled[0][0] * (max_val - min_val) + min_val
            
            st.metric("Predicted Yield (hg/ha)", f"{pred_inv:.2f}")
        else:
            st.error("Yield Scaler not found!")

def run_rainfall_prediction(preprocessor, model):
    st.header("⛈️ Rainfall Prediction")
    st.write("Predict rainfall based on atmospheric conditions.")
    
    # Rainfall Auto-Detect
    if 'r_temp' not in st.session_state: st.session_state['r_temp'] = 30.0
    if 'r_hum' not in st.session_state: st.session_state['r_hum'] = 75.0
    if 'r_wind' not in st.session_state: st.session_state['r_wind'] = 10.0
    if 'r_press' not in st.session_state: st.session_state['r_press'] = 1010.0
    if 'r_prev_rain' not in st.session_state: st.session_state['r_prev_rain'] = 0.0
    
    c1, c2 = st.columns([3, 1])
    with c1: loc = st.text_input("Geographic Location (City/Region)", "New Delhi")
    with c2:
        st.write("")
        st.write("")
        if st.button("🔄 Sense Atmosphere"):
            w = get_live_weather(loc)
            st.session_state['r_temp'] = w['temp']
            st.session_state['r_hum'] = w['hum']
            st.session_state['r_wind'] = w['wind']
            st.session_state['r_press'] = w['press']
            # Simulate previous rainfall (e.g., slightly different from current random 'rain' or same)
            st.session_state['r_prev_rain'] = w['rain'] 
            st.toast("Updated Atmosphere Model!")
    
    col1, col2 = st.columns(2)
    with col1:
        temp = st.number_input("Temperature (°C)", 0.0, 50.0, key='r_temp')
        hum = st.number_input("Humidity (%)", 0.0, 100.0, key='r_hum')
    with col2:
        wind = st.number_input("Wind Speed (km/h)", 0.0, 100.0, key='r_wind')
        pressure = st.number_input("Pressure (hPa)", 900.0, 1100.0, key='r_press')
        
    prev_rain = st.number_input("Previous Rainfall (mm)", 0.0, 500.0, key='r_prev_rain')
    
    if st.button("Predict Rainfall"):
        scaler = preprocessor.scalers.get('Rainfall')
        if scaler:
            # Columns: ['Temperature', 'Humidity', 'WindSpeed', 'Pressure', 'Rainfall']
            raw_input = np.array([[temp, hum, wind, pressure, prev_rain]])
            scaled_input = scaler.transform(raw_input)
            
            # Keep ALL 5 features
            scaled_features = scaled_input[0]
            
            # Reshape to (1, 10, 5)
            input_seq = np.tile(scaled_features, (10, 1)).reshape(1, 10, 5)
            
            pred_scaled = model.predict(input_seq)
            
            # Inverse transform target (last col, idx 4)
            idx = 4
            min_val = scaler.data_min_[idx]
            max_val = scaler.data_max_[idx]
            
            pred_inv = pred_scaled[0][0] * (max_val - min_val) + min_val
            
            st.metric(f"Expected Rainfall for {loc} (mm)", f"{pred_inv:.2f}")
        else:
            st.error("Rainfall Scaler not found!")

if __name__ == "__main__":
    main()
