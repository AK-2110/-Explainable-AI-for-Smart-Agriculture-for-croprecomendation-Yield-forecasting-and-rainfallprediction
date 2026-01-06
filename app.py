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
st.set_page_config(
    page_title="Smart Agri XAI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Main Background with Image and Overlay */
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.2)), 
                          url('https://images.unsplash.com/photo-1625246333195-bfw292634356?q=80&w=1920&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Sidebar - Dark Glassmorphism */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 30, 10, 0.7); /* Dark Emerald Tint */
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Sidebar Text - Force White */
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* Radio Button (Model Selection) Styling */
    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stRadio label {
        color: #ffffff !important;
        font-size: 1.1rem;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #ffffff; 
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    h2, h3 {
        color: #2E8B57; /* Restore green for content headers inside dark/light cards if needed. 
                           But wait, main body text is on background? 
                           No, we will wrap content in cards. */
    }
    
    /* Content Cards (Simulated by styling generic containers or specific blocks if possible) 
       Streamlit doesn't allow direct labeling easily, so we rely on global styles: */
    
    .block-container {
        padding-top: 2rem;
    }

    /* Specific coloring for sidebar text to ensure contrast */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] p {
        color: #2E8B57 !important;
        text-shadow: none;
    }
    
    /* Buttons */
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #2E8B57 0%, #3CB371 100%);
        border-radius: 25px;
        padding: 12px 28px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 15px rgba(46, 139, 87, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(46, 139, 87, 0.4);
    }
    
    /* Input Fields - White backgrounds with shadow */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 1px solid rgba(0,0,0,0.1);
        color: #333;
        font-weight: 500;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #white;
        font-size: 2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    /* Metric Labels */
    [data-testid="stMetricLabel"] {
        color: #f0f0f0;
    }

    /* Input Labels - Make them legible on dark background */
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {
        color: #ffffff !important;
        font-weight: 500;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(0,0,0,0.2);
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255,255,255,0.8);
        border-radius: 5px;
        color: #333;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E8B57;
        color: white;
    }

</style>
""", unsafe_allow_html=True)

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
    yield_lstm = tf.keras.models.load_model(os.path.join(MODELS_DIR, "yield_lstm.keras"))
    rainfall_transformer = tf.keras.models.load_model(os.path.join(MODELS_DIR, "rainfall_transformer.keras"))
    
    return preprocessor, crop_svm, ebmo, yield_lstm, rainfall_transformer

def main():
    # Sidebar Profile
    with st.sidebar:
        # Plant/Crop related icon (Sprout/Growth)
        st.image("https://cdn-icons-png.flaticon.com/512/1892/1892747.png", width=110) 
        st.title("Smart Agri XAI")
        st.markdown("---")
        app_mode = st.radio("Navigate", 
            ["Crop Recommendation", "Yield Forecasting", "Rainfall Prediction"],
            captions=["Get best crop & preventive advice", "Estimate harvest volume", "Forecast precipitation"]
        )
        st.markdown("---")
        st.info("💡 **Tip**: Use Auto-Detect for local weather!")
        st.caption("v1.3.0 | XAI Enabled")

    st.title("🌿 Explainable AI for Smart Agriculture")
    st.markdown("### Intelligent Decisions for Better Farming")
    
    preprocessor, crop_svm, ebmo, yield_lstm, rainfall_transformer = load_models()
    
    if not preprocessor:
        st.error("⚠️ Models not found! Please run `python main.py` first.")
        return

    # Container for Main Content
    with st.container():
        if app_mode == "Crop Recommendation":
            run_crop_recommendation(preprocessor, crop_svm, ebmo)
            
        elif app_mode == "Yield Forecasting":
            run_yield_forecasting(preprocessor, yield_lstm)
            
        elif app_mode == "Rainfall Prediction":
            run_rainfall_prediction(preprocessor, rainfall_transformer)
            
    # Professional Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; font-size: 0.8rem; color: white; opacity: 0.8;">
        Smart Agri XAI © 2025 | Developed for Sustainable Farming 🌾
    </div>
    """, unsafe_allow_html=True)

import zlib
import random

def get_live_weather(location):
    """
    Simulate fetching live data based on location.
    Uses robust hashing to ensure consistent but varied results for unknown locations.
    """
    loc_lower = location.lower().strip()
    
    # 1. Expanded Realism Database (Major Indian Cities & Zones)
    # Added 'soil' keys: N (Nitrogen), P (Phosphorus), K (Potassium), ph
    weather_db = {
        # North (Alluvial Soil - Balanced/High N)
        'delhi':      {'temp': 34.0, 'hum': 35.0, 'rain': 50.0, 'wind': 15.0, 'press': 1005.0, 'n': 90, 'p': 40, 'k': 40, 'ph': 7.0},
        'new delhi':  {'temp': 34.0, 'hum': 35.0, 'rain': 50.0, 'wind': 15.0, 'press': 1005.0, 'n': 90, 'p': 40, 'k': 40, 'ph': 7.0},
        'shimla':     {'temp': 15.0, 'hum': 60.0, 'rain': 120.0, 'wind': 10.0, 'press': 950.0,  'n': 60, 'p': 40, 'k': 30, 'ph': 5.5}, # Acidic
        'srinagar':   {'temp': 12.0, 'hum': 55.0, 'rain': 60.0, 'wind': 8.0, 'press': 960.0,   'n': 50, 'p': 30, 'k': 30, 'ph': 6.0},
        'chandigarh': {'temp': 30.0, 'hum': 40.0, 'rain': 80.0, 'wind': 12.0, 'press': 1002.0, 'n': 100, 'p': 45, 'k': 40, 'ph': 7.2},
        'jaipur':     {'temp': 38.0, 'hum': 25.0, 'rain': 20.0, 'wind': 18.0, 'press': 1000.0, 'n': 60, 'p': 50, 'k': 60, 'ph': 8.0}, # Sandy/Alkaline
        'lucknow':    {'temp': 32.0, 'hum': 50.0, 'rain': 90.0, 'wind': 10.0, 'press': 1004.0, 'n': 110, 'p': 50, 'k': 50, 'ph': 7.5},
        'varanasi':   {'temp': 33.0, 'hum': 55.0, 'rain': 100.0, 'wind': 8.0, 'press': 1003.0, 'n': 100, 'p': 40, 'k': 40, 'ph': 7.4},

        # West (Black Soil - Rich in Clay/K)
        'mumbai':    {'temp': 28.0, 'hum': 85.0, 'rain': 350.0, 'wind': 22.0, 'press': 1008.0, 'n': 60, 'p': 40, 'k': 60, 'ph': 6.8}, 
        'pune':      {'temp': 26.0, 'hum': 65.0, 'rain': 120.0, 'wind': 15.0, 'press': 1010.0, 'n': 70, 'p': 50, 'k': 70, 'ph': 7.0},
        'nagpur':    {'temp': 36.0, 'hum': 40.0, 'rain': 110.0, 'wind': 10.0, 'press': 1002.0, 'n': 80, 'p': 45, 'k': 50, 'ph': 7.2},
        'ahmedabad': {'temp': 35.0, 'hum': 45.0, 'rain': 60.0, 'wind': 14.0, 'press': 1005.0, 'n': 70, 'p': 50, 'k': 60, 'ph': 7.6},
        'surat':     {'temp': 30.0, 'hum': 75.0, 'rain': 200.0, 'wind': 16.0, 'press': 1007.0, 'n': 65, 'p': 55, 'k': 55, 'ph': 7.3},

        # South (Red/Laterite Soil)
        'chennai':            {'temp': 32.0, 'hum': 80.0, 'rain': 180.0, 'wind': 25.0, 'press': 1006.0, 'n': 50, 'p': 40, 'k': 30, 'ph': 6.5},
        'bangalore':          {'temp': 23.0, 'hum': 60.0, 'rain': 100.0, 'wind': 18.0, 'press': 980.0,  'n': 80, 'p': 40, 'k': 50, 'ph': 6.9},
        'hyderabad':          {'temp': 31.0, 'hum': 50.0, 'rain': 70.0, 'wind': 12.0, 'press': 1009.0,  'n': 75, 'p': 50, 'k': 50, 'ph': 7.0},
        'kochi':              {'temp': 28.0, 'hum': 90.0, 'rain': 450.0, 'wind': 20.0, 'press': 1008.0, 'n': 120, 'p': 40, 'k': 40, 'ph': 5.5}, # Acidic
        'thiruvananthapuram': {'temp': 29.0, 'hum': 85.0, 'rain': 300.0, 'wind': 18.0, 'press': 1008.0, 'n': 100, 'p': 35, 'k': 40, 'ph': 5.8},

        # East (Alluvial/Clay)
        'kolkata':     {'temp': 30.0, 'hum': 85.0, 'rain': 280.0, 'wind': 15.0, 'press': 1001.0, 'n': 85, 'p': 45, 'k': 45, 'ph': 6.6},
        'patna':       {'temp': 31.0, 'hum': 60.0, 'rain': 110.0, 'wind': 10.0, 'press': 1003.0, 'n': 90, 'p': 40, 'k': 35, 'ph': 7.0},
        'bhubaneswar': {'temp': 32.0, 'hum': 75.0, 'rain': 200.0, 'wind': 18.0, 'press': 1005.0, 'n': 80, 'p': 40, 'k': 40, 'ph': 6.8},
        'guwahati':    {'temp': 26.0, 'hum': 88.0, 'rain': 300.0, 'wind': 10.0, 'press': 1002.0, 'n': 100, 'p': 40, 'k': 40, 'ph': 5.9},
    }
    
    if loc_lower in weather_db:
        base = weather_db[loc_lower]
        seed_val = zlib.adler32(loc_lower.encode('utf-8'))
        random.seed(seed_val)
        
        return {
            'temp': base['temp'] + round(random.uniform(-2, 2), 1),
            'hum': max(0, min(100, base['hum'] + round(random.uniform(-5, 5), 1))),
            'rain': max(0, base['rain'] + round(random.uniform(-20, 20), 1)),
            'wind': base['wind'] + round(random.uniform(-2, 2), 1),
            'press': base['press'] + round(random.uniform(-2, 2), 1),
            # Soil Params with jitter
            'n': max(0, int(base['n'] + random.uniform(-10, 10))),
            'p': max(0, int(base['p'] + random.uniform(-5, 5))),
            'k': max(0, int(base['k'] + random.uniform(-5, 5))),
            'ph': max(0, min(14, round(base['ph'] + random.uniform(-0.5, 0.5), 1)))
        }
    
    else:
        seed_val = zlib.adler32(loc_lower.encode('utf-8'))
        random.seed(seed_val)
        
        return {
            'temp': round(random.uniform(10.0, 42.0), 1),
            'hum': round(random.uniform(20.0, 95.0), 1),
            'rain': round(random.uniform(10.0, 400.0), 1),
            'wind': round(random.uniform(5.0, 30.0), 1),
            'press': round(random.uniform(990.0, 1020.0), 1),
            # Random Soil for unknown
            'n': int(random.uniform(20, 120)),
            'p': int(random.uniform(10, 80)),
            'k': int(random.uniform(10, 80)),
            'ph': round(random.uniform(5.0, 8.5), 1)
        }

def display_preventive_measures(rain, temp, hum):
    """Displays risk warnings and preventive measures with calibrated thresholds."""
    
    # CSS for Risk Cards
    st.markdown("""
    <style>
    .risk-card {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    .risk-title { font-weight: bold; color: #856404; display: flex; align-items: center; gap: 10px; }
    .risk-body { font-size: 0.95rem; color: #555; margin-top: 5px; }
    
    .safe-card {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🛡️ Climate Risk Advisory")
    
    risks = []
    
    # Thresholds calibrated to Data Generator
    # 1. Drought
    if rain < 100: 
        risks.append({
            "title": "Drought Risk / Water Scarcity",
            "icon": "🌵",
            "msg": f"Rainfall ({rain}mm) is significantly below average.",
            "steps": ["Implement drip irrigation immediately.", "Apply organic mulch to retain soil moisture.", "Avoid water-intensive crops like Rice."]
        })
        
    # 2. Flood
    elif rain > 300:
        risks.append({
            "title": "Flood Risk / Excess Moisture",
            "icon": "🌊",
            "msg": f"Rainfall ({rain}mm) is potentially excessive.",
            "steps": ["Ensure field drainage channels are clear.", "Use raised bed planting.", "Monitor for root rot diseases."]
        })

    # 3. Heat Stress
    if temp > 32:
        risks.append({
            "title": "Heat Stress Warning",
            "icon": "☀️",
            "msg": f"Temperature ({temp}°C) is high for many crops.",
            "steps": ["Irrigate during evening hours.", "Use shade nets for sensitive plants.", "Increase potassium fertilizer to boost stress tolerance."]
        })
        
    # 4. Cold/Frost
    elif temp < 18:
        risks.append({
            "title": "Cold Stress / Frost Risk",
            "icon": "❄️",
            "msg": f"Temperature ({temp}°C) is lower than optimal.",
            "steps": ["Use row covers or plastic tunnels.", "Apply irrigation before frost nights (water holds heat).", "Smoke generation around fields."]
        })

    # 5. Disease (Humidity)
    if hum > 85:
        risks.append({
            "title": "Fungal Disease Alert",
            "icon": "🍄",
            "msg": f"High Humidity ({hum}%) favors pathogen growth.",
            "steps": ["Monitor leaves for spots/mold daily.", "Improve air circulation by pruning/spacing.", "Apply preventive organic fungicides (Neem oil)."]
        })
        
    if risks:
        for r in risks:
            steps_html = "".join([f"<li>{s}</li>" for s in r['steps']])
            st.markdown(f"""
            <div class="risk-card">
                <div class="risk-title">{r['icon']} {r['title']}</div>
                <div class="risk-body">{r['msg']}
                    <ul>{steps_html}</ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="safe-card">
            <div class="risk-title" style="color: #155724;">✅ Optimal Conditions</div>
            <div class="risk-body">The current climate profile is balanced and favorable for most crops. No specific preventive actions required.</div>
        </div>
        """, unsafe_allow_html=True) 

def run_crop_recommendation(preprocessor, model, ebmo):
    st.header("🌱 Crop Recommendation System")
    
    # Session State (Initialize Weather AND Soil)
    if 'c_temp' not in st.session_state: st.session_state['c_temp'] = 25.0
    if 'c_hum' not in st.session_state: st.session_state['c_hum'] = 80.0
    if 'c_rain' not in st.session_state: st.session_state['c_rain'] = 200.0
    
    if 'c_n' not in st.session_state: st.session_state['c_n'] = 90
    if 'c_p' not in st.session_state: st.session_state['c_p'] = 40
    if 'c_k' not in st.session_state: st.session_state['c_k'] = 40
    if 'c_ph' not in st.session_state: st.session_state['c_ph'] = 6.5
    
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
            
            # Update Soil State
            st.session_state['c_n'] = w['n']
            st.session_state['c_p'] = w['p']
            st.session_state['c_k'] = w['k']
            st.session_state['c_ph'] = w['ph']
            
            st.toast(f"Detected Environment for {loc}!")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Soil Parameters")
        n = st.number_input("Nitrogen (N)", 0, 140, key='c_n')
        p = st.number_input("Phosphorous (P)", 0, 145, key='c_p')
        k = st.number_input("Potassium (K)", 0, 205, key='c_k')
        ph = st.number_input("pH Level", 0.0, 14.0, key='c_ph')
        
    with col2:
        st.subheader("Weather Parameters")
        temp = st.number_input("Temperature (°C)", 0.0, 50.0, key='c_temp')
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, key='c_hum')
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, key='c_rain')

def display_preventive_measures(rain, temp, hum):
    """Displays risk warnings and preventive measures with calibrated thresholds."""
    
    # CSS for Risk Cards
    st.markdown("""
    <style>
    .risk-card {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    .risk-title { font-weight: bold; color: #856404; display: flex; align-items: center; gap: 10px; }
    .risk-body { font-size: 0.95rem; color: #555; margin-top: 5px; }
    
    .safe-card {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🛡️ Climate Risk Advisory")
    
    risks = []
    
    # Thresholds calibrated to Data Generator (Mean Rain ~200mm, Temp ~25-30C)
    # 1. Drought (Lower 10th percentile approx)
    if rain < 100: 
        risks.append({
            "title": "Drought Risk / Water Scarcity",
            "icon": "🌵",
            "msg": f"Rainfall ({rain}mm) is significantly below average.",
            "steps": ["Implement drip irrigation immediately.", "Apply organic mulch to retain soil moisture.", "Avoid water-intensive crops like Rice."]
        })
        
    # 2. Flood (Upper 10th percentile)
    elif rain > 300:
        risks.append({
            "title": "Flood Risk / Excess Moisture",
            "icon": "🌊",
            "msg": f"Rainfall ({rain}mm) is potentially excessive.",
            "steps": ["Ensure field drainage channels are clear.", "Use raised bed planting.", "Monitor for root rot diseases."]
        })

    # 3. Heat Stress
    if temp > 32:
        risks.append({
            "title": "Heat Stress Warning",
            "icon": "☀️",
            "msg": f"Temperature ({temp}°C) is high for many crops.",
            "steps": ["Irrigate during evening hours.", "Use shade nets for sensitive plants.", "Increase potassium fertilizer to boost stress tolerance."]
        })
        
    # 4. Cold/Frost
    elif temp < 18:
        risks.append({
            "title": "Cold Stress / Frost Risk",
            "icon": "❄️",
            "msg": f"Temperature ({temp}°C) is lower than optimal.",
            "steps": ["Use row covers or plastic tunnels.", "Apply irrigation before frost nights (water holds heat).", "Smoke generation around fields."]
        })

    # 5. Disease (Humidity)
    if hum > 85:
        risks.append({
            "title": "Fungal Disease Alert",
            "icon": "🍄",
            "msg": f"High Humidity ({hum}%) favors pathogen growth.",
            "steps": ["Monitor leaves for spots/mold daily.", "Improve air circulation by pruning/spacing.", "Apply preventive organic fungicides (Neem oil)."]
        })
        
    if risks:
        for r in risks:
            steps_html = "".join([f"<li>{s}</li>" for s in r['steps']])
            st.markdown(f"""
            <div class="risk-card">
                <div class="risk-title">{r['icon']} {r['title']}</div>
                <div class="risk-body">{r['msg']}
                    <ul>{steps_html}</ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="safe-card">
            <div class="risk-title" style="color: #155724;">✅ Optimal Conditions</div>
            <div class="risk-body">The current climate profile is balanced and favorable for most crops. No specific preventive actions required.</div>
        </div>
        """, unsafe_allow_html=True)

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
                
                best_agronomic_crop = top_3_crops[0]
                best_prob = probs[top_3_idx[0]]
                
                # Logic Refinement:
                # Only switch to "Economic Choice" if:
                # 1. It is reasonably suitable (>20% probability)
                # 2. The Current Best isn't overwhelmingly sure (>70% confidence usually implies strict conditions)
                
                suitable_crops = [crop for i, crop in enumerate(top_3_crops) if probs[top_3_idx[i]] > 0.2]
                if not suitable_crops: suitable_crops = [best_agronomic_crop]
                
                # Find most profitable among strictly suitable
                best_profit_crop = max(suitable_crops, key=lambda c: CROP_PROFIT.get(c, 0))
                
                st.write("---")
                st.markdown("### 💰 Profitability & Selection")
                
                cA, cB = st.columns(2)
                
                with cA:
                    st.success(f"🌱 **Agronomic Best Choice**: **{best_agronomic_crop}**")
                    st.caption(f"Best match for soil/weather (Confidence: {best_prob:.2%})")
                    
                with cB:
                    st.info(f"💵 **Economic Best Choice**: **{best_profit_crop}**")
                    profit = CROP_PROFIT.get(best_profit_crop, 0)
                    st.caption(f"Highest profit among suitable candidates (~₹{profit}/ha)")
                
                # Final Decision Logic
                # If Agronomic Confidence is High (>50%), trust biology over small profit risks.
                # Only switch if the profitable crop is also a strong candidate (>30%).
                
                is_high_confidence = best_prob > 0.50
                # Fix: Convert numpy array to list for .index() method
                profit_candidate_prob = probs[top_3_idx[list(top_3_crops).index(best_profit_crop)]] if best_profit_crop in top_3_crops else 0

                if best_profit_crop != best_agronomic_crop:
                    if is_high_confidence and profit_candidate_prob < 0.3:
                        st.warning(f"**Insight**: **{best_profit_crop}** is more profitable, but **{best_agronomic_crop}** is significantly better suited for your conditions ({best_prob:.1%} vs {profit_candidate_prob:.1%}). **We recommend sticking with {best_agronomic_crop}.**")
                        crop_name = best_agronomic_crop
                    else:
                        st.success(f"**Insight**: **{best_profit_crop}** is a great alternative! It is suitable for your land and offers higher returns than {best_agronomic_crop}. **Recommended: {best_profit_crop}**")
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
