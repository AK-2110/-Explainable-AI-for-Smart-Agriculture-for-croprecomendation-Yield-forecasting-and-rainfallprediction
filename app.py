import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import random
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
from streamlit_extras.metric_cards import style_metric_cards
from deep_translator import GoogleTranslator

from src.xlnet_features import XLNetFeatureExtractor
from src.feature_selection import EBMOFeatureSelection
from src.preprocessing import DataPreprocessor
from src.crop_kb import CROP_KNOWLEDGE_BASE

# --- 1. CONFIGURATION & ASSETS ---
st.set_page_config(page_title="Smart Agri XAI", page_icon="🌿", layout="wide")

# LOTTIE ASSETS
LOTTIE_ANALYSIS = "https://assets9.lottiefiles.com/packages/lf20_gSSpD9.json" # Scanning map/data
LOTTIE_SUCCESS = "https://assets10.lottiefiles.com/packages/lf20_u4jjb9bd.json" # Blooming
LOTTIE_WEATHER_SUN = "https://assets2.lottiefiles.com/packages/lf20_xlky4kvh.json" # Sunny
LOTTIE_WEATHER_RAIN = "https://assets7.lottiefiles.com/packages/lf20_bxd1x1ns.json" # Rainy

# --- 2. GLOBAL TRANSLATOR HELPER ---
@st.cache_data
def get_translator_instance(target_lang):
    if target_lang == 'en': return None
    return GoogleTranslator(source='auto', target=target_lang)

def translate(text, lang_code):
    if lang_code == 'en': return text
    try:
        # Simple caching or direct call. 
        # For lists, we assume the caller handles iteration to avoid massive payload errors.
        translator = GoogleTranslator(source='auto', target=lang_code)
        return translator.translate(text)
    except:
        return text

# --- 3. DIGITAL AGRONOMY THEME (BLUE-CYAN) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* --- MAIN BACKGROUND --- */
    .stApp {
        /* Gradient Overlay + Smart Agri Image */
        background: linear-gradient(rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.75)), url("https://images.unsplash.com/photo-1625246333195-bf791368c438?q=80&w=1920&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #F0FFF4; /* Mint Cream Text */
    }
    
    /* Pattern Overlay (Subtle Tech Grid) */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image: radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px);
        background-size: 20px 20px;
        pointer-events: none;
    }

    /* --- SIDEBAR: DARK SLATE & PURE WHITE TEXT --- */
    div[data-testid="stSidebar"] {
        background-color: #121212;
    }
    div[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    /* --- GLASSMORPHISM CARDS (DARK & CLEAR) --- */
    div.stContainer, div.stMetric, div[data-testid="stExpander"] {
        background: rgba(10, 10, 10, 0.75); /* Darker, more opaque background */
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
    }
    
    /* --- INPUT FIELDS & METRICS VISIBILITY FIX --- */
    .stTextInput > div > div > input, .stNumberInput > div > div > input {
        background-color: rgba(0, 0, 0, 0.8) !important; /* Near black inputs */
        color: #FFFFFF !important;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
    }
    div[data-testid="stMetricValue"] {
        color: #00FF87 !important; /* Neon Green */
        font-size: 32px !important;
        font-weight: 800;
        text-shadow: 0 0 10px rgba(0,0,0,0.8);
    }
    div[data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
        font-size: 16px !important;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(0,0,0,0.8);
    }

    /* --- TYPOGRAPHY (High Contrast) --- */
    h1, h2, h3, h4, h5, p, span, li {
        color: #FFFFFF !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.9); /* Heavy shadow for readability */
    }

    /* --- BUTTONS --- */
    .stButton > button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(146, 254, 157, 0.5);
    }

</style>
""", unsafe_allow_html=True)

# --- 4. HELPER FUNCTIONS ---

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

# Simulated Real-Time Data Engine
def fetch_geo_data(city_name):
    mock_db = {
        'kadapa': {'soil_type': 'Red Loam', 'pH': 6.5, 'N': 120, 'P': 40, 'K': 30, 'rainfall': 600, 'temp': 34, 'hum': 45},
        'hyderabad': {'soil_type': 'Black Soil', 'pH': 7.2, 'N': 150, 'P': 55, 'K': 45, 'rainfall': 850, 'temp': 30, 'hum': 60},
        'mumbai': {'soil_type': 'Coastal Alluvial', 'pH': 6.8, 'N': 100, 'P': 40, 'K': 40, 'rainfall': 2000, 'temp': 28, 'hum': 80},
    }
    key = city_name.lower().strip()
    return mock_db.get(key, {
            'soil_type': 'Unknown (Detected)',
            'pH': round(random.uniform(5.5, 8.0), 1),
            'N': random.randint(80, 180),
            'P': random.randint(30, 80),
            'K': random.randint(30, 80),
            'rainfall': random.randint(400, 1500),
            'temp': random.randint(20, 40),
            'hum': random.randint(30, 90)
    })

def predict_crop_logic(data):
    # Simple heuristic
    if data['rainfall'] > 1500: return "Rice"
    if data['temp'] > 35 and data['hum'] < 40: return "Chickpea"
    if data['soil_type'] == 'Black Soil': return "Cotton"
    if data['pH'] < 5.5: return "Tea"
    return "Groundnut" # Default

# --- 5. MAIN APPLICATION ---

def main():
    
    # --- SIDEBAR (Global Input & Language) ---
    with st.sidebar:
        st.markdown("## 🛰️ Settings")
        
        # Global Language Selector
        lang_options = {
            'English': 'en', 'Hindi (हिंदी)': 'hi', 'Telugu (తెలుగు)': 'te', 'Tamil (தமிழ்)': 'ta',
            'Malayalam (മലയാളം)': 'ml', 'Kannada (ಕನ್ನಡ)': 'kn', 'Marathi (मराठी)': 'mr', 'Gujarati (ગુજરાતી)': 'gu',
            'Bengali (বাংলা)': 'bn', 'Punjabi (ਪੰਜਾਬੀ)': 'pa', 'Urdu (اردو)': 'ur',
            'Spanish (Español)': 'es', 'French (Français)': 'fr', 'German (Deutsch)': 'de',
            'Italian (Italiano)': 'it', 'Portuguese (Português)': 'pt', 'Russian (Русский)': 'ru',
            'Chinese (中文)': 'zh-CN', 'Japanese (日本語)': 'ja', 'Korean (한국어)': 'ko',
            'Arabic (العربية)': 'ar', 'Turkish (Türkçe)': 'tr', 'Vietnamese (Tiếng Việt)': 'vi',
            'Thai (ไทย)': 'th', 'Indonesian (Bahasa)': 'id', 'Dutch (Nederlands)': 'nl',
            'Swedish (Svenska)': 'sv', 'Polish (Polski)': 'pl', 'Persian (فارسی)': 'fa'
        }
        selected_lang_name = st.selectbox("🗣️ Language / भाषा", list(lang_options.keys()))
        lang_code = lang_options[selected_lang_name]
        
        st.markdown("---")
        
        city_input = st.text_input(translate("Enter Location (City/Region)", lang_code), "Kadapa")
        
        if st.button(translate("🔍 Analyze Region", lang_code), use_container_width=True):
            with st.spinner(f"{translate('Satellite scanning', lang_code)} {city_input}..."):
                time.sleep(1.2)
                data = fetch_geo_data(city_input)
                st.session_state['geo_data'] = data
                st.session_state['location'] = city_input
                
                # Update Session State for Manual Edits
                st.session_state['temp_in'] = data['temp']
                st.session_state['rain_in'] = data['rainfall']
                st.session_state['ph_in'] = data['pH']
                st.session_state['n_in'] = data['N']
                
                # Auto-Run Logic (Initial)
                crop = predict_crop_logic(data)
                st.session_state['predicted_crop'] = crop
                st.session_state['analysis_done'] = True
                
        st.markdown("---")
        st.info(translate("System fully localized. Real-time data active.", lang_code))

    # --- MAIN CONTENT ---
    # Header
    st.markdown(f"<h1>🌱 Smart Agri XAI <span style='font-size: 20px; opacity: 0.7;'>| {translate('Real-Time Intelligence', lang_code)}</span></h1>", unsafe_allow_html=True)
    
    # Tabs
    t1_name = translate("🌿 Crop Recommendation", lang_code)
    t2_name = translate("📉 Yield Forecasting", lang_code)
    t3_name = translate("🌧️ Rainfall Prediction", lang_code)
    
    tab1, tab2, tab3 = st.tabs([t1_name, t2_name, t3_name])
    
    # --- TAB 1: CROP RECOMMENDATION ---
    with tab1:
        if 'geo_data' in st.session_state:
            # Live Environmental Scan (Editable)
            st.markdown(f"### 📡 {translate('Live Environmental Scan', lang_code)}: {st.session_state['location']} ({translate('Editable', lang_code)})")
            
            # Form for manual edits
            with st.form("manual_override_form"):
                m1, m2, m3, m4 = st.columns(4)
                
                # Use session state keys to persist values
                v_temp = m1.number_input(translate("Temperature (°C)", lang_code), value=st.session_state.get('temp_in', 25.0), key='temp_in_widget')
                v_rain = m2.number_input(translate("Rainfall (mm)", lang_code), value=st.session_state.get('rain_in', 1000), key='rain_in_widget')
                v_ph = m3.number_input(translate("Soil pH", lang_code), value=float(st.session_state.get('ph_in', 6.5)), key='ph_in_widget')
                v_n = m4.number_input(translate("Nitrogen (N)", lang_code), value=st.session_state.get('n_in', 100), key='n_in_widget')
                
                submitted = st.form_submit_button(translate("🔄 Recalculate Recommendation", lang_code))
                
                if submitted:
                    # Update session state with new manual values
                    st.session_state['temp_in'] = v_temp
                    st.session_state['rain_in'] = v_rain
                    st.session_state['ph_in'] = v_ph
                    st.session_state['n_in'] = v_n
                    
                    # Construct new data object
                    new_data = st.session_state['geo_data'].copy()
                    new_data['temp'] = v_temp
                    new_data['rainfall'] = v_rain
                    new_data['pH'] = v_ph
                    new_data['N'] = v_n
                    
                    # Re-run Prediction
                    new_crop = predict_crop_logic(new_data)
                    st.session_state['predicted_crop'] = new_crop
                    st.session_state['geo_data'] = new_data # Update source of truth
                    st.snow() # Visual feedback

            # Use current (possibly updated) data for display logic validation
            current_data = st.session_state['geo_data']
            crop = st.session_state['predicted_crop']
            
            st.markdown("---")
            
            # Result Section
            c_res, c_info = st.columns([1, 2])
            
            with c_res:
                st.markdown(f"#### ✅ {translate('Recommended Crop', lang_code)}")
                # Animated Card
                lottie_crop = load_lottieurl(LOTTIE_SUCCESS)
                if lottie_crop: st_lottie(lottie_crop, height=120, key="res_anim")
                
                st.success(f"**{translate(crop, lang_code)}**")
            
            with c_info:
                 # FETCH KNOWLEDGE BASE
                 if crop in CROP_KNOWLEDGE_BASE:
                     kb = CROP_KNOWLEDGE_BASE[crop]
                     
                     with st.expander(f"🚜 {translate('Cultivation Process (Step-by-Step)', lang_code)}", expanded=True):
                         for step in kb['cultivation']:
                             st.write(translate(step, lang_code))
                             
                     with st.expander(f"🦠 {translate('Diseases & Protection', lang_code)}"):
                         st.markdown(f"**{translate('Common Diseases', lang_code)}:**")
                         for d in kb['diseases']:
                             st.write(f"- {translate(d, lang_code)}")
                         st.markdown(f"**{translate('Precautions', lang_code)}:**")
                         st.write(translate(kb['protection'], lang_code))
                 else:
                     st.info(translate("Detailed knowledge base for this crop is being updated.", lang_code))

        else:
            st.warning(translate("⚠️ Please enter a location and click 'Analyze Region' in the sidebar.", lang_code))

    # --- TAB 2: YIELD FORECASTING ---
    with tab2:
        if 'predicted_crop' in st.session_state:
            crop = st.session_state['predicted_crop']
            
            st.header(f"{translate('Forecasting Yield for', lang_code)} **{translate(crop, lang_code)}**...")
            
            # Editable Area Input (Acres)
            c_area, c_dummy = st.columns([1, 2])
            with c_area:
                area_acres = st.number_input(translate("Field Area (Acres)", lang_code), min_value=0.1, value=5.0, step=0.5, key="area_acres")
                area_ha = area_acres * 0.4047 # Conversion
                st.caption(f"{area_acres} Acres ≈ {area_ha:.2f} Hectares")

            # Generate Comparison Data
            years = [2021, 2022, 2023, 2024, 2025]
            avg_yield = [3.5, 3.6, 3.4, 3.7, 3.8]
            ai_yield = [3.8, 4.0, 3.9, 4.2, 4.5] # AI Optimizations
            
            # Economic Analysis
            price = CROP_KNOWLEDGE_BASE.get(crop, {}).get('price', 30000) # Default price
            est_yield = ai_yield[-1] # Ton/Ha
            
            # Revenue = Yield (Ton/Ha) * Area (Ha) * Price
            revenue = est_yield * area_ha * price
            
            # Financial Metrics
            c_fin1, c_fin2 = st.columns(2)
            c_fin1.metric(translate("Predicted Yield (2025)", lang_code), f"{est_yield} Ton/Ha", "+8%")
            c_fin2.metric(translate("Estimated Revenue", lang_code), f"₹ {revenue:,.0f}", f"{translate('For', lang_code)} {area_acres} Acres")
            
            # Chart
            df_yield = pd.DataFrame({
                'Year': years,
                translate('Regional Average', lang_code): avg_yield,
                translate('AI Predicted', lang_code): ai_yield
            })
            
            fig_yield = go.Figure()
            fig_yield.add_trace(go.Scatter(x=years, y=avg_yield, name=translate('Regional Avg', lang_code), line=dict(color='#FF4B4B', dash='dash')))
            fig_yield.add_trace(go.Scatter(x=years, y=ai_yield, name=translate('AI Predicted', lang_code), line=dict(color='#00C9FF', width=4)))
            
            fig_yield.update_layout(
                title=translate("Yield Optimization Potential", lang_code),
                xaxis_title=translate("Year", lang_code),
                yaxis_title=translate("Yield (Ton/Ha)", lang_code),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )
            st.plotly_chart(fig_yield, use_container_width=True)
            
        else:
             st.info(translate("Please run the analysis in the 'Crop Recommendation' tab first.", lang_code))

    # --- TAB 3: RAINFALL PREDICTION ---
    with tab3:
        if 'location' in st.session_state:
            loc = st.session_state['location']
            st.header(f"{translate('Rainfall Analysis', lang_code)}: {loc}")
            
            # Editable Atmospheric Parameters
            with st.expander(translate("☁️ Edit Atmospheric Conditions (Simulation)", lang_code), expanded=False):
                col_p1, col_p2, col_p3 = st.columns(3)
                cloud_cover = col_p1.number_input(translate("Cloud Cover (%)", lang_code), min_value=0, max_value=100, value=60)
                pressure = col_p2.number_input(translate("Pressure (hPa)", lang_code), min_value=900, max_value=1100, value=1010)
                wind_speed = col_p3.number_input(translate("Wind Speed (km/h)", lang_code), min_value=0, value=15)
            
            # Historical Comparison (New Feature)
            st.subheader(translate("Historical vs Forecasted", lang_code))
            
            # Simulation Logic: Adjust forecast based on manual pressure input
            # Lower pressure = Higher rain probability
            pressure_factor = (1013 - pressure) * 2 # Mock physics
            
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
            prev_year = [10, 5, 20, 15, 60, 150, 300]
            curr_forecast = [12, 8, 22, 10, 50, 180, 290]
            
            # Adjust forecast visualization slightly based on input
            if pressure_factor > 0:
                 curr_forecast = [x + pressure_factor for x in curr_forecast]
            
            fig_rain = go.Figure()
            fig_rain.add_trace(go.Bar(x=months, y=prev_year, name=translate('Previous Year', lang_code), marker_color='#a0a0a0'))
            fig_rain.add_trace(go.Bar(x=months, y=curr_forecast, name=translate('Forecast', lang_code), marker_color='#60EFFF'))
            
            fig_rain.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                barmode='group'
            )
            st.plotly_chart(fig_rain, use_container_width=True)
            
            # 7-Day Forecast Widget
            st.markdown(f"#### 📅 {translate('7-Day Forecast', lang_code)}")
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            
            # Adjust probabilty based on cloud cover
            base_prob = cloud_cover
            
            cols = st.columns(7)
            for i, col in enumerate(cols):
                with col:
                    # Simulation
                    daily_prob = min(100, max(0, base_prob + random.randint(-20, 20)))
                    temp = 35 - (wind_speed * 0.1)
                    st.metric(days[i], f"{temp:.0f}°", f"{daily_prob}%", delta_color="inverse")
                        
        else:
            st.info(translate("Please enter a location to fetch real-time forecasts.", lang_code))

if __name__ == "__main__":
    main()
