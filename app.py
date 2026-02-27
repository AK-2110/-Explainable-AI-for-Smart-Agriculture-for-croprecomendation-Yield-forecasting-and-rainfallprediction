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
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import base64
import os
from PIL import Image

from src.xlnet_features import XLNetFeatureExtractor
from src.feature_selection import EBMOFeatureSelection
from src.preprocessing import DataPreprocessor

# Force reload knowledge base so updates take effect immediately in Streamlit
import sys
import importlib
if 'src.crop_kb' in sys.modules:
    importlib.reload(sys.modules['src.crop_kb'])
from src.crop_kb import CROP_KNOWLEDGE_BASE, REGION_CROP_MAP
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

def text_to_audio(text, lang_code='en'):
    try:
        # Create audio file with gTTS
        tts = gTTS(text=text, lang=lang_code[:2] if '-' in lang_code else lang_code)
        filename = f"temp_audio_{random.randint(1000, 9999)}.mp3"
        tts.save(filename)
        
        # Read file as bytes
        with open(filename, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            
        # Clean up temp file
        os.remove(filename)
        
        # Return HTML audio player
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        return md
    except Exception as e:
        return None

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

def predict_crop_logic(data, city_name=None):
    # 1. Base list of possible crops based on simple heuristics
    recommendations = []
    if data['rainfall'] > 1500: recommendations.append("Rice")
    if data['temp'] > 35 and data['hum'] < 40: recommendations.append("Chickpea")
    if data['soil_type'] == 'Black Soil': recommendations.append("Cotton")
    if data['pH'] < 5.5: recommendations.append("Tea")
    
    # Defaults
    recommendations.extend(["Groundnut", "Maize"])
    
    # Remove duplicates while preserving some order
    recommendations = list(dict.fromkeys(recommendations))
    
    # 2. Apply Regional Constraint (if city is known)
    if city_name:
        key = city_name.lower().strip()
        allowed_crops = REGION_CROP_MAP.get(key)
        
        if allowed_crops:
            # Prioritize crops that naturally match heuristics AND region
            valid_recs = [c for c in recommendations if c in allowed_crops]
            
            # Fill the rest with other allowed crops for that region
            for c in allowed_crops:
                if c not in valid_recs:
                    valid_recs.append(c)
                    
            return valid_recs[:3] # Return top 3
                
    return recommendations[:3] # Fallback top 3

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
        
        # Voice Input for Location
        st.write(translate("🎤 Speak Location:", lang_code))
        voice_input = speech_to_text(language=lang_code, use_container_width=True, just_once=True, key='STT')
        
        # Populate City Input with voice input if available, else use default/typed
        default_city = "Kadapa"
        if voice_input:
             # Remove trailing periods that speech-to-text sometimes adds
             current_input = voice_input.strip('.')
        else:
             current_input = default_city
             
        city_input = st.text_input(translate("Enter Location (City/Region)", lang_code), current_input)
        
        analyze_clicked = st.button(translate("🔍 Analyze Region", lang_code), use_container_width=True)
        
        if analyze_clicked or voice_input:
            # Predict based on the just-spoken voice input immediately, or the typed input
            city_to_analyze = current_input if voice_input else city_input
            
            with st.spinner(f"{translate('Satellite scanning', lang_code)} {city_to_analyze}..."):
                time.sleep(1.2)
                data = fetch_geo_data(city_to_analyze)
                st.session_state['geo_data'] = data
                st.session_state['location'] = city_to_analyze
                
                # Update Session State for Manual Edits
                st.session_state['temp_in'] = data['temp']
                st.session_state['rain_in'] = data['rainfall']
                st.session_state['ph_in'] = data['pH']
                st.session_state['n_in'] = data['N']
                
                # Auto-Run Logic (Initial)
                crop = predict_crop_logic(data, city_input)
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
    t4_name = translate("🩺 Disease Detection", lang_code)
    t5_name = translate("🎙️ Voice Assistant", lang_code)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([t1_name, t2_name, t3_name, t4_name, t5_name])
    
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
                    new_crop = predict_crop_logic(new_data, st.session_state.get('location'))
                    st.session_state['predicted_crop'] = new_crop
                    st.session_state['geo_data'] = new_data # Update source of truth
                    st.snow() # Visual feedback

            # Use current (possibly updated) data for display logic validation
            current_data = st.session_state['geo_data']
            crops_list = st.session_state['predicted_crop']
            primary_crop = crops_list[0]
            
            st.markdown("---")
            
            # Result Section
            c_res, c_info = st.columns([1, 2])
            
            with c_res:
                st.markdown(f"#### ✅ {translate('Recommended Crop', lang_code)}")
                # Animated Card
                lottie_crop = load_lottieurl(LOTTIE_SUCCESS)
                if lottie_crop: st_lottie(lottie_crop, height=120, key="res_anim")
                
                st.success(f"**{translate(primary_crop, lang_code)}**")
                
                if len(crops_list) > 1:
                    st.markdown(f"**{translate('Other highly suitable crops:', lang_code)}**")
                    for c in crops_list[1:]:
                        st.info(f"🌿 {translate(c, lang_code)}")
                
                # Text-to-Audio for the result
                base_text = f"{primary_crop} is the highly recommended crop based on your region's analysis."
                spoken_result = translate(base_text, lang_code)
                audio_html = text_to_audio(spoken_result, lang_code)
                if audio_html:
                    st.markdown(audio_html, unsafe_allow_html=True)
            
            with c_info:
                 # FETCH KNOWLEDGE BASE
                 if primary_crop in CROP_KNOWLEDGE_BASE:
                     kb = CROP_KNOWLEDGE_BASE[primary_crop]
                     
                     with st.expander(f"🚜 {translate(primary_crop, lang_code)} - {translate('Cultivation Process (Step-by-Step)', lang_code)}", expanded=True):
                         for step in kb['cultivation']:
                             st.write(translate(step, lang_code))
                             
                     with st.expander(f"🦠 {translate(primary_crop, lang_code)} - {translate('Diseases & Protection', lang_code)}"):
                         st.markdown(f"**{translate('Common Diseases', lang_code)}:**")
                         for d in kb['diseases']:
                             st.write(f"- {translate(d, lang_code)}")
                         st.markdown(f"**{translate('Precautions', lang_code)}:**")
                         st.write(translate(kb['protection'], lang_code))
                 else:
                     st.info(translate("Detailed knowledge base for this crop is being updated.", lang_code))

            # --- COMPARISON SECTION ---
            st.markdown("---")
            st.subheader(f"📊 {translate('Crop Comparison Analysis', lang_code)}")
            
            comp_data = []
            for c in crops_list:
                if c in CROP_KNOWLEDGE_BASE:
                    kb = CROP_KNOWLEDGE_BASE[c]
                    comp_data.append({
                        translate('Crop', lang_code): translate(c, lang_code),
                        translate('Season', lang_code): translate(kb.get('season', 'Varied'), lang_code),
                        translate('Cultivation Time', lang_code): translate(kb.get('duration', 'N/A'), lang_code),
                        translate('Harvest Time', lang_code): translate(kb.get('harvest_time', 'N/A'), lang_code),
                        translate('Budget (INR/Acre)', lang_code): kb.get('budget_per_acre', 0),
                        translate('Expected Price (INR/Ton)', lang_code): kb.get('price', 0)
                    })
            
            if comp_data:
                df_comp = pd.DataFrame(comp_data)
                
                # Show DataFrame
                st.dataframe(
                    df_comp.style.format({
                        translate('Budget (INR/Acre)', lang_code): "₹{:,.0f}",
                        translate('Expected Price (INR/Ton)', lang_code): "₹{:,.0f}"
                    }), 
                    use_container_width=True, 
                    hide_index=True
                )
                
                # Visual Chart
                fig_comp = px.bar(
                    df_comp, 
                    x=translate('Crop', lang_code), 
                    y=translate('Budget (INR/Acre)', lang_code),
                    title=translate('Estimated Cultivation Budget Comparison', lang_code),
                    color=translate('Crop', lang_code),
                    template="plotly_dark",
                    text_auto='.2s'
                )
                st.plotly_chart(fig_comp, use_container_width=True)

        else:
            st.warning(translate("⚠️ Please enter a location and click 'Analyze Region' in the sidebar.", lang_code))

    # --- TAB 2: YIELD FORECASTING ---
    with tab2:
        if 'predicted_crop' in st.session_state:
            crops_list = st.session_state['predicted_crop']
            crop = crops_list[0]
            
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

    # --- TAB 4: DISEASE DETECTION ---
    with tab4:
        st.header(translate("🩺 Crop Disease Detection", lang_code))
        st.write(translate("Upload an image of your crop leaf or plant to detect the crop type and any diseases, plus get step-by-step solutions.", lang_code))

        uploaded_file = st.file_uploader(translate("Upload an image (JPG/PNG)", lang_code), type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            c1, c2 = st.columns(2)
            
            with c1:
                # Display the uploaded image
                st.image(uploaded_file, caption=translate("Uploaded Image", lang_code), use_container_width=True)
            
            with c2:
                # Mock AI Analysis Delay for Crop & Disease Detection
                with st.spinner(translate("🤖 AI scanning image to identify crop and anomalies...", lang_code)):
                    time.sleep(2.5)
                    
                # Mock AI Logic: detect crop type from image
                available_crops = list(CROP_KNOWLEDGE_BASE.keys())
                # For demo, randomly select a crop to simulate AI classification
                detected_crop = random.choice(available_crops)
                
                # Show detected crop
                st.info(f"**{translate('Detected Crop', lang_code)}:** {translate(detected_crop, lang_code)}")

                # Mock AI Logic: randomly select a disease for this detected crop, or "Healthy"
                kb_diseases = CROP_KNOWLEDGE_BASE[detected_crop]['diseases']
                # Add "Healthy" as an option occasionally, but heavily lean towards finding an issue for demo purposes
                analysis_results = kb_diseases + ["Healthy"]
                
                # Predict (Mock)
                detected = random.choice(analysis_results)
                
                if detected == "Healthy":
                    st.success(f"### ✅ {translate('Plant appears Healthy!', lang_code)}")
                    st.write(translate("No significant disease detected in the uploaded image. Maintain good agricultural practices.", lang_code))
                else:
                    st.error(f"### ⚠️ {translate('Disease Detected', lang_code)}!")
                    
                    # Split the disease string into Name and Remedy (format: "**Name**: Description. *Remedy*: Steps")
                    if "*Remedy*:" in detected:
                        issue_part, remedy_part = detected.split("*Remedy*:")
                        issue_text = issue_part.replace("**", "").strip()
                        remedy_text = remedy_part.strip()
                        
                        st.warning(f"**{translate('Issue Identified', lang_code)}:** {translate(issue_text, lang_code)}")
                        
                        st.markdown(f"#### 🛠️ {translate('Step-by-Step Solution', lang_code)}")
                        
                        # Generate step-by-step string
                        # Since remedy_text might be a simple sentence, we'll format it into steps
                        remedy_steps = [s.strip() for s in remedy_text.split(";") if s.strip()]
                        if not remedy_steps:
                            remedy_steps = [s.strip() for s in remedy_text.split(".") if s.strip()]
                            
                        for idx, step in enumerate(remedy_steps, 1):
                            if step:
                                st.write(f"**{translate('Step', lang_code)} {idx}:** {translate(step, lang_code)}")
                                
                    else:
                        st.warning(f"**{translate('Issue', lang_code)}:** {translate(detected, lang_code)}")

    # --- TAB 5: VOICE ASSISTANT ---
    with tab5:
        st.header(translate("🎙️ Interactive Voice Assistant", lang_code))
        st.write(translate("Speak your agriculture-related questions, and the AI will respond with voice!", lang_code))
        
        c_mic, c_ans = st.columns([1, 2])
        
        with c_mic:
            st.info(translate("Click the microphone to start speaking:", lang_code))
            assistant_query = speech_to_text(language=lang_code, use_container_width=True, just_once=True, key='STT_Assistant')
            
        with c_ans:
            # First, display any pending response from a previous rerun
            if 'voice_response_text' in st.session_state:
                st.write(f"**{translate('🎤 You asked:', lang_code)}** {st.session_state.get('voice_query', '')}")
                st.success(f"**{translate('🤖 AI Response:', lang_code)}** {st.session_state['voice_response_text']}")
                if st.session_state.get('voice_response_audio'):
                    st.markdown(st.session_state['voice_response_audio'], unsafe_allow_html=True)
                
                # Clear the pending response so it doesn't show again on next interaction
                del st.session_state['voice_response_text']
                del st.session_state['voice_response_audio']
                del st.session_state['voice_query']

            elif assistant_query:
                # Remove trailing periods
                clean_query = assistant_query.strip('.')
                st.write(f"**{translate('🎤 You asked:', lang_code)}** {clean_query}")
                
                with st.spinner(translate("🤖 AI is thinking...", lang_code)):
                    time.sleep(1.5)
                    
                    # Keyword-based AI Response Logic
                    # In a production app, this would be an LLM API call (e.g., Gemini or OpenAI)
                    crops_list = st.session_state.get('predicted_crop', ['your crops'])
                    crop_context = crops_list[0] if isinstance(crops_list, list) else crops_list
                    query_lower = clean_query.lower()
                    
                    if any(word in query_lower for word in ['yield', 'production', 'harvest', 'grow']):
                        selected_response_en = f"To maximize yield for {crop_context}, ensure optimal spacing and follow the recommended nutrient schedule."
                    elif any(word in query_lower for word in ['rain', 'weather', 'water', 'irrigation', 'dry']):
                        selected_response_en = f"Water management is critical. For {crop_context}, avoid waterlogging and provide light irrigation during dry spells."
                    elif any(word in query_lower for word in ['disease', 'sick', 'pest', 'insect', 'yellow', 'health']):
                        selected_response_en = f"If you notice issues in {crop_context}, look for common symptoms like leaf spots. You can also upload a photo in the Disease Detection tab for a precise diagnosis."
                    elif any(word in query_lower for word in ['fertilizer', 'soil', 'nutrient', 'ph', 'nitrogen', 'potassium', 'phosphorus']):
                        selected_response_en = f"For best results with {crop_context}, test your soil regularly and maintain a balanced pH. Apply fertilizers in split doses."
                    elif any(word in query_lower for word in ['recommend', 'crop', 'what to plant']):
                        # Extract location if mentioned
                        mock_cities = ['kadapa', 'hyderabad', 'mumbai', 'kashmir', 'nagpur', 'assam', 'punjab']
                        detected_city = st.session_state.get('location', 'Kadapa')
                        for city in mock_cities:
                            if city in query_lower:
                                detected_city = city.capitalize()
                                break
                        
                        # Auto-trigger analysis
                        data = fetch_geo_data(detected_city)
                        st.session_state['geo_data'] = data
                        st.session_state['location'] = detected_city
                        st.session_state['temp_in'] = data['temp']
                        st.session_state['rain_in'] = data['rainfall']
                        st.session_state['ph_in'] = data['pH']
                        st.session_state['n_in'] = data['N']
                        crop = predict_crop_logic(data, detected_city)
                        st.session_state['predicted_crop'] = crop
                        st.session_state['analysis_done'] = True
                        
                        selected_response_en = f"I have analyzed the location {detected_city}. Based on the environmental data, I recommend planting {crop}. All dashboard features including yield and rainfall predictions have been automatically updated for this region."
                    else:
                        selected_response_en = f"I am your Smart Agri Assistant. You asked about: '{clean_query}'. For {crop_context}, maintaining good agricultural practices and monitoring weather updates is always advised."
                    
                    translated_response = translate(selected_response_en, lang_code)
                    audio_html = text_to_audio(translated_response, lang_code)
                    
                    # Save response to state and force a rerun to update globally
                    st.session_state['voice_response_text'] = translated_response
                    st.session_state['voice_response_audio'] = audio_html
                    st.session_state['voice_query'] = clean_query
                    
                    if hasattr(st, 'rerun'):
                        st.rerun()
                    else:
                        st.experimental_rerun()

if __name__ == "__main__":
    main()
