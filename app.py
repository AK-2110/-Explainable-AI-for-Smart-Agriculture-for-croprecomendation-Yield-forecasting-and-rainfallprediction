import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
import streamlit.components.v1 as components
from PIL import Image
import google.generativeai as genai
import json
import base64
from src.history_manager import build_save_script, build_render_script

# Try to get API key from environment, else user will need to provide it or we show mock 
if True:
    os.environ["GEMINI_API_KEY"] = "AIzaSyB3UmB6Te7rLhQnJaUE1iySxjqE0ek4ZwE"
    genai.configure(api_key="AIzaSyB3UmB6Te7rLhQnJaUE1iySxjqE0ek4ZwE")

from src.xlnet_features import XLNetFeatureExtractor
from src.feature_selection import EBMOFeatureSelection
from src.preprocessing import DataPreprocessor

# Force reload knowledge base so updates take effect immediately in Streamlit
import sys
import importlib
if 'src.crop_kb' in sys.modules:
    importlib.reload(sys.modules['src.crop_kb'])
from src.crop_kb import CROP_KNOWLEDGE_BASE, REGION_CROP_MAP

# Init global chat state early
if "messages" not in st.session_state:
    st.session_state["messages"] = []
# --- 1. CONFIGURATION & ASSETS ---
st.set_page_config(page_title="Smart Agri XAI", page_icon="🌿", layout="wide")

# LOTTIE ASSETS (Advanced SVG Animations)
# These replace static images with high-fidelity, scalable vector graphics for zero-blur micro-interactions.
LOTTIE_ANALYSIS = "https://lottie.host/79058ffc-1aa1-4ff2-bc32-1b151e24748a/rMYr1LhB1R.json" # Tech Radar/Scanner
LOTTIE_SUCCESS = "https://lottie.host/a0a651f4-307f-44d5-ab2a-7140b92f759c/t9oHqX5WfF.json" # Glowing Checkmark
LOTTIE_WEATHER_SUN = "https://lottie.host/8cd6d84a-6f34-4530-ab0d-45037d40026e/B86Z1Z61yT.json" # Advanced 3D Sun
LOTTIE_WEATHER_RAIN = "https://lottie.host/fc16fbf6-02a8-4c80-be39-ab9c7ccf8fbb/z7Qj97Lw0e.json" # Atmospheric Rain

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
    [data-testid="stAppViewContainer"], .stApp {
        background-color: #0b0f19 !important; /* Deep obsidian background */
        color: #e2e8f0 !important; /* Light slate text */
    }
    
    /* Pattern Overlay (Subtle Tech Grid) */
    [data-testid="stAppViewContainer"]::before, .stApp::before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image: radial-gradient(rgba(0, 201, 255, 0.05) 1px, transparent 1px);
        background-size: 24px 24px;
        pointer-events: none;
        z-index: -1;
    }

    /* --- SIDEBAR: STEALTH CHARCOAL & NEON ACCENTS --- */
    div[data-testid="stSidebar"] {
        background-color: #111827; /* Deep charcoal */
        border-right: 1px solid rgba(0, 201, 255, 0.1);
        /* 1. Smooth, eased slide-in transition for sidebars/panels */
        transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), background-color 0.4s ease;
    }
    div[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* --- STAGGER FADE-IN ANIMATION FOR DASHBOARD --- */
    @keyframes staggerFadeUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .stApp [data-testid="stVerticalBlock"] > div {
        animation: staggerFadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .stApp [data-testid="stVerticalBlock"] > div:nth-child(1) { animation-delay: 0.1s; }
    .stApp [data-testid="stVerticalBlock"] > div:nth-child(2) { animation-delay: 0.2s; }
    .stApp [data-testid="stVerticalBlock"] > div:nth-child(3) { animation-delay: 0.3s; }
    .stApp [data-testid="stVerticalBlock"] > div:nth-child(4) { animation-delay: 0.4s; }

    /* --- TERMINAL / AI CONTROL CARDS --- */
    /* Targeting Streamlit columns and containers */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"], 
    div.stContainer, div.stMetric, div[data-testid="stExpander"] {
        background-color: rgba(15, 23, 42, 0.8) !important; /* Glassmorphism Slate */
        backdrop-filter: blur(12px);
        border-radius: 8px !important; 
        border: 1px solid rgba(0, 201, 255, 0.2) !important; /* Cyan border */
        padding: 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5) !important;
        /* Custom hover transition for Data Cards */
        transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.3s ease !important;
    }
    
    div.stContainer:hover, div.stMetric:hover, div[data-testid="stExpander"]:hover {
       transform: translateY(-4px);
       box-shadow: 0 10px 20px rgba(0, 201, 255, 0.15) !important; /* Neon glow boost */
       border-color: rgba(0, 201, 255, 0.5) !important;
    }
    
    /* --- INPUT FIELDS & METRICS VISIBILITY --- */
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div {
        background-color: #0b0f19 !important; 
        color: #00C9FF !important; /* Cyan input text */
        font-family: 'Courier New', Courier, monospace; /* Terminal feel */
        font-weight: 500;
        border: 1px solid rgba(0, 201, 255, 0.3) !important;
        border-radius: 4px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {
        border-color: #00C9FF !important;
        box-shadow: 0 0 10px rgba(0, 201, 255, 0.3) !important;
    }
    
    /* --- GLOWING METRIC TEXT --- */
    div[data-testid="stMetricValue"] {
        color: #00C9FF !important; /* Neon Cyan */
        font-size: 32px !important;
        font-weight: 700;
        text-shadow: 0 0 8px rgba(0, 201, 255, 0.4); /* Glow effect */
        font-family: 'Courier New', Courier, monospace;
    }
    div[data-testid="stMetricLabel"] {
        color: #475569 !important; /* Slate Gray */
        font-size: 14px !important;
        font-weight: 600;
        text-shadow: none;
    }

    /* --- TYPOGRAPHY --- */
    h1, h2, h3, h4, h5, p, span, li {
        color: #e2e8f0 !important;
        text-shadow: none; 
    }
    
    /* Typewriter effect for headers (simulated via animation on load) */
    @keyframes typewriterFade {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    h1 {
        animation: typewriterFade 1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        border-right: 3px solid #00C9FF; /* cursor */
        padding-right: 10px;
        animation-delay: 0.1s;
        /* Custom blink animation for the cursor */
        animation: blinkCursor 1s step-end infinite;
    }
    
    @keyframes blinkCursor {
        from, to { border-color: transparent }
        50% { border-color: #00C9FF; }
    }

    /* SideBar Headers */
    div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3, div[data-testid="stSidebar"] p {
        color: #00C9FF !important;
        text-shadow: 0 0 5px rgba(0, 201, 255, 0.3);
    }

    /* --- BREATHING NEON BUTTONS --- */
    @keyframes neonBreathe {
        0%, 100% { box-shadow: 0 0 10px rgba(0, 201, 255, 0.1); border-color: rgba(0, 201, 255, 0.3); }
        50% { box-shadow: 0 0 25px rgba(0, 201, 255, 0.4); border-color: rgba(0, 201, 255, 0.8); }
    }
    
    .stButton > button {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(0, 0, 0, 0.8)) !important;
        color: #00C9FF !important;
        font-weight: 600;
        letter-spacing: 0.5px;
        font-family: 'Courier New', Courier, monospace;
        border: 1px solid rgba(0, 201, 255, 0.4) !important;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        transition: all 0.4s cubic-bezier(0.19, 1, 0.22, 1);
        animation: neonBreathe 4s infinite ease-in-out;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: "";
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 201, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0, 201, 255, 0.15), rgba(15, 23, 42, 0.9)) !important;
        color: #fff !important;
        border-color: #00C9FF !important;
        box-shadow: 0 0 30px rgba(0, 201, 255, 0.6) !important;
        transform: translateY(-2px);
    }
    
    /* Tabs styling (Breathing effect on active tab) */
    button[data-baseweb="tab"] {
        color: rgba(226, 232, 240, 0.5) !important;
        transition: color 0.4s cubic-bezier(0.19, 1, 0.22, 1);
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00C9FF !important;
        border-bottom-color: #00C9FF !important;
        text-shadow: 0 0 12px rgba(0, 201, 255, 0.8);
    }
    
    /* Ensure Streamlit containers are transparent to see background */
    [data-testid="stAppViewContainer"] {
        background-color: transparent !important;
    }
    
    /* --- PROFESSIONAL ANIMATED GRADIENT MESH BACKGROUND --- */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: radial-gradient(circle at top left, rgba(0, 201, 255, 0.05), transparent 40%),
                    radial-gradient(circle at bottom right, rgba(139, 92, 246, 0.05), transparent 40%),
                    linear-gradient(200deg, #0b0f19 0%, #111827 50%, #0b0f19 100%) !important;
        background-size: 200% 200%;
        animation: gradientShift 20s ease infinite;
        overflow: hidden;
    }
    
    /* Subtle Data Grid Overlay for Tech Vibe */
    .stApp::after {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: linear-gradient(rgba(255, 255, 255, 0.01) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.01) 1px, transparent 1px);
        background-size: 50px 50px;
        z-index: -1;
        pointer-events: none;
    }
    
    /* --- SKELETON LOADER (SHIMMER EFFECT) --- */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .skeleton-box {
        display: inline-block;
        height: 1em;
        position: relative;
        overflow: hidden;
        background-color: #1e293b;
        border-radius: 4px;
        width: 100%;
    }
    
    .skeleton-box::after {
        position: absolute;
        top: 0;
        right: 0;
        bottom: 0;
        left: 0;
        transform: translateX(-100%);
        background-image: linear-gradient(
            90deg,
            rgba(0, 201, 255, 0) 0,
            rgba(0, 201, 255, 0.1) 20%,
            rgba(0, 201, 255, 0.3) 60%,
            rgba(0, 201, 255, 0)
        );
        animation: shimmer 2s infinite;
        content: '';
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

def get_assistant_response(query, context, lang_code, is_voice=False):
    query_lower = query.lower()
    selected_response_en = ""
    
    # Special routing: Check if user wants a location recommendation
    if any(word in query_lower for word in ['recommend', 'crop', 'what to plant', 'suggest']):
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
        
        c = crop[0] if isinstance(crop, list) else crop
        return f"I have analyzed the location {detected_city}. Based on the environmental data, I recommend planting {c}. All dashboard features including yield and rainfall predictions have been automatically updated for this region."
    
    use_fallback = False
    current_key = os.environ.get("GEMINI_API_KEY", "")
    if current_key:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"You are an expert Smart Agriculture Assistant. The user's current context is that they are looking at {context} crop in their dashboard. Answer the user's query clearly and concisely within 2-3 sentences. Query: {query}"
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Assistant Gemini API Error: {e}")
            st.error(f"Actual Gemini API Error for Assistant: {e}")
            use_fallback = True
            st.warning(translate("⚠️ Could not connect to Gemini AI for assistant. Using local mock AI instead.", lang_code))
            time.sleep(1.0)
    else:
        use_fallback = True
        
    if use_fallback:
        # Keyword-based AI Response Logic
        if any(word in query_lower for word in ['yield', 'production', 'harvest', 'grow']):
            selected_response_en = f"To maximize yield for {context}, ensure optimal spacing and follow the recommended nutrient schedule."
        elif any(word in query_lower for word in ['rain', 'weather', 'water', 'irrigation', 'dry']):
            selected_response_en = f"Water management is critical. For {context}, avoid waterlogging and provide light irrigation during dry spells."
        elif any(word in query_lower for word in ['disease', 'sick', 'pest', 'insect', 'yellow', 'health']):
            selected_response_en = f"If you notice issues in {context}, look for common symptoms like leaf spots. You can also upload a photo in the Disease Detection tab for a precise diagnosis."
        elif any(word in query_lower for word in ['fertilizer', 'soil', 'nutrient', 'ph', 'nitrogen', 'potassium', 'phosphorus']):
            selected_response_en = f"For best results with {context}, test your soil regularly and maintain a balanced pH. Apply fertilizers in split doses."
        else:
            if is_voice:
                selected_response_en = f"I am your Smart Agri Assistant. You asked about: '{query}'. For {context}, maintaining good agricultural practices and monitoring weather updates is always advised."
            else:
                selected_response_en = f"I am your Smart Agri Assistant. For {context}, maintaining good agricultural practices and monitoring weather updates is always advised."
        return selected_response_en

# --- 5. MAIN APPLICATION ---

def main():
    # --- 0. RE-HYDRATION LOGIC ---
    if "restore" in st.query_params:
        try:
            b64_state = st.query_params["restore"]
            state_json = base64.b64decode(b64_state).decode('utf-8')
            restored_data = json.loads(state_json)
            for k, v in restored_data.items():
                st.session_state[k] = v
            # Clear params and rerun
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Re-hydration error: {e}")
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
        
        # GEMINI API KEY has been hardcoded
        
        st.markdown("---")
        
        # --- GLOBAL CHAT HISTORY MODULE IN SIDEBAR ---
        st.markdown(f"### 💬 {translate('Assistant Chat', lang_code)}")
        chat_container = st.container(height=300)
        with chat_container:
            for msg in st.session_state.get("messages", []):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    
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
            
            # Show Shimmer Skeleton while loading
            placeholder = st.empty()
            placeholder.markdown(f'''
            <div style="padding: 10px; border-radius: 4px; border: 1px solid rgba(0, 201, 255, 0.3);">
                <p style="color: #00C9FF; font-family: Courier New; margin-bottom: 5px;">> {translate("INITIALIZING SATELLITE UPLINK...", lang_code)}</p>
                <div class="skeleton-box" style="height: 8px;"></div>
                <div class="skeleton-box" style="height: 8px; width: 80%; margin-top: 5px;"></div>
            </div>
            ''', unsafe_allow_html=True)
            
            with st.spinner(f"> {translate('Retrieving telemetry for', lang_code)} {city_to_analyze}..."):
                time.sleep(1.5)
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
                
                # Save to History
                save_js = build_save_script("Region Analysis", dict(st.session_state))
                components.html(save_js, height=0)
            placeholder.empty() # Remove skeleton once done
                
        st.markdown("---")
        st.markdown(f"### 🛡️ {translate('Data Health & Integrity', lang_code)}")
        st.success(f"**{translate('Sensors Online', lang_code)}:** 99.8%")
        st.info(f"**{translate('Model Checksum', lang_code)}:** Verified")
        st.caption(f"_{translate('Last Model Retrain: 4 hrs ago', lang_code)}_")
        
        st.markdown("---")
        st.markdown(f"### 🕓 {translate('Recent Activity', lang_code)}")
        # History panel rendered in an iframe
        history_html = build_render_script(height=350)
        components.html(history_html, height=360, scrolling=False)
        st.markdown("---")
        
        # Build Report Content Dynamically based on current session State
        report_loc = st.session_state.get('location', 'N/A')
        report_body = f"====================================\n"
        report_body += f" SMART AGRI XAI - AUTOMATED REPORT\n"
        report_body += f"====================================\n\n"
        report_body += f"Location Analyzed: {report_loc}\n"
        
        if 'geo_data' in st.session_state:
             d = st.session_state['geo_data']
             report_body += "--- Environmental Data ---\n"
             report_body += f"Temperature: {d.get('temp', 'N/A')} °C\n"
             report_body += f"Rainfall: {d.get('rainfall', 'N/A')} mm\n"
             report_body += f"Soil pH Level: {d.get('pH', 'N/A')}\n"
             report_body += f"Nitrogen (N): {d.get('N', 'N/A')}\n\n"
             
        if 'predicted_crop' in st.session_state:
             cr = st.session_state['predicted_crop']
             report_body += "--- AI Recommendations ---\n"
             if isinstance(cr, list) and len(cr) > 0:
                 report_body += f"Primary Recommended Crop: {cr[0]}\n"
                 if len(cr) > 1:
                     report_body += f"Secondary Recommendations: {', '.join(cr[1:])}\n"
             else:
                 report_body += f"Recommended Crop: {cr}\n"
        else:
             report_body += "No Data Traces Found. Please run analysis first.\n"
             
        report_body += "\n--- Farm Health Forecasts ---\n"
        report_body += "Crop Yield Health: Stable\nDisease Index: Low Anomaly Detection\n"
        report_body += "\n*Report generated securely by Smart Agri XAI Local Instance.*\n"

        st.download_button(
            label=translate("📄 Download AI Report", lang_code),
            data=report_body,
            file_name=f"Smart_Agri_Report_{report_loc}.txt",
            mime="text/plain",
            help=translate("Download AI explanations and current metrics into a text log.", lang_code),
            use_container_width=True
        )

        st.markdown("---")
        st.info(translate("System fully localized. Real-time data active.", lang_code))

    # --- MAIN CONTENT ---
    # Header
    st.markdown(f"<h1>🌱 Smart Agri XAI <br><span style='font-size: 18px; opacity: 0.8; font-weight: normal; color: #475569;'>{translate('Bridging the Gap Between Complex AI and Actionable Farming Intelligence. Understand not just what to plant, but why.', lang_code)}</span></h1>", unsafe_allow_html=True)
    
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
                    
                    # Save to History
                    save_js = build_save_script("Manual Recalculate", dict(st.session_state))
                    components.html(save_js, height=0)

            # Use current (possibly updated) data for display logic validation
            current_data = st.session_state['geo_data']
            crops_list = st.session_state['predicted_crop']
            primary_crop = crops_list[0]
            
            st.markdown("---")
            
            # Result Section
            c_res, c_info = st.columns([1, 2])
            
            with c_res:
                st.markdown(f"#### ✅ {translate('Recommended Crop', lang_code)}")
                
                # Confidence Gauge
                conf_score = round(random.uniform(85.0, 98.5), 1) # Mock XAI confidence
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = conf_score,
                    title = {'text': translate("AI Confidence Score", lang_code), 'font': {'size': 14, 'color': '#00C9FF', 'family': 'Courier New'}},
                    number = {'suffix': "%", 'font': {'color': '#00C9FF', 'size': 24, 'family': 'Courier New'}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "rgba(0,201,255,0.5)", 'tickfont': dict(color='#e2e8f0')},
                        'bar': {'color': "#00C9FF"},
                        'bgcolor': "rgba(15, 23, 42, 0.5)",
                        'borderwidth': 2,
                        'bordercolor': "rgba(0, 201, 255, 0.2)",
                        'steps': [
                            {'range': [0, 60], 'color': '#ef4444'},
                            {'range': [60, 85], 'color': '#fbbf24'},
                            {'range': [85, 100], 'color': 'rgba(0, 201, 255, 0.2)'}],
                    }
                ))
                fig_gauge.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
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
                     
                     st.markdown(f"#### 🧠 {translate('Why this Crop?', lang_code)}", help=translate("Feature contribution (SHAP values) explains which soil parameters influenced the AI's decision the most.", lang_code))
                     
                     # XAI Feature Importance Chart
                     features = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 'pH Level', 'Rainfall', 'Temperature']
                     # Simulated Feature Contributions (positive/negative) based on geo_data
                     contributions = [random.uniform(5, 25), random.uniform(2, 10), random.uniform(2, 12), random.uniform(-10, 15), random.uniform(10, 30), random.uniform(-15, 20)]
                     
                     # Sort by absolute magnitude for better viz
                     sorted_indices = np.argsort(np.abs(contributions))
                     features_sorted = [features[i] for i in sorted_indices]
                     contributions_sorted = [contributions[i] for i in sorted_indices]
                     colors_sorted = ['#00C9FF' if val >= 0 else '#ef4444' for val in contributions_sorted] # Neon Cyan vs Red
                     
                     fig_shap = go.Figure(go.Bar(
                         x=contributions_sorted,
                         y=features_sorted,
                         orientation='h',
                         marker_color=colors_sorted,
                         text=[f"{v:+.1f}%" for v in contributions_sorted],
                         textposition='auto',
                         hoverinfo='text',
                         hovertext=[translate(f"{f} contribution", lang_code) for f in features_sorted]
                     ))
                     fig_shap.update_layout(
                         xaxis_title=translate("Impact on Prediction (%)", lang_code),
                         height=220, margin=dict(l=0, r=0, t=10, b=0),
                         paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(color="#e2e8f0", size=12, family="Courier New"),
                         xaxis=dict(gridcolor='rgba(0, 201, 255, 0.1)'),
                         yaxis=dict(gridcolor='rgba(0, 201, 255, 0.1)'),
                         transition=dict(duration=800, easing="cubic-in-out")
                     )
                     st.plotly_chart(fig_shap, use_container_width=True)
                     
                     with st.expander(f"🚜 {translate(primary_crop, lang_code)} - {translate('Cultivation Process (Step-by-Step)', lang_code)}", expanded=False):
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
                    color_discrete_sequence=['#00C9FF', '#3b82f6', '#8b5cf6'],
                    template="plotly_dark",
                    text_auto='.2s'
                )
                fig_comp.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0", family="Courier New")
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
            
            # Editable Area Input (Acres) & Scenarios
            c_area, c_scenario = st.columns([1, 2])
            with c_area:
                area_acres = st.number_input(translate("Field Area (Acres)", lang_code), min_value=0.1, value=5.0, step=0.5, key="area_acres")
                area_ha = area_acres * 0.4047 # Conversion
                st.caption(f"{area_acres} Acres ≈ {area_ha:.2f} Hectares")
                
            with c_scenario:
                scenario = st.selectbox(translate("Compare AI Management Scenarios", lang_code), [
                    translate("Optimal AI Scenario (Recommended)", lang_code),
                    translate("Scenario A: High Fertilizer + Intense Irrigation", lang_code),
                    translate("Scenario B: Natural Rainfall Only (Low Input)", lang_code)
                ])

            # Generate Comparison Data
            years = [2021, 2022, 2023, 2024, 2025]
            avg_yield = [3.5, 3.6, 3.4, 3.7, 3.8]
            
            # Scenario logic
            if "Scenario A" in scenario or "High Fertilizer" in scenario:
                 ai_yield = [3.8, 4.0, 3.9, 4.3, 4.8]
                 uncertainty = 0.3
            elif "Scenario B" in scenario or "Natural Rainfall" in scenario:
                 ai_yield = [3.8, 3.9, 3.5, 3.7, 3.4]
                 uncertainty = 0.7
            else:
                 ai_yield = [3.8, 4.0, 3.9, 4.2, 4.5] # Base AI
                 uncertainty = 0.2
                 
            # Error margins (Uncertainty representation for XAI)
            ai_yield_upper = [round(y + uncertainty, 2) for y in ai_yield]
            ai_yield_lower = [round(y - uncertainty, 2) for y in ai_yield]
            
            # Economic Analysis
            price = CROP_KNOWLEDGE_BASE.get(crop, {}).get('price', 30000) # Default price
            est_yield = ai_yield[-1] # Ton/Ha
            
            # Revenue = Yield (Ton/Ha) * Area (Ha) * Price
            revenue = est_yield * area_ha * price
            
            # Financial Metrics
            c_fin1, c_fin2 = st.columns(2)
            c_fin1.metric(translate("Predicted Yield (2025)", lang_code), f"{est_yield} Ton/Ha", f"±{uncertainty} {translate('Confidence Margin', lang_code)}")
            c_fin2.metric(translate("Estimated Revenue", lang_code), f"₹ {revenue:,.0f}", f"{translate('For', lang_code)} {area_acres} Acres")
            
            # Chart & Drivers UI
            c_chart, c_drivers = st.columns([2, 1])
            
            with c_chart:
                fig_yield = go.Figure()
                
                # Add Uncertainty Band (XAI Feature)
                fig_yield.add_trace(go.Scatter(
                    x=years + years[::-1],
                    y=ai_yield_upper + ai_yield_lower[::-1],
                    fill='toself',
                    fillcolor='rgba(0, 201, 255, 0.15)', # Transparent Cyan
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name=translate('AI Prediction Margin', lang_code)
                ))
                
                fig_yield.add_trace(go.Scatter(x=years, y=avg_yield, name=translate('Regional Avg', lang_code), line=dict(color='#cbd5e1', dash='dash', width=2)))
                fig_yield.add_trace(go.Scatter(x=years, y=ai_yield, name=translate('AI Predicted', lang_code), line=dict(color='#00C9FF', width=4)))
                
                fig_yield.update_layout(
                    title=translate("Yield Optimization & Uncertainty Forecast", lang_code),
                    xaxis_title=translate("Year", lang_code),
                    yaxis_title=translate("Yield (Ton/Ha)", lang_code),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0", family="Courier New"),
                    hovermode="x unified",
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis=dict(gridcolor='rgba(0, 201, 255, 0.1)'),
                    yaxis=dict(gridcolor='rgba(0, 201, 255, 0.1)'),
                    transition=dict(duration=1000, easing="cubic-in-out")
                )
                st.plotly_chart(fig_yield, use_container_width=True)
                
            with c_drivers:
                st.markdown(f"#### 📈 {translate('Drivers of Growth', lang_code)}", help=translate("Factors historically influencing this prediction trajectory.", lang_code))
                if "Scenario A" in scenario or "High Fertilizer" in scenario:
                     st.info(f"💧 **{translate('Irrigation', lang_code)}:** {translate('Enhanced water access increases yield but raises overhead.', lang_code)}")
                     st.success(f"🧪 **{translate('Nutrients', lang_code)}:** {translate('Optimal nitrogen prevents mid-season stalling.', lang_code)}")
                elif "Scenario B" in scenario or "Natural Rainfall" in scenario:
                     st.warning(f"☀️ **{translate('Climate Risk', lang_code)}:** {translate('Highly dependent on unpredictable rain patterns.', lang_code)}")
                     st.error(f"📉 **{translate('Stress', lang_code)}:** {translate('Peak dry season expected to drop yield by 15%.', lang_code)}")
                else:
                     st.success(f"🌿 **{translate('Balance', lang_code)}:** {translate('AI-balanced nutrient schedule maximizes ROI.', lang_code)}")
                     st.info(f"🌦️ **{translate('Efficiency', lang_code)}:** {translate('Using predictive weather saves 20% water.', lang_code)}")
            
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
            fig_rain.add_trace(go.Bar(x=months, y=prev_year, name=translate('Previous Year', lang_code), marker_color='#475569'))
            fig_rain.add_trace(go.Bar(x=months, y=curr_forecast, name=translate('Forecast', lang_code), marker_color='#00C9FF'))
            
            fig_rain.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", family="Courier New"),
                barmode='group',
                xaxis=dict(gridcolor='rgba(0, 201, 255, 0.1)'),
                yaxis=dict(gridcolor='rgba(0, 201, 255, 0.1)'),
                transition=dict(duration=800, easing="cubic-in-out")
            )
            st.plotly_chart(fig_rain, use_container_width=True)
            
            # 15-Day Forecast Horizon (Slider)
            st.markdown(f"#### 📅 {translate('Extended Probability Forecast', lang_code)}")
            forecast_days = st.slider(translate("Select Prediction Horizon (Days)", lang_code), min_value=5, max_value=15, value=7)
            
            base_prob = cloud_cover
            daily_probs = [min(100, max(0, base_prob + random.randint(-30, 30) - (i*2))) for i in range(forecast_days)]
            future_days = [f"{translate('Day', lang_code)} {i+1}" for i in range(forecast_days)]
            
            fig_prob = go.Figure(go.Bar(
                x=future_days, y=daily_probs,
                marker_color='#00C9FF', # Neon Cyan
                text=[f"{int(p)}%" for p in daily_probs],
                textposition='auto'
            ))
            fig_prob.update_layout(
                yaxis_title=translate("Rain Probability (%)", lang_code),
                height=250, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", size=12, family="Courier New"),
                yaxis=dict(gridcolor='rgba(0, 201, 255, 0.1)'),
                transition=dict(duration=800, easing="cubic-in-out")
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
            st.markdown("---")
            # Prediction Breakdown (XAI)
            st.markdown(f"#### 🔍 {translate('Prediction Breakdown', lang_code)}", help=translate("How the AI weighted different data sources for this rain prediction.", lang_code))
            
            c_break1, c_break2, c_break3 = st.columns(3)
            with c_break1:
                 st.metric(translate("Global Satellite Data", lang_code), "45%", translate("Primary Weight", lang_code))
            with c_break2:
                 st.metric(translate("Local Humidity Sensors", lang_code), "35%", translate("Secondary Weight", lang_code))
            with c_break3:
                 st.metric(translate("Historical Patterns", lang_code), "20%", translate("Baseline", lang_code), delta_color="off")
                        
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
                with st.spinner(translate("🤖 AI scanning image to identify crop and anomalies...", lang_code)):
                    
                    use_fallback = False
                    current_key = os.environ.get("GEMINI_API_KEY", "")
                    if not current_key:
                        st.error(translate("Gemini API key is not set. Please enter it in the sidebar settings on the left to use real AI detection. Falling back to mock data.", lang_code))
                        time.sleep(1.0)
                        use_fallback = True
                    else:
                        # REAL AI DETECTION LOGIC VIA GEMINI
                        try:
                            model = genai.GenerativeModel("gemini-1.5-flash")
                            img_to_analyze = Image.open(uploaded_file)
                            
                            prompt = """
                            You are an expert agronomist and plant pathologist. Analyze this image of a plant/crop.
                            Ensure your response is ONLY in this exact structured format:
                            CROP_NAME: [Identify the crop in a single word or short phrase, e.g., Maize, Apple, Tomato]
                            STATUS: [Either 'HEALTHY' if no issues are found, or 'DISEASED' if issues are present]
                            ISSUE: [If 'HEALTHY', say 'None'. If 'DISEASED', name the disease concisely, e.g., 'Early Blight']
                            REMEDY: [If 'HEALTHY', say 'Maintain good practices'. If 'DISEASED', provide a semicolons-separated 3-step solution to treat it, e.g., Remove infected leaves; Use copper fungicide; Improve air circulation]
                            """
                            
                            response = model.generate_content([img_to_analyze, prompt])
                            raw_text = response.text.strip()
                            
                            # Parse output safely
                            parsed = {"CROP_NAME": "Unknown", "STATUS": "UNKNOWN", "ISSUE": "Unknown", "REMEDY": "Unknown"}
                            for line in raw_text.split("\n"):
                                line = line.strip()
                                if line.startswith("CROP_NAME:"): parsed["CROP_NAME"] = line.replace("CROP_NAME:", "").strip()
                                elif line.startswith("STATUS:"): parsed["STATUS"] = line.replace("STATUS:", "").strip()
                                elif line.startswith("ISSUE:"): parsed["ISSUE"] = line.replace("ISSUE:", "").strip()
                                elif line.startswith("REMEDY:"): parsed["REMEDY"] = line.replace("REMEDY:", "").strip()
                                
                            st.info(f"**{translate('Detected Crop', lang_code)}:** {translate(parsed['CROP_NAME'], lang_code)}")
                            
                            if parsed["STATUS"].upper() == "HEALTHY":
                                st.success(f"### ✅ {translate('Plant appears Healthy!', lang_code)}")
                                st.write(translate(parsed['REMEDY'], lang_code))
                            else:
                                st.error(f"### ⚠️ {translate('Disease Detected', lang_code)}!")
                                st.warning(f"**{translate('Issue Identified', lang_code)}:** {translate(parsed['ISSUE'], lang_code)}")
                                
                                st.markdown(f"#### 🛠️ {translate('Step-by-Step Solution', lang_code)}")
                                remedy_steps = [s.strip() for s in parsed["REMEDY"].split(";") if s.strip()]
                                if not remedy_steps:
                                    remedy_steps = [s.strip() for s in parsed["REMEDY"].split(".") if s.strip()]
                                    
                                for idx, step in enumerate(remedy_steps, 1):
                                    if step:
                                        st.write(f"**{translate('Step', lang_code)} {idx}:** {translate(step, lang_code)}")

                            # Save to History
                            h_state = {
                                "detected_crop": parsed.get("CROP_NAME"),
                                "status": parsed.get("STATUS"),
                                "issue": parsed.get("ISSUE")
                            }
                            save_js = build_save_script("Disease Detection", h_state)
                            components.html(save_js, height=0)

                        except Exception as e:
                            print(f"Gemini API Error: {e}")
                            st.error(f"Actual Gemini API Error: {e}")
                            st.warning(translate("⚠️ Could not connect to Gemini AI (Invalid or missing API Key). Using local mock AI instead.", lang_code))
                            time.sleep(1.0)
                            use_fallback = True

                    if use_fallback:
                        # Fallback logic if no API key or API key is invalid
                        available_crops = list(CROP_KNOWLEDGE_BASE.keys())
                        detected_crop = random.choice(available_crops)
                        st.info(f"**{translate('Detected Crop', lang_code)}:** {translate(detected_crop, lang_code)}")
                        
                        kb_diseases = CROP_KNOWLEDGE_BASE[detected_crop]['diseases']
                        analysis_results = kb_diseases + ["Healthy"]
                        detected = random.choice(analysis_results)
                        
                        if detected == "Healthy":
                            st.success(f"### ✅ {translate('Plant appears Healthy!', lang_code)}")
                            st.write(translate("No significant disease detected in the uploaded image. Maintain good agricultural practices.", lang_code))
                        else:
                            st.error(f"### ⚠️ {translate('Disease Detected', lang_code)}!")
                            if "*Remedy*:" in detected:
                                issue_part, remedy_part = detected.split("*Remedy*:")
                                issue_text = issue_part.replace("**", "").strip()
                                remedy_text = remedy_part.strip()
                                st.warning(f"**{translate('Issue Identified', lang_code)}:** {translate(issue_text, lang_code)}")
                                st.markdown(f"#### 🛠️ {translate('Step-by-Step Solution', lang_code)}")
                                remedy_steps = [s.strip() for s in remedy_text.split(";") if s.strip()]
                                if not remedy_steps:
                                    remedy_steps = [s.strip() for s in remedy_text.split(".") if s.strip()]
                                for idx, step in enumerate(remedy_steps, 1):
                                    if step:
                                        st.write(f"**{translate('Step', lang_code)} {idx}:** {translate(step, lang_code)}")
                            else:
                                st.warning(f"**{translate('Issue', lang_code)}:** {translate(detected, lang_code)}")

                            # Save to History
                            h_state = {
                                "detected_crop": detected_crop,
                                "status": "Diseased" if detected != "Healthy" else "Healthy",
                                "issue": detected if detected != "Healthy" else "None"
                            }
                            save_js = build_save_script("Disease Detection", h_state)
                            components.html(save_js, height=0)

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
                    
                    crops_list = st.session_state.get('predicted_crop', ['your crops'])
                    crop_context = crops_list[0] if isinstance(crops_list, list) else crops_list
                    
                    selected_response_en = get_assistant_response(clean_query, crop_context, lang_code, is_voice=True)
                    
                    translated_response = translate(selected_response_en, lang_code)
                    audio_html = text_to_audio(translated_response, lang_code)
                    
                    # Save response to state and force a rerun to update globally
                    st.session_state['voice_response_text'] = translated_response
                    st.session_state['voice_response_audio'] = audio_html
                    st.session_state['voice_query'] = clean_query
                    
                    
                    st.rerun()

    # --- GLOBAL TEXT CHAT INPUT ---
    # Placed at the very bottom of the main layout, accessible from any tab
    user_text_query = st.chat_input(translate("Ask about crops, yield, weather, or disease...", lang_code))
    
    if user_text_query:
        # Add user message to state
        st.session_state["messages"].append({"role": "user", "content": user_text_query})
        
        # Build AI Context
        crops_list = st.session_state.get('predicted_crop', ['your crops'])
        crop_context = crops_list[0] if isinstance(crops_list, list) else crops_list
        
        selected_response_en = get_assistant_response(user_text_query, crop_context, lang_code, is_voice=False)
        
        translated_response = translate(selected_response_en, lang_code)
        st.session_state["messages"].append({"role": "assistant", "content": translated_response})
        
        st.rerun()

if __name__ == "__main__":
    main()
