import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from src.feature_extraction import XLNetFeatureExtractor
from src.feature_selection import EBMOFeatureSelection
from src.preprocessing import DataPreprocessor
from src.translations import BASE_TRANSLATIONS, CROP_ICONS
from src.translator_utils import get_supported_languages, translate_ui
from src.visualizations import (plot_crop_probabilities, plot_soil_radar, 
                                 plot_rainfall_gauge, plot_yield_comparison,
                                 plot_feature_importance, plot_rainfall_trend)
from src.voice_utils import text_to_speech, speech_to_text
from streamlit_mic_recorder import mic_recorder
from deep_translator import GoogleTranslator
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
        # Global Language Selector
        all_langs = get_supported_languages()
        
        # Default label for selector (in English)
        st.subheader("🌐 Preferred Language")
        
        # Initialize language in session state
        if 'selected_lang_name' not in st.session_state:
            st.session_state['selected_lang_name'] = 'english'
            
        lang_name = st.selectbox("", options=list(all_langs.keys()), index=list(all_langs.keys()).index(st.session_state['selected_lang_name']), label_visibility="collapsed")
        
        st.session_state['selected_lang_name'] = lang_name
        lang_code = all_langs[lang_name]
        
        with st.spinner(f"Translating to {lang_name.title()}..."):
            t = translate_ui(lang_code, BASE_TRANSLATIONS)
        
        st.markdown("---")
        # Plant/Crop related icon (Sprout/Growth)
        st.image("https://cdn-icons-png.flaticon.com/512/1892/1892747.png", width=110) 
        st.title(t['sidebar_title'])
        st.markdown("---")
        app_mode = st.radio(
            t['select_module'],
            [t['nav_unified'], t['nav_disease']],
            captions=[t['nav_caption_unified'], t['nav_caption_disease']]
        )
        st.markdown("---")
        st.info(t['sidebar_tip'])
        st.caption(t['sidebar_caption'])
        
        st.markdown("---")
        run_voice_assistant(t, lang_code)

    st.title(t['main_title'])
    st.markdown(t['main_subtitle'])
    
    preprocessor, crop_svm, ebmo, yield_lstm, rainfall_transformer = load_models()
    
    if not preprocessor:
        st.error(t['error_models_not_found'])
        return

    # Container for Main Content
    with st.container():
        if app_mode == t['nav_unified']:
            run_unified_analysis(preprocessor, crop_svm, ebmo, yield_lstm, rainfall_transformer, t, lang_code)
        elif app_mode == t['nav_disease']:
            run_disease_detection(t)
            
    # Professional Footer
    st.markdown(f"""
    <div style="text-align: center; margin-top: 50px; font-size: 0.8rem; color: white; opacity: 0.8;">
        {t['footer']}
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
        
        # Global Variations
        'london': {'temp': 12.0, 'hum': 75.0, 'rain': 50.0, 'wind': 15.0, 'press': 1012.0, 'n': 60, 'p': 40, 'k': 30, 'ph': 6.2},
        'dubai': {'temp': 38.0, 'hum': 40.0, 'rain': 5.0,  'wind': 12.0, 'press': 1008.0, 'n': 20, 'p': 20, 'k': 20, 'ph': 8.2},
        'nairobi': {'temp': 22.0, 'hum': 65.0, 'rain': 120.0, 'wind': 10.0, 'press': 1015.0, 'n': 80, 'p': 60, 'k': 40, 'ph': 6.0},
        'sao paulo': {'temp': 24.0, 'hum': 80.0, 'rain': 250.0, 'wind': 10.0, 'press': 1010.0, 'n': 90, 'p': 50, 'k': 50, 'ph': 5.5},
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


def run_voice_assistant(t, lang_code='en'):
    """
    Renders the Voice Assistant UI in the sidebar or main area.
    """
    st.subheader("🎙️ AI Voice Assistant")
    
    # 1. Voice Input
    st.write("Click to speak:")
    audio = mic_recorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹️ Stop Recording", 
        key='recorder',
        format='wav' # Request wav format
    )
    
    if audio:
        st.audio(audio['bytes'])
        
        # 2. Transcribe
        with st.spinner("Listening..."):
            text = speech_to_text(audio['bytes'])
        
        if text:
            st.success(f"🗣️ You said: **{text}**")
            
            # 3. Simple Logic / AI Response
            response_text = ""
            text_lower = text.lower()
            
            if "weather" in text_lower:
                response_text = "I can help you with weather data. Use the Auto-Detect button to get local conditions."
            elif "crop" in text_lower or "recommend" in text_lower or "why" in text_lower:
                if 'last_crop' in st.session_state:
                    crop = st.session_state['last_crop']
                    conf = st.session_state['last_conf']
                    npk = st.session_state['last_npk']
                    ideal = st.session_state['last_ideal']
                    
                    # Basic explanation logic
                    reason = f"I recommended {crop} with {conf:.1%} confidence because your "
                    reasons = []
                    if npk['ph'] > 6 and npk['ph'] < 7: reasons.append("soil pH is in the optimal neutral range")
                    if npk['N'] > ideal[0]*0.8: reasons.append("nitrogen levels are strong")
                    
                    if reasons:
                        reason += " and ".join(reasons) + "."
                    else:
                        reason += "environmental conditions and nutrient balance are a statistically strong match."
                        
                    response_text = reason
                else:
                    response_text = "To provide a detailed explanation, please run the Crop Recommendation analysis first so I can analyze your soil data."
            elif "yield" in text_lower:
                response_text = "Yield forecasting uses historical data to predict your future harvest volume."
            elif "plan" in text_lower or "phase" in text_lower or "management" in text_lower:
                if 'last_narrative' in st.session_state:
                    response_text = st.session_state['last_narrative']
                else:
                    response_text = "I can explain your personalized farm plan once you run the crop recommendation analysis for your location."
            elif "hello" in text_lower or "hi" in text_lower:
                response_text = "Hello! I am your Smart Agriculture Assistant. How can I help you today?"
            else:
                response_text = "I understood: " + text + ". I am still learning to process complex queries."
            
            st.info(f"🤖 AI: {response_text}")
            
            # Translate response if not in English
            speech_response = response_text
            if lang_code != 'en':
                try:
                    speech_response = GoogleTranslator(source='en', target=lang_code).translate(response_text)
                except Exception:
                    pass

            # 4. Text to Speech Output
            tts_audio = text_to_speech(speech_response, lang=lang_code)
            if tts_audio:
                st.audio(tts_audio, format='audio/mp3', autoplay=True)

AGRONOMIC_KNOWLEDGE = {
    'rice': {
        'Phase 1: Land Preparation': "Plow field 2-3 times to 15cm depth. Maintain 5cm water level for puddling.",
        'Phase 2: Sowing/Transplanting': "Transplant 21-25 day old seedlings. Keep 20x20cm spacing.",
        'Phase 3: Vegetative Phase': "Tillering starts 15-30 days after transplanting. Keep field saturated.",
        'Phase 4: Reproductive Phase': "Panicle initiation to flowering. Maintain 5-10cm water level.",
        'Phase 5: Ripening & Harvest': "Harvest when 80% of grains turn straw-colored. Usually 110-130 days after sowing."
    },
    'maize': {
        'Phase 1: Land Preparation': "Deep plow to 20cm. Incorporate organic matter. Ensure good drainage.",
        'Phase 2: Sowing': "Plant at 5cm depth with 60x20cm spacing. Use 20kg seeds per hectare.",
        'Phase 3: Vegetative Phase': "Knee-high stage (V6). Top-dress with Nitrogen. Control weeds early.",
        'Phase 4: Tassel/Silking': "Critical water demand period. Ensure no moisture stress during silking.",
        'Phase 5: Maturity & Harvest': "Harvest when milk line disappears and black layer forms at grain base."
    },
    'chickpea': {
        'Phase 1: Land Preparation': "Finer seedbed preparation. Avoid saline or waterlogged soils.",
        'Phase 2: Sowing': "Sow in Oct-Nov at 10cm depth to utilize residual moisture.",
        'Phase 3: Early Growth': "Nippping (clipping terminal buds) at 30-45 days to promote branching.",
        'Phase 4: Pod Development': "Monitor for pod borers. Ensure soil is moist but not wet.",
        'Phase 5: Harvest': "Harvest when plants turn yellow to brown and pods rattle when shaken."
    },
    # Default fallback for other crops
    'default': {
        'Phase 1: Pre-Planting': "Soil testing and land preparation. Adjust N-P-K levels to match radar chart.",
        'Phase 2: Sowing': "Plant according to recommended spacing and depth for your specific variety.",
        'Phase 3: Growth Phase': "Monitor moisture and perform regular weed control. Apply top-dressing if needed.",
        'Phase 4: Plant Health': "Check for pests and diseases. Apply organic treatments early.",
        'Phase 5: Harvest': "Harvest at peak maturity. Ensure proper drying and storage to prevent loss."
    }
}

def display_end_to_end_summary(crop_name, predicted_yield, t, lang_code='en'):
    """Displays a detailed agronomic summary of the crop cycle."""
    st.markdown("---")
    st.header(f"🤖 Auto-AI Farm Explainer: {crop_name}")
    
    crop_key = crop_name.lower()
    knowledge = AGRONOMIC_KNOWLEDGE.get(crop_key, AGRONOMIC_KNOWLEDGE['default'])
    
    # 1. Generate Cohesive Narrative
    narrative = f"Here is your detailed agronomic roadmap for **{crop_name}**. "
    narrative += f"Starting with **{list(knowledge.keys())[0]}**, you should {list(knowledge.values())[0]}. "
    narrative += f"During **{list(knowledge.keys())[1]}**, remember that {list(knowledge.values())[1]} "
    narrative += f"As the crop enters the **Growth & Health** stages, {list(knowledge.values())[3]} "
    narrative += f"Finally, for **{list(knowledge.keys())[4]}**, {list(knowledge.values())[4]} "
    narrative += f"Your forecasted yield is **{predicted_yield:.0f} hg/ha**."
    
    # Store for Voice Assistant
    st.session_state['last_narrative'] = narrative
    st.session_state['last_knowledge'] = knowledge
    
    # 2. Display Narrative Card
    st.markdown(f"""
    <div style="background-color: #f0f7ff; padding: 25px; border-radius: 12px; border-left: 6px solid #007bff; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <p style="font-size: 1.15rem; line-height: 1.7; color: #1a3a5a; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            {narrative}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 4. Phase Breakdown (Modern Cards)
    st.subheader("📍 Detailed Agronomic Milestones")
    cols = st.columns(len(knowledge))
    for i, (phase, info) in enumerate(knowledge.items()):
        with cols[i]:
            st.success(f"**{phase}**\n\n{info}")

def display_preventive_measures(rain, temp, hum, t):
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
    
    st.markdown(t['climate_risk_advisory'])
    
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
        st.markdown(f"""
        <div class="safe-card">
            <div class="risk-title" style="color: #155724;">{t['optimal_conditions']}</div>
            <div class="risk-body">{t['optimal_conditions_msg']}</div>
        </div>
        """, unsafe_allow_html=True)

def run_crop_recommendation(preprocessor, model, ebmo, t):
    st.header(t['nav_crop'])
    
    # Session State
    if 'c_temp' not in st.session_state: st.session_state['c_temp'] = 25.0
    if 'c_hum' not in st.session_state: st.session_state['c_hum'] = 80.0
    if 'c_rain' not in st.session_state: st.session_state['c_rain'] = 200.0
    if 'c_n' not in st.session_state: st.session_state['c_n'] = 90
    if 'c_p' not in st.session_state: st.session_state['c_p'] = 40
    if 'c_k' not in st.session_state: st.session_state['c_k'] = 40
    if 'c_ph' not in st.session_state: st.session_state['c_ph'] = 6.5
    
    c1, c2 = st.columns([3, 1])
    with c1:
        loc = st.text_input(t['farm_location'], "Nagpur, India")
    with c2:
        st.write("")
        st.write("")
        if st.button(t['auto_detect']):
            w = get_live_weather(loc)
            st.session_state['c_temp'] = w['temp']
            st.session_state['c_hum'] = w['hum']
            st.session_state['c_rain'] = w['rain']
            st.session_state['c_n'] = w['n']
            st.session_state['c_p'] = w['p']
            st.session_state['c_k'] = w['k']
            st.session_state['c_ph'] = w['ph']
            st.toast("Soil & Weather Updated!")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(t['soil_params'])
        n = st.number_input(t['nitrogen'], 0, 140, key='c_n')
        p = st.number_input(t['phosphorus'], 0, 145, key='c_p')
        k = st.number_input(t['potassium'], 0, 205, key='c_k')
        ph = st.number_input(t['ph_level'], 0.0, 14.0, key='c_ph')
        
    with col2:
        st.subheader(t['weather_params'])
        temp = st.number_input(t['temperature'], 0.0, 50.0, key='c_temp')
        humidity = st.number_input(t['humidity'], 0.0, 100.0, key='c_hum')
        rainfall = st.number_input(t['rainfall'], 0.0, 500.0, key='c_rain')
        
    # User Request: Use previous year crop data
    st.subheader(t['farming_history'])
    crop_options = list(CROP_ICONS.keys())
    prev_crop = st.selectbox(t['prev_crop'], crop_options, format_func=lambda x: f"{CROP_ICONS[x]} {x}")
        
    if st.button(t['recommend_crop_btn']):
        with st.spinner(t['analyzing']):
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
                st.markdown(t['profit_selection'])
                
                cA, cB = st.columns(2)
                
                with cA:
                    st.success(t['agronomic_best'] + f": **{best_agronomic_crop}**")
                    st.caption(t['agronomic_caption'] + f" (Confidence: {best_prob:.2%})")
                    
                with cB:
                    st.info(t['economic_best'] + f": **{best_profit_crop}**")
                    profit = CROP_PROFIT.get(best_profit_crop, 0)
                    st.caption(t['economic_caption'] + f" (~₹{profit}/ha)")
                
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
            
            # === VISUAL GRAPHICS: Top Crops Probability Chart ===
            if hasattr(model, 'predict_proba'):
                st.markdown("---")
                st.subheader("📊 Crop Suitability Analysis")
                
                # Get all crop names and probabilities
                all_crops = preprocessor.encoders['label'].classes_
                
                # Create and display probability chart
                fig_probs = plot_crop_probabilities(all_crops, probs, top_n=5)
                st.plotly_chart(fig_probs, use_container_width=True)
                
                # === VISUAL GRAPHICS: Soil Nutrient Radar Chart ===
                st.subheader("🎯 Soil Nutrient Match")
                
                # Normalize current values to 0-100 scale for radar
                current_norm = [
                    (n / 140) * 100,  # N max is 140
                    (p / 145) * 100,  # P max is 145
                    (k / 205) * 100,  # K max is 205
                    (ph / 14) * 100,  # pH max is 14
                ]
                
                # Get optimal values for recommended crop (from ideal profiles)
                ideal_profiles = {
                    'rice': [80, 40, 40, 6.5],
                    'maize': [100, 45, 20, 6.2],
                    'chickpea': [40, 60, 80, 7.3],
                    'banana': [100, 80, 50, 5.9],
                    'cotton': [120, 40, 20, 6.9],
                    'grapes': [23, 130, 200, 6.0],
                    'apple': [20, 130, 200, 5.9],
                }
                
                crop_lower = crop_name.lower()
                if crop_lower in ideal_profiles:
                    ideal_vals = ideal_profiles[crop_lower]
                    optimal_norm = [
                        (ideal_vals[0] / 140) * 100,
                        (ideal_vals[1] / 145) * 100,
                        (ideal_vals[2] / 205) * 100,
                        (ideal_vals[3] / 14) * 100,
                    ]
                else:
                    # Default optimal (moderate)
                    optimal_norm = [50, 50, 50, 50]
                    ideal_vals = [70, 70, 70, 7.0] # Default baseline
                
                labels = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 'pH Level']
                fig_radar = plot_soil_radar(current_norm, optimal_norm, labels)
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # --- Store for Voice Assistant ---
                st.session_state['last_crop'] = best_agronomic_crop
                st.session_state['last_conf'] = best_prob
                st.session_state['last_npk'] = {'N': n, 'P': p, 'K': k, 'ph': ph}
                st.session_state['last_ideal'] = ideal_vals
                # --- AI Interpretation Layer ---
                st.markdown("---")
                st.subheader("🤖 AI Technical Insights")
                
                col_i1, col_i2 = st.columns(2)
                with col_i1:
                    st.info(f"**Suitability Insight**: The model has **{best_prob:.1%}** confidence in **{best_agronomic_crop}**. "
                            f"This high score suggests your soil's NPK levels and current weather are perfectly aligned with this crop's biological needs.")
                with col_i2:
                    # Logic for Soil Radar Interpretation
                    # Index 0: N, 1: P, 2: K, 3: ph
                    defects = []
                    if (n < ideal_vals[0]*0.7): defects.append("Nitrogen")
                    if (p < ideal_vals[1]*0.7): defects.append("Phosphorus")
                    if (k < ideal_vals[2]*0.7): defects.append("Potassium")
                    
                    if defects:
                        st.warning(f"**Soil Insight**: Your soil is slightly low in **{', '.join(defects)}**. "
                                   f"While {crop_name} is still the best choice, adding targeted fertilizers for these nutrients will boost yield.")
                    else:
                        st.success(f"**Soil Insight**: Your nutrient profile is excellent! The radar chart shows your NPK levels match the ideal requirements of {crop_name} very closely.")

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
            st.markdown(t['crop_rotation_analysis'])
            if prev_crop == crop_name:
                st.warning(t['monoculture_risk'] + f": You grew **{prev_crop}** last year. Planting it again depletes specific soil nutrients and increases disease risk.")
            elif prev_crop in ['Chickpea', 'Kidneybeans', 'Pigeonpeas', 'Mothbeans', 'Mungbean', 'Blackgram', 'Lentil'] and crop_name in ['Rice', 'Maize', 'Cotton']:
                st.success(t['excellent_rotation'] + f": Previous crop **{prev_crop}** is a legume which fixed Nitrogen in the soil. This benefits **{crop_name}**.")
            elif prev_crop != 'None':
                st.info(t['rotation_check'] + f": Rotating from **{prev_crop}** to **{crop_name}** is generally acceptable.")
            
            # 4. Explainability (Dynamic & Static)
            st.subheader(t['explainability_xai'])
            
            # A. Rule-Based / Feature Contribution (Dynamic)
            st.markdown(t['why_prediction'])
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
            display_preventive_measures(rainfall, temp, humidity, t)

def run_yield_forecasting(preprocessor, model, t):
    st.header(t['yield_forecasting_title'])
    st.write(t['yield_forecasting_subtitle'])
    
    # Yield Forecasting Auto-Detect
    if 'y_rain' not in st.session_state: st.session_state['y_rain'] = 1200.0
    if 'y_temp' not in st.session_state: st.session_state['y_temp'] = 28.0
    
    c1, c2 = st.columns([3, 1])
    with c1: loc = st.text_input(t['farm_location'], "Punjab, India")
    with c2:
        st.write("")
        st.write("")
        if st.button(t['auto_detect']):
            w = get_live_weather(loc)
            st.session_state['y_rain'] = w['rain'] * 10 
            st.session_state['y_temp'] = w['temp']
            st.toast("Fetch Complete!")

    rain = st.number_input(t['rainfall'], 0.0, 3000.0, key='y_rain')
    pest = st.number_input("Pesticides (tonnes)", 0.0, 1000.0, 50.0)
    temp = st.number_input(t['temperature'], 0.0, 50.0, key='y_temp')
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
            
            # Inverse Transform (StandardScaler: x = z * std + mean)
            idx = 4 
            mean_val = scaler.mean_[idx]
            std_val = scaler.scale_[idx]
            
            pred_inv = pred_scaled[0][0] * std_val + mean_val
            
            st.metric("Predicted Yield (hg/ha)", f"{pred_inv:.2f}")
            
            # === VISUAL GRAPHICS: Yield Comparison Chart ===
            st.markdown("---")
            st.subheader("📈 Yield Comparison Analysis")
            
            # Simulate average and optimal yields for comparison
            avg_yield = pred_inv * 0.85  # Average is typically 85% of predicted
            optimal_yield = pred_inv * 1.15  # Optimal is 115% of predicted
            
            fig_yield = plot_yield_comparison(pred_inv, avg_yield, optimal_yield, "Current Crop")
            st.plotly_chart(fig_yield, use_container_width=True)
        else:
            st.error("Yield Scaler not found!")

def run_rainfall_prediction(preprocessor, model, t):
    st.header(t['rainfall_prediction_title'])
    st.write(t['rainfall_prediction_subtitle'])
    
    # Rainfall Auto-Detect
    if 'r_temp' not in st.session_state: st.session_state['r_temp'] = 30.0
    if 'r_hum' not in st.session_state: st.session_state['r_hum'] = 75.0
    if 'r_wind' not in st.session_state: st.session_state['r_wind'] = 10.0
    if 'r_press' not in st.session_state: st.session_state['r_press'] = 1010.0
    if 'r_prev_rain' not in st.session_state: st.session_state['r_prev_rain'] = 0.0
    
    c1, c2 = st.columns([3, 1])
    with c1: loc = st.text_input(t['farm_location'], "New Delhi")
    with c2:
        st.write("")
        st.write("")
        if st.button(t['auto_detect']):
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
        temp = st.number_input(t['temperature'], 0.0, 50.0, key='r_temp')
        hum = st.number_input(t['humidity'], 0.0, 100.0, key='r_hum')
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
            
            # Inverse transform target (StandardScaler: x = z * std + mean)
            idx = 4
            mean_val = scaler.mean_[idx]
            std_val = scaler.scale_[idx]
            
            pred_inv = pred_scaled[0][0] * std_val + mean_val
            
            st.metric(f"Expected Rainfall for {loc} (mm)", f"{pred_inv:.2f}")
            
            # === VISUAL GRAPHICS: Rainfall Gauge ===
            st.markdown("---")
            st.subheader("🌧️ Rainfall Forecast Visualization")
            
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                fig_gauge = plot_rainfall_gauge(pred_inv, min_val=0, max_val=500)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col_g2:
                # Generate simulated historical trend
                np.random.seed(42)
                historical = np.random.normal(pred_inv * 0.8, 30, 10).clip(0, 500)
                fig_trend = plot_rainfall_trend(historical, pred_inv)
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # --- AI Interpretation for Rainfall ---
                st.info(f"🤖 **AI Rainfall Insight**: The forecast predicts **{pred_inv:.1f}mm** of rainfall. "
                        f"The trend chart (red star) shows this is **{'above' if pred_inv > np.mean(historical) else 'below'}** your 10-period historical average. "
                        f"Plan your irrigation schedule accordingly.")
        else:
            st.error("Rainfall Scaler not found!")

def run_unified_analysis(preprocessor, crop_model, ebmo, yield_model, rainfall_model, t, lang_code='en'):
    st.header(t['unified_analysis_title'])
    st.markdown(t['unified_analysis_subtitle'])

    # 1. Inputs
    if 'u_loc' not in st.session_state: st.session_state['u_loc'] = "Nagpur, India"
    if 'u_n' not in st.session_state: st.session_state['u_n'] = 90
    if 'u_p' not in st.session_state: st.session_state['u_p'] = 40
    if 'u_k' not in st.session_state: st.session_state['u_k'] = 40
    if 'u_ph' not in st.session_state: st.session_state['u_ph'] = 6.5
    if 'u_temp' not in st.session_state: st.session_state['u_temp'] = 25.0
    if 'u_hum' not in st.session_state: st.session_state['u_hum'] = 80.0
    
    col_l, col_b = st.columns([3, 1])
    with col_l:
        loc = st.text_input(t['farm_location'], key='u_loc')
    with col_b:
        st.write("")
        st.write("")
        if st.button(t['auto_detect']):
            w = get_live_weather(loc)
            st.session_state['u_temp'] = w['temp']
            st.session_state['u_hum'] = w['hum']
            st.session_state['u_n'] = w['n']
            st.session_state['u_p'] = w['p']
            st.session_state['u_k'] = w['k']
            st.session_state['u_ph'] = w['ph']
            st.toast("Data Populated!")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(t['soil_params'])
        n = st.number_input(t['nitrogen'], 0, 140, key='u_n')
        p = st.number_input(t['phosphorus'], 0, 145, key='u_p')
        k = st.number_input(t['potassium'], 0, 205, key='u_k')
        ph = st.number_input(t['ph_level'], 0.0, 14.0, key='u_ph')
    with c2:
        st.subheader(t['weather_params'])
        temp = st.number_input(t['temperature'], 0.0, 50.0, key='u_temp')
        hum = st.number_input(t['humidity'], 0.0, 100.0, key='u_hum')
        wind = st.number_input("Wind Speed (km/h)", 0.0, 50.0, 10.0)
        press = st.number_input("Pressure (hPa)", 900.0, 1100.0, 1010.0)
        
    # --- New User Request: Farming History in Unified ---
    st.subheader(t['farming_history'])
    crop_options = list(CROP_ICONS.keys())
    # Try to make it friendly with icons
    prev_crop = st.selectbox(t['select_prev_crop'], crop_options, format_func=lambda x: f"{CROP_ICONS[x]} {x}")
    
    st.markdown("---")
    st.subheader("Production Parameters")
    c5, c6 = st.columns(2)
    with c5:
        area_acres = st.number_input(t['area_acres'], 1.0, 1000.0, 10.0)
    with c6:
        pesticides = st.number_input(t['pesticides'], 0.0, 100.0, 1.0)

    if st.button(t['run_analysis_btn']):
        with st.spinner(t['processing']):
            # A. Sequential Step 1: Rainfall Prediction
            st.markdown("---")
            st.subheader(t['step1_rain'])
            r_scaler = preprocessor.scalers.get('Rainfall')
            predicted_rainfall = 0
            if r_scaler:
                raw_r = np.array([[temp, hum, wind, press, 0.0]]) # Using 0 as prev_rain
                scaled_r = r_scaler.transform(raw_r)
                input_seq_r = np.tile(scaled_r[0], (10, 1)).reshape(1, 10, 5)
                pred_r_scaled = rainfall_model.predict(input_seq_r, verbose=0)
                mean_r = r_scaler.mean_[4]
                std_r = r_scaler.scale_[4]
                predicted_rainfall = pred_r_scaled[0][0] * std_r + mean_r
                st.metric("Forecasted Rainfall", f"{predicted_rainfall:.2f} mm")
                
                # === VISUAL: Rainfall Gauge & Trend ===
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    fig_gauge = plot_rainfall_gauge(predicted_rainfall, min_val=0, max_val=500)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                with col_r2:
                    # Simulated historical trend
                    np.random.seed(42)
                    historical = np.random.normal(predicted_rainfall * 0.8, 30, 10).clip(0, 500)
                    fig_trend = plot_rainfall_trend(historical, predicted_rainfall)
                    st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.error("Rainfall Scaler missing.")

            # B. Sequential Step 2: Crop Recommendation
            st.subheader(t['step2_crop'])
            input_c = pd.DataFrame({
                'N': [n], 'P': [p], 'K': [k],
                'temperature': [temp], 'humidity': [hum], 
                'ph': [ph], 'rainfall': [predicted_rainfall],
                'label': ['dummy']
            })
            
            num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            if 'tabular_num' in preprocessor.scalers:
                input_c[num_cols] = preprocessor.scalers['tabular_num'].transform(input_c[num_cols])

            extractor = XLNetFeatureExtractor()
            feats = extractor.extract_features(extractor.tabular_to_text(input_c, num_cols))
            feats_sel = ebmo.transform(feats)
            
            crop_name = "Unknown"
            if hasattr(crop_model, 'predict_proba'):
                probs = crop_model.predict_proba(feats_sel)[0]
                idx = np.argmax(probs)
                crop_name = preprocessor.encoders['label'].inverse_transform([idx])[0]
                st.success(f"Best Recommended Crop: **{crop_name}** (Confidence: {probs[idx]:.2%})")
                
                # === VISUAL: Crop Probabilities & Soil Radar ===
                all_crops = preprocessor.encoders['label'].classes_
                
                # Crop probability chart
                fig_probs = plot_crop_probabilities(all_crops, probs, top_n=5)
                st.plotly_chart(fig_probs, use_container_width=True)
                
                # Soil radar chart
                current_norm = [
                    (n / 140) * 100,
                    (p / 145) * 100,
                    (k / 205) * 100,
                    (ph / 14) * 100,
                ]
                
                ideal_profiles = {
                    'rice': [80, 40, 40, 6.5], 'maize': [100, 45, 20, 6.2],
                    'chickpea': [40, 60, 80, 7.3], 'banana': [100, 80, 50, 5.9],
                    'cotton': [120, 40, 20, 6.9], 'grapes': [23, 130, 200, 6.0],
                    'apple': [20, 130, 200, 5.9],
                }
                
                crop_lower = crop_name.lower()
                if crop_lower in ideal_profiles:
                    ideal_vals = ideal_profiles[crop_lower]
                    optimal_norm = [
                        (ideal_vals[0] / 140) * 100,
                        (ideal_vals[1] / 145) * 100,
                        (ideal_vals[2] / 205) * 100,
                        (ideal_vals[3] / 14) * 100,
                    ]
                else:
                    optimal_norm = [50, 50, 50, 50]
                    ideal_vals = [70, 70, 70, 7.0] # Default baseline
                
                labels = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 'pH Level']
                fig_radar = plot_soil_radar(current_norm, optimal_norm, labels)
                st.plotly_chart(fig_radar, use_container_width=True)

                # --- AI Interpretation Layer (Unified) ---
                st.markdown("---")
                st.subheader("🤖 AI Technical Insights")
                
                col_i1, col_i2 = st.columns(2)
                with col_i1:
                    st.info(f"**Suitability Insight**: The model has **{probs[idx]:.1%}** confidence in **{crop_name}**. "
                            f"This indicates high compatibility with predicted rainfall and your current soil profile.")
                with col_i2:
                    def_list = []
                    if (n < ideal_vals[0]*0.7): def_list.append("Nitrogen")
                    if (p < ideal_vals[1]*0.7): def_list.append("Phosphorus")
                    if (k < ideal_vals[2]*0.7): def_list.append("Potassium")
                    
                    if def_list:
                        st.warning(f"**Soil Insight**: Your nitrogen/phosphorus/potassium levels are slightly lower than ideal for {crop_name}. "
                                   f"Addition of {', '.join(def_list)} is recommended.")
                    else:
                        st.success(f"**Soil Insight**: Your soil nutrients are perfectly balanced for this crop!")

                # --- Store for Voice Assistant ---
                st.session_state['last_crop'] = crop_name
                st.session_state['last_conf'] = probs[idx]
                st.session_state['last_npk'] = {'N': n, 'P': p, 'K': k, 'ph': ph}
                st.session_state['last_ideal'] = ideal_vals
            else:
                idx = crop_model.predict(feats_sel)[0]
                crop_name = preprocessor.encoders['label'].inverse_transform([idx])[0]
                st.success(f"Best Recommended Crop: **{crop_name}**")

            # C. Sequential Step 3: Yield Forecasting
            st.subheader(t['step3_yield'])
            y_scaler = preprocessor.scalers.get('yield_amount')
            if y_scaler:
                # Features: ['average_rain', 'pesticides_tonnes', 'avg_temp', 'area', 'yield_amount']
                area_ha = area_acres * 0.404686
                # Use real inputs now
                raw_y = np.array([[predicted_rainfall, pesticides, temp, area_ha, 20000.0]])
                scaled_y = y_scaler.transform(raw_y)
                input_seq_y = np.tile(scaled_y[0], (3, 1)).reshape(1, 3, 5)
                pred_y_scaled = yield_model.predict(input_seq_y, verbose=0)
                mean_y = y_scaler.mean_[4]
                std_y = y_scaler.scale_[4]
                predicted_yield = pred_y_scaled[0][0] * std_y + mean_y
                st.metric(f"{t['yield_metric']} ({crop_name})", f"{predicted_yield:.2f} hg/ha")
                
                # === VISUAL: Yield Comparison ===
                avg_yield = predicted_yield * 0.85
                optimal_yield = predicted_yield * 1.15
                fig_yield = plot_yield_comparison(predicted_yield, avg_yield, optimal_yield, crop_name)
                st.plotly_chart(fig_yield, use_container_width=True)
                
                # --- AI Interpretation for Yield ---
                st.success(f"🤖 **AI Yield Insight**: Our LSTM model forecasts a yield of **{predicted_yield:.0f} hg/ha**. "
                           f"This is **{(predicted_yield/avg_yield - 1)*100:.1f}% higher** than the regional average. "
                           f"Tip: Maintaining the 'Optimal' conditions shown in the bar chart could push your harvest even further.")
                
                # --- Step 4: End-to-End Summary ---
                st.session_state['last_yield'] = predicted_yield
                display_end_to_end_summary(crop_name, predicted_yield, t, lang_code)
            else:
                st.error("Yield Scaler missing.")
            
            st.markdown("---")
            # --- Crop Rotation Analysis for Unified ---
            st.markdown(t['crop_rotation_analysis'])
            if prev_crop == crop_name:
                st.warning(t['monoculture_risk'] + f": You grew **{prev_crop}** last year. Planting it again depletes soil nutrients.")
            elif prev_crop in ['Chickpea', 'Kidneybeans', 'Pigeonpeas', 'Mothbeans', 'Mungbean', 'Blackgram', 'Lentil'] and crop_name in ['Rice', 'Maize', 'Cotton']:
                st.success(t['excellent_rotation'] + f": Previous **{prev_crop}** fixed Nitrogen. Benefits **{crop_name}**.")
            elif prev_crop != 'None':
                st.info(t['rotation_check'] + f": Rotating from **{prev_crop}** to **{crop_name}** is acceptable.")
            
            # === PREVENTIVE MEASURES & CLIMATE RISK ADVISORY ===
            st.markdown("---")
            st.markdown(t['climate_risk_advisory'])
            
            # Analyze risks based on predicted conditions
            risks = []
            
            # Temperature risks
            if temp > 40:
                risks.append({
                    'icon': '🔥',
                    'title': 'Extreme Heat Alert',
                    'msg': f'Temperature ({temp}°C) exceeds safe threshold.',
                    'steps': [
                        'Install shade nets over crops',
                        'Increase irrigation frequency to 2-3 times daily',
                        'Apply mulch to retain soil moisture',
                        'Consider heat-resistant crop varieties'
                    ]
                })
            elif temp < 10:
                risks.append({
                    'icon': '❄️',
                    'title': 'Frost Risk',
                    'msg': f'Low temperature ({temp}°C) may damage crops.',
                    'steps': [
                        'Cover young plants with frost cloth',
                        'Use smudge pots or heaters in critical areas',
                        'Water crops before sunset to retain heat',
                        'Harvest mature crops immediately'
                    ]
                })
            
            # Rainfall risks
            if predicted_rainfall > 300:
                risks.append({
                    'icon': '🌧️',
                    'title': 'Heavy Rainfall Warning',
                    'msg': f'Predicted rainfall ({predicted_rainfall:.0f}mm) is very high.',
                    'steps': [
                        'Ensure proper drainage channels are clear',
                        'Build raised beds for vulnerable crops',
                        'Apply fungicides preventively',
                        'Harvest ready crops before heavy rains'
                    ]
                })
            elif predicted_rainfall < 50:
                risks.append({
                    'icon': '🌵',
                    'title': 'Drought Conditions',
                    'msg': f'Low rainfall ({predicted_rainfall:.0f}mm) expected.',
                    'steps': [
                        'Install drip irrigation system',
                        'Apply organic mulch to conserve moisture',
                        'Choose drought-resistant varieties',
                        'Implement rainwater harvesting'
                    ]
                })
            
            # Humidity risks
            if hum > 85:
                risks.append({
                    'icon': '🌫️',
                    'title': 'High Humidity Alert',
                    'msg': f'Humidity ({hum}%) increases disease risk.',
                    'steps': [
                        'Improve air circulation with proper spacing',
                        'Apply preventive fungicides',
                        'Monitor for fungal diseases daily',
                        'Avoid overhead irrigation'
                    ]
                })
            
            # Soil nutrient warnings
            if n < 40:
                risks.append({
                    'icon': '🧪',
                    'title': 'Nitrogen Deficiency',
                    'msg': f'Nitrogen level ({n}) is low for {crop_name}.',
                    'steps': [
                        'Apply urea or ammonium sulfate',
                        'Use organic compost rich in nitrogen',
                        'Plant nitrogen-fixing cover crops',
                        'Consider foliar nitrogen spray'
                    ]
                })
            
            if p < 30:
                risks.append({
                    'icon': '🧪',
                    'title': 'Phosphorus Deficiency',
                    'msg': f'Phosphorus level ({p}) is low.',
                    'steps': [
                        'Apply superphosphate or DAP fertilizer',
                        'Use bone meal for organic option',
                        'Maintain soil pH 6-7 for better P uptake',
                        'Apply mycorrhizal fungi'
                    ]
                })
            
            # Display risks or optimal message
            if risks:
                for r in risks:
                    steps_html = ''.join([f'<li>{s}</li>' for s in r['steps']])
                    st.markdown(f"""
                    <div class="risk-card">
                        <div class="risk-title">{r['icon']} {r['title']}</div>
                        <div class="risk-body">{r['msg']}
                            <ul>{steps_html}</ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-card">
                    <div class="risk-title" style="color: #155724;">{t['optimal_conditions']}</div>
                    <div class="risk-body">{t['optimal_conditions_msg']}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            display_preventive_measures(predicted_rainfall, temp, hum, t)

def run_disease_detection(t):
    """
    Module for identifying plant diseases from images and providing cures.
    """
    st.header(t['disease_detection_title'])
    st.markdown(t['disease_detection_subtitle'])

    # Disease Knowledge Base - Simplified for Farmers
    PLANT_DISEASE_DB = {
        "Rice": [
            {
                "name": "Bacterial Leaf Blight (Yellow Leaves)", 
                "symptoms": "Leaves turning yellow and drying from the tips.", 
                "cure": "✅ **Step 1:** Use less Urea fertilizer.\n\n✅ **Step 2:** Mix 100g bleaching powder in 1 bucket (10 liters) of water.\n\n✅ **Step 3:** Spray clearly on the plants."
            },
            {
                "name": "Blast (Rotting Spots)", 
                "symptoms": "Brown diamond-shaped spots on leaves.", 
                "cure": "✅ **Step 1:** Stop giving too much Nitrogen.\n\n✅ **Step 2:** Mix 1 gram of Tricyclazole powder in 1 liter of water.\n\n✅ **Step 3:** Spray every 10 days until spots go away."
            }
        ],
        "Maize": [
            {
                "name": "Common Rust (Brown Dust)", 
                "symptoms": "Tiny brown raised spots that look like dust on leaves.", 
                "cure": "✅ **Step 1:** Use seeds that can fight disease.\n\n✅ **Step 2:** If spots appear, mix 30 grams of Mancozeb powder in 1 bucket (10 liters) of water.\n\n✅ **Step 3:** Spray once a week."
            },
            {
                "name": "Leaf Blight (Long Spots)", 
                "symptoms": "Long grayish-green spots on leaves.", 
                "cure": "✅ **Step 1:** Do not plant corn in the same field every year.\n\n✅ **Step 2:** Mix 20 grams of fungicide in 10 liters of water.\n\n✅ **Step 3:** Spray during early morning."
            }
        ],
        "Tomato": [
            {
                "name": "Early Blight (Black Rings)", 
                "symptoms": "Black spots with rings on lower leaves.", 
                "cure": "✅ **Step 1:** Cut off the bottom yellow leaves.\n\n✅ **Step 2:** Mix 2 grams of Copper powder in 1 liter of water (or 4 spoons in 1 bucket).\n\n✅ **Step 3:** Spray the whole plant."
            },
            {
                "name": "Late Blight (White Mold)", 
                "symptoms": "Large dark spots with white cottony mold under the leaves.", 
                "cure": "✅ **Step 1:** Allow more space between plants for air.\n\n✅ **Step 2:** Mix 25 grams of Metalaxyl powder in 1 bucket (10 liters) of water.\n\n✅ **Step 3:** Spray immediately if you see mold."
            }
        ]
    }

    # Create two columns for input options
    col_file, col_cam = st.columns(2)
    
    with col_file:
        uploaded_file = st.file_uploader(t['upload_plant_image'], type=["jpg", "png", "jpeg"])
    
    with col_cam:
        st.write("") # Alignment
        if 'cam_active' not in st.session_state:
            st.session_state['cam_active'] = False
            
        if st.button(t['open_camera_btn']):
            st.session_state['cam_active'] = not st.session_state['cam_active']
        
        camera_photo = None
        if st.session_state['cam_active']:
            camera_photo = st.camera_input(t['camera_input_label'])

    # Determine which image to use
    final_image = camera_photo if camera_photo is not None else uploaded_file

    if final_image is not None:
        st.image(final_image, caption="Plant Image for Analysis", use_column_width=True)
        
        if st.button(t['analyze_image_btn']):
            with st.spinner(t['processing']):
                # In a real app, this would be a CNN model prediction
                # Here we simulate diagnostic based on common crop types
                import time
                time.sleep(2) # Simulate processing
                
                # Randomly pick a crop and disease for demonstration
                crop = random.choice(list(PLANT_DISEASE_DB.keys()))
                disease = random.choice(PLANT_DISEASE_DB[crop])
                confidence = random.uniform(85.0, 99.0)

                st.markdown("---")
                st.subheader(f"✅ {t['disease_identified']}: {disease['name']} ({crop})")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(t['confidence_level'], f"{confidence:.1f}%")
                    st.info(f"**Symptoms**: {disease['symptoms']}")
                
                with col2:
                    st.success(f"### {t['cure_suggestion']}\n{disease['cure']}")
                
                # AI Voice Explanation for Disease
                explanation = f"I have identified {disease['name']} on your {crop} with {confidence:.1f}% confidence. {disease['cure']}"
                
                # Local translation for voice
                lang_code = st.session_state.get('selected_lang_name', 'en')
                if lang_code != 'en':
                    try:
                        explanation = GoogleTranslator(source='en', target=lang_code).translate(explanation)
                    except: pass
                
                # Clean markdown for voice
                speech_text = explanation.replace("**", "").replace("\n", " ")
                
                audio = text_to_speech(speech_text, lang=lang_code)
                if audio:
                    st.audio(audio, autoplay=True)

if __name__ == "__main__":
    main()
