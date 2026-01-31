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
    # Sidebar Language Selector
    with st.sidebar:
        st.header("🗣️ Language / भाषा")
        lang_map = {'English': 'English', 'Hindi': 'Hindi', 'Marathi': 'Marathi', 'Telugu': 'Telugu'}
        selected_lang_key = st.selectbox("Select Language", list(TRANSLATIONS.keys()), label_visibility="collapsed")
    
        # Plant/Crop related icon (Sprout/Growth)
        st.image("https://cdn-icons-png.flaticon.com/512/1892/1892747.png", width=110) 
        st.title(get_text('title', selected_lang_key))
        st.markdown("---")
        app_mode = st.radio(get_text('sidebar_title', selected_lang_key), 
            [get_text('crop_rec', selected_lang_key), 
             get_text('yield_fore', selected_lang_key), 
             get_text('rain_pred', selected_lang_key)],
            captions=["Get best crop & preventive advice", "Estimate harvest volume", "Forecast precipitation"]
        )
        st.markdown("---")
        st.info("💡 **Tip**: Use Auto-Detect for local weather!")
        st.caption("v1.3.0 | XAI Enabled")

    st.title(get_text('title', selected_lang_key))
    st.markdown("### Intelligent Decisions for Better Farming")
    
    preprocessor, crop_svm, ebmo, yield_lstm, rainfall_transformer = load_models()
    
    if not preprocessor:
        st.error("⚠️ Models not found! Please run `python main.py` first.")
        return

    # Container for Main Content
    with st.container():
        if app_mode == get_text('crop_rec', selected_lang_key):
             # Pass lang to Crop Rec (already updated to accept it)
             run_crop_recommendation(preprocessor, crop_svm, ebmo, selected_lang_key)
            
        elif app_mode == get_text('yield_fore', selected_lang_key):
             # Pass lang (will update signature next)
             run_yield_forecasting(preprocessor, yield_lstm, selected_lang_key)
            
        elif app_mode == get_text('rain_pred', selected_lang_key):
             # Pass lang (will update signature next)
             run_rainfall_prediction(preprocessor, rainfall_transformer, selected_lang_key)
            
    # Professional Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; font-size: 0.8rem; color: white; opacity: 0.8;">
        Smart Agri XAI © 2025 | Developed for Sustainable Farming 🌾
    </div>
    """, unsafe_allow_html=True)

import zlib
import random
from deep_translator import GoogleTranslator

# --- Language Dictionary (Static for Speed) ---
TRANSLATIONS = {
    'English': {
        'title': "🌱 Smart Agri XAI Assistant",
        'sidebar_title': "Navigation",
        'model_select': "Select Module",
        'crop_rec': "Crop Recommendation System",
        'yield_fore': "Yield Forecasting",
        'rain_pred': "Rainfall Prediction",
        'loc_label': "📍 Farm Location",
        'auto_detect': "🔄 Auto-Detect",
        'soil_params': "Soil Parameters",
        'weather_params': "Weather Parameters",
        'humidity_label': "Humidity (%)",
        'temp_label': "Temperature (°C)",
        'recommend_btn': "🌍 Recommend Best Crop",
        'analyzing': "🔍 Analyzing Soil & Weather Patterns...",
        'rec_success': "✅ Recommended Crop:",
        'risk_title': "🛡️ Climate Risk Advisory",
        'opt_cond': "✅ Optimal Conditions",
        'opt_desc': "The current climate profile is balanced and favorable for most crops. No specific preventive actions required.",
        # Risk Titles & Messages
        'risk_drought': "Drought Risk / Water Scarcity",
        'risk_flood': "Flood Risk / Excess Moisture",
        'risk_heat': "Heat Stress Warning",
        'risk_cold': "Cold Stress / Frost Risk",
        'risk_disease': "Fungal Disease Alert",
        'drought_msg': "Rainfall is significantly below average.",
        'flood_msg': "Rainfall is potentially excessive.",
        'heat_msg': "Temperature is high for many crops.",
        'cold_msg': "Temperature is lower than optimal.",
        'disease_msg': "High Humidity favors pathogen growth.",
        # New Additions
        'profit_trend': "📊 Profit Trend Analysis",
        'prev_year': "Previous Year",
        'this_year': "This Year",
        'profit_inc': "📈 Profit Increase",
        'profit_warn': "📉 Profit Warning",
        'rot_analysis': "🔄 Crop Rotation Analysis",
        'mono_risk': "⚠️ Monoculture Risk",
        'rot_check': "ℹ️ Rotation Check",
        'exc_rot': "✅ Excellent Rotation",
        'explain': "Explainability (XAI)",
        'why_pred': "🧠 Why this prediction?",
        'analyzing_factors': "Analyzing key factors for",
        'yield_fore_title': "📈 Yield Forecasting",
        'rain_pred_title': "⛈️ Rainfall Prediction",
        'geo_loc': "Geographic Location (City/Region)",
        'sense_atmos': "🔄 Sense Atmosphere",
        'predict_rain': "Predict Rainfall",
        'forecast_yield': "Forecast Yield",
        'farming_hist': "Farming History",
        'prev_crop_label': "Previous Year Crop",
        # Yield/Rain Inputs
        'region': "📍 Region",
        'get_weather_btn': "🔄 Get Weather",
        'avg_rain_yr': "Average Rainfall (mm/year)",
        'pesticides': "Pesticides (tonnes)",
        'avg_temp': "Average Temp (°C)",
        'area_acres': "Area (Acres)",
        'prior_yield': "Prior Year Yield (HG/HA)",
        'pred_yield_metric': "Predicted Yield (hg/ha)",
        'wind_speed': "Wind Speed (km/h)",
        'pressure': "Pressure (hPa)",
        'prev_rain': "Previous Rainfall (mm)"
    },
    'Hindi': {
        'title': "🌱 स्मार्ट एग्री एआई सहायक",
        'sidebar_title': "नेविगेशन",
        'model_select': "मॉड्यूल चुनें",
        'crop_rec': "फसल सिफारिश प्रणाली",
        'yield_fore': "उपज पूर्वानुमान",
        'rain_pred': "वर्षा की भविष्यवाणी",
        'loc_label': "📍 खेत का स्थान",
        'auto_detect': "🔄 ऑटो-डिटेक्ट",
        'soil_params': "मिट्टी के पैरामीटर",
        'weather_params': "मौसम के पैरामीटर",
        'recommend_btn': "🌍 सबसे अच्छी फसल सुझाएं",
        'analyzing': "🔍 मिट्टी और मौसम के पैटर्न का विश्लेषण...",
        'rec_success': "✅ अनुशंसित फसल:",
        'risk_title': "🛡️ जलवायु जोखिम सलाह",
        'opt_cond': "✅ अनुकूल स्थिति",
        'opt_desc': "वर्तमान जलवायु प्रोफ़ाइल संतुलित है और अधिकांश फसलों के लिए अनुकूल है।",
        # Risk Titles & Messages
        'risk_drought': "सूखा जोखिम / पानी की कमी",
        'risk_flood': "बाढ़ का खतरा / अधिक नमी",
        'risk_heat': "गर्मी तनाव चेतावनी",
        'risk_cold': "शीत तनाव / पाला जोखिम",
        'risk_disease': "कवक रोग चेतावनी",
        'drought_msg': "वर्षा औसत से काफी कम है।",
        'flood_msg': "वर्षा संभावित रूप से अत्यधिक है।",
        'heat_msg': "तापमान कई फसलों के लिए उच्च है।",
        'cold_msg': "तापमान इष्टतम से कम है।",
        'disease_msg': "उच्च आर्द्रता रोगजनकों के विकास के पक्ष में है।",
        # New Additions
        'profit_trend': "📊 लाभ प्रवृत्ति विश्लेषण",
        'prev_year': "पछला वर्ष",
        'this_year': "इस वर्ष",
        'profit_inc': "📈 लाभ में वृद्धि",
        'profit_warn': "📉 लाभ चेतावनी",
        'rot_analysis': "🔄 फसल चक्र विश्लेषण",
        'mono_risk': "⚠️ एकसस्य जोखिम",
        'rot_check': "ℹ️ रोटेशन जांच",
        'exc_rot': "✅ उत्कृष्ट रोटेशन",
        'explain': "व्याख्या (XAI)",
        'why_pred': "🧠 यह भविष्यवाणी क्यों?",
        'analyzing_factors': "इसके लिए प्रमुख कारकों का विश्लेषण",
        'yield_fore_title': "📈 उपज पूर्वानुमान",
        'rain_pred_title': "⛈️ वर्षा भविष्यवाणी",
        'geo_loc': "भौगोलिक स्थान (शहर/क्षेत्र)",
        'sense_atmos': "🔄 वातावरण का पता लगाएं",
        'predict_rain': "वर्षा की भविष्यवाणी करें",
        'forecast_yield': "उपज का अनुमान लगाएं",
        'farming_hist': "खेती का इतिहास",
        'prev_crop_label': "पिछले वर्ष की फसल",
         # Yield/Rain Inputs Hindi
        'region': "📍 क्षेत्र",
        'get_weather_btn': "🔄 मौसम प्राप्त करें",
        'avg_rain_yr': "औसत वर्षा (मिमी/वर्ष)",
        'pesticides': "कीटनाशक (टन)",
        'avg_temp': "औसत तापमान (°C)",
        'area_acres': "क्षेत्रफल (एकड़)",
        'prior_yield': "पिछले वर्ष की उपज (HG/HA)",
        'pred_yield_metric': "अनुमानित उपज (hg/ha)",
        'wind_speed': "हवा की गति (किमी/घंटा)",
        'pressure': "दबाव (hPa)",
        'prev_rain': "पिछली वर्षा (मिमी)"
    },
    'Marathi': {
        'title': "🌱 स्मार्ट एग्री एआई सहाय्यक",
        'sidebar_title': "नेव्हिगेशन",
        'model_select': "मॉड्यूल निवडा",
        'crop_rec': "पीक शिफारस प्रणाली",
        'yield_fore': "उत्पन्न अंदाज",
        'rain_pred': "पाऊस अंदाज",
        'loc_label': "📍 शेताचे स्थान",
        'auto_detect': "🔄 स्वयंचलित शोधा",
        'soil_params': "मातीचे निकष",
        'weather_params': "हवामानाचे निकष",
        'recommend_btn': "🌍 सर्वोत्तम पीक सुचवा",
        'analyzing': "🔍 माती आणि हवामान नमुन्यांचे विश्लेषण...",
        'rec_success': "✅ शिफारस केलेले पीक:",
        'risk_title': "🛡️ हवामान जोखीम सल्ला",
        'opt_cond': "✅ अनुकूल परिस्थिती",
        'opt_desc': "सद्य हवामान परिस्थिती संतुलित आहे आणि बहुतांश पिकांसाठी अनुकूल आहे।",
        # Risk Titles & Messages
        'risk_drought': "दुष्काळ जोखीम / पाण्याची कमतरता",
        'risk_flood': "पूर जोखीम / जास्त ओलावा",
        'risk_heat': "उष्णता ताण चेतावणी",
        'risk_cold': "थंड ताण / दंव जोखीम",
        'risk_disease': "बुरशीजन्य रोग अलर्ट",
        'drought_msg': "पाऊस सरासरीपेक्षा खूप कमी आहे.",
        'flood_msg': "पाऊस संभाव्यतः जास्त आहे.",
        'heat_msg': "तापमान अनेक पिकांसाठी जास्त आहे.",
        'cold_msg': "तापमान इष्टतम पेक्षा कमी आहे.",
        'disease_msg': "जास्त आर्द्रता रोगजनकांच्या वाढीस अनुकूल आहे।",
         # New Additions
        'profit_trend': "📊 नफा कल विश्लेषण",
        'prev_year': "मागील वर्ष",
        'this_year': "यावर्षी",
        'profit_inc': "📈 नफ्यात वाढ",
        'profit_warn': "📉 नफा चेतावणी",
        'rot_analysis': "🔄 पीक फेरपालट विश्लेषण",
        'mono_risk': "⚠️ एकपैक धोका",
        'rot_check': "ℹ️ फेरपालट तपासा",
        'exc_rot': "✅ उत्कृष्ट फेरपालट",
        'explain': "स्पष्टीकरण (XAI)",
        'why_pred': "🧠 हे भाकीत का?",
        'analyzing_factors': "मुख्य घटकांचे विश्लेषण",
        'yield_fore_title': "📈 उत्पन्न अंदाज",
        'rain_pred_title': "⛈️ पाऊस अंदाज",
        'geo_loc': "भौगोलिक स्थान (शहर/विभाग)",
        'sense_atmos': "🔄 वातावरण ओळखा",
        'predict_rain': "पाऊस अंदाज वर्तवा",
        'forecast_yield': "उत्पन्न अंदाज वर्तवा",
        'farming_hist': "शेती इतिहास",
        'prev_crop_label': "मागील वर्षाचे पीक",
         # Yield/Rain Inputs Marathi
        'region': "📍 विभाग",
        'get_weather_btn': "🔄 हवामान मिळवा",
        'avg_rain_yr': "सरासरी पाऊस (मिमी/वर्ष)",
        'pesticides': "कीटकनाशके (टन)",
        'avg_temp': "सरासरी तापमान (°C)",
        'area_acres': "क्षेत्रफळ (एकर)",
        'prior_yield': "मागील वर्षाचे उत्पन्न (HG/HA)",
        'pred_yield_metric': "अंदाजित उत्पन्न (hg/ha)",
        'wind_speed': "वाऱ्याचा वेग (किमी/तास)",
        'pressure': "दाब (hPa)",
        'prev_rain': "मागील पाऊस (मिमी)"
    },
    'Telugu': {
        'title': "🌱 స్మార్ట్ అగ్రి AI అసిస్టెంట్",
        'sidebar_title': "నావిగేషన్",
        'model_select': "మాడ్యూల్ ఎంచుకోండి",
        'crop_rec': "పంట సిఫార్సు వ్యవస్థ",
        'yield_fore': "దిగుబడి అంచనా",
        'rain_pred': "వర్షపాతం అంచనా",
        'loc_label': "📍 పొలం ఆచూకీ",
        'auto_detect': "🔄 ఆటో-డిటెక్ట్",
        'soil_params': "నేల పారామితులు",
        'weather_params': "వాతావరణ పారామితులు",
        'recommend_btn': "🌍 ఉత్తమ పంటను సిఫార్సు చేయండి",
        'analyzing': "🔍 నేల & వాతావరణ విశ్లేషణ...",
        'rec_success': "✅ సిఫార్సు చేసిన పంట:",
        'risk_title': "🛡️ వాతావరణ ప్రమాద సలహా",
        'opt_cond': "✅ అనుకూల పరిస్థితులు",
        'opt_desc': "ప్రస్తుత వాతావరణం సమతుల్యంగా ఉంది మరియు చాలా పంటలకు అనుకూలంగా ఉంది.",

        # Risk Titles & Messages
        'risk_drought': "కరువు ప్రమాదం / నీటి కొరత",
        'risk_flood': "వరద ప్రమాదం / అధిక తేమ",
        'risk_heat': "వేడి ఒత్తిడి హెచ్చరిక",
        'risk_cold': "శీతల ఒత్తిడి / మంచు ప్రమాదం",
        'risk_disease': "ఫంగల్ వ్యాధి హెచ్చరిక",
        'drought_msg': "వర్షపాతం సగటు కంటే తక్కువగా ఉంది.",
        'flood_msg': "వర్షపాతం అధికంగా ఉండే అవకాశం ఉంది.",
        'heat_msg': "చాలా పంటలకు ఉష్ణోగ్రత ఎక్కువగా ఉంది.",
        'cold_msg': "ఉష్ణోగ్రత సరైన స్థాయి కంటే తక్కువగా ఉంది.",
        'disease_msg': "అధిక తేమ వ్యాధికారక పెరుగుదలకు అనుకూలంగా ఉంటుంది.",
        # New Additions
        'profit_trend': "📊 లాభం ధోరణి విశ్లేషణ",
        'prev_year': "గత సంవత్సరం",
        'this_year': "ఈ సంవత్సరం",
        'profit_inc': "📈 లాభం పెరుగుదల",
        'profit_warn': "📉 లాభం హెచ్చరిక",
        'rot_analysis': "🔄 పంట మార్పిడి విశ్లేషణ",
        'mono_risk': "⚠️ ఏక పంట ప్రమాదం",
        'rot_check': "ℹ️ మార్పిడి తనిఖీ",
        'exc_rot': "✅ అద్భుతమైన మార్పిడి",
        'explain': "వివరణ (XAI)",
        'why_pred': "🧠 ఎందుకు ఈ అంచనా?",
        'analyzing_factors': "ముఖ్య అంశాల విశ్లేషణ",
        'yield_fore_title': "📈 దిగుబడి అంచనా",
        'rain_pred_title': "⛈️ వర్షపాతం అంచనా",
        'geo_loc': "భౌగోళిక స్థానం (నగరం/ప్రాంతం)",
        'sense_atmos': "🔄 వాతావరణాన్ని గుర్తించండి",
        'predict_rain': "వర్షపాతం అంచనా వేయండి",
        'forecast_yield': "దిగుబడిని అంచనా వేయండి",
        'farming_hist': "వ్యవసాయ చరిత్ర",
        'prev_crop_label': "గత సంవత్సరం పంట",
         # Yield/Rain Inputs Telugu
        'region': "📍 ప్రాంతం",
        'get_weather_btn': "🔄 వాతావరణాన్ని పొందండి",
        'avg_rain_yr': "సగటు వర్షపాతం (మి.మీ/సంవత్సరం)",
        'pesticides': "పురుగుమందులు (టన్నులు)",
        'avg_temp': "సగటు ఉష్ణోగ్రత (°C)",
        'area_acres': "విస్తీర్ణం (ఎకరాలు)",
        'prior_yield': "గత సంవత్సరం దిగుబడి (HG/HA)",
        'pred_yield_metric': "అంచనా వేసిన దిగుబడి (hg/ha)",
        'wind_speed': "గాలి వేగం (km/h)",
        'pressure': "ఒత్తిడి (hPa)",
        'prev_rain': "మునుపటి వర్షపాతం (మి.మీ)"
    }
}

def get_text(key, lang):
    # Try target language first
    text = TRANSLATIONS.get(lang, {}).get(key)
    if text:
        return text
    # Fallback to English
    return TRANSLATIONS['English'].get(key, key)

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

def display_preventive_measures(rain, temp, hum, lang='English'):
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
    
    st.markdown(f"### {get_text('risk_title', lang)}")
    
    risks = []
    
    # Helper for dynamic translation
    def trans(text):
        if lang == 'English': return text
        try:
            return GoogleTranslator(source='auto', target=lang.lower()).translate(text)
        except:
            return text
    
    # Thresholds calibrated to Data Generator
    # 1. Drought
    if rain < 100: 
        risks.append({
            "title": get_text('risk_drought', lang),
            "icon": "🌵",
            "msg": trans(f"Rainfall ({rain}mm) is significantly below average."),
            "steps": [trans(s) for s in ["Implement drip irrigation immediately.", "Apply organic mulch to retain soil moisture.", "Avoid water-intensive crops like Rice."]]
        })
        
    # 2. Flood
    elif rain > 300:
        risks.append({
            "title": get_text('risk_flood', lang),
            "icon": "🌊",
            "msg": trans(f"Rainfall ({rain}mm) is potentially excessive."),
            "steps": [trans(s) for s in ["Ensure field drainage channels are clear.", "Use raised bed planting.", "Monitor for root rot diseases."]]
        })

    # 3. Heat Stress
    if temp > 32:
        risks.append({
            "title": get_text('risk_heat', lang),
            "icon": "☀️",
            "msg": trans(f"Temperature ({temp}°C) is high for many crops."),
            "steps": [trans(s) for s in ["Irrigate during evening hours.", "Use shade nets for sensitive plants.", "Increase potassium fertilizer to boost stress tolerance."]]
        })
        
    # 4. Cold/Frost
    elif temp < 18:
        risks.append({
            "title": get_text('risk_cold', lang),
            "icon": "❄️",
            "msg": trans(f"Temperature ({temp}°C) is lower than optimal."),
            "steps": [trans(s) for s in ["Use row covers or plastic tunnels.", "Apply irrigation before frost nights (water holds heat).", "Smoke generation around fields."]]
        })

    # 5. Disease (Humidity)
    if hum > 85:
        risks.append({
            "title": get_text('risk_disease', lang),
            "icon": "🍄",
            "msg": trans(f"High Humidity ({hum}%) favors pathogen growth."),
            "steps": [trans(s) for s in ["Monitor leaves for spots/mold daily.", "Improve air circulation by pruning/spacing.", "Apply preventive organic fungicides (Neem oil)."]]
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
            <div class="risk-title" style="color: #155724;">{get_text('opt_cond', lang)}</div>
            <div class="risk-body">{get_text('opt_desc', lang)}</div>
        </div>
        """, unsafe_allow_html=True) 


def run_crop_recommendation(preprocessor, model, ebmo, lang='English'):
    st.header(get_text('crop_rec', lang))
    
    # Session State
    if 'c_temp' not in st.session_state: st.session_state['c_temp'] = 25.0
    if 'c_hum' not in st.session_state: st.session_state['c_hum'] = 80.0
    if 'c_rain' not in st.session_state: st.session_state['c_rain'] = 200.0
    
    c1, c2 = st.columns([3, 1])
    with c1:
        loc = st.text_input(get_text('loc_label', lang), "Nagpur, India")
    with c2:
        st.write("")
        st.write("")
        if st.button(get_text('auto_detect', lang)):
            w = get_live_weather(loc)
            st.session_state['c_temp'] = w['temp']
            st.session_state['c_hum'] = w['hum']
            st.session_state['c_rain'] = w['rain']
            st.toast("Weather Updated!")
 
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(get_text('soil_params', lang))
        n = st.number_input("Nitrogen (N)", 0, 140, 90)
        p = st.number_input("Phosphorous (P)", 0, 145, 40)
        k = st.number_input("Potassium (K)", 0, 205, 40)
        ph = st.number_input("pH Level", 0.0, 14.0, 6.5)
        
    with col2:
        st.subheader(get_text('weather_params', lang))
        temp = st.number_input("Temperature (°C)", 0.0, 50.0, key='c_temp')
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, key='c_hum')
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, key='c_rain')
        
    # User Request: Use previous year crop data
    st.subheader(get_text('farming_hist', lang))
    crop_options = ['None', 'Rice', 'Maize', 'Chickpea', 'Kidneybeans', 'Pigeonpeas', 'Mothbeans', 'Mungbean', 'Blackgram', 'Lentil', 'Pomegranate', 'Banana', 'Mango', 'Grapes', 'Watermelon', 'Muskmelon', 'Apple', 'Orange', 'Papaya', 'Coconut', 'Cotton', 'Jute', 'Coffee']
    prev_crop = st.selectbox(get_text('prev_crop_label', lang), crop_options)
        

        
    if st.button(get_text('recommend_btn', lang)):
        with st.spinner(get_text('analyzing', lang)):
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
                top_3_idx = np.argsort(probs)[::-1][:3]
                # Decode numerical classes back to string labels
                classes_indices = model.classes_
                classes_names = preprocessor.encoders['label'].inverse_transform(classes_indices)
                top_3_crops = classes_names[top_3_idx]
                
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
                suitable_crops = [crop for i, crop in enumerate(top_3_crops) if probs[top_3_idx[i]] > 0.2]
                if not suitable_crops: suitable_crops = [best_agronomic_crop]
                
                # Find most profitable among strictly suitable
                best_profit_crop = max(suitable_crops, key=lambda c: CROP_PROFIT.get(c, 0))
                
                # Final Decision Logic
                is_high_confidence = best_prob > 0.50
                # Fix: Convert numpy array to list for .index() method
                profit_candidate_prob = probs[top_3_idx[list(top_3_crops).index(best_profit_crop)]] if best_profit_crop in top_3_crops else 0

                if best_profit_crop != best_agronomic_crop:
                    if is_high_confidence and profit_candidate_prob < 0.3:
                         crop_name = best_agronomic_crop
                    else:
                         crop_name = best_profit_crop
                else:
                    crop_name = best_agronomic_crop
                    
            else:
                # Fallback if no probability
                pred_idx = model.predict(features_selected)[0]
                crop_name = preprocessor.encoders['label'].inverse_transform([pred_idx])[0]
                
            # --- TRANSLATION OF RESULT ---
            if lang != 'English':
                 try:
                    translator = GoogleTranslator(source='auto', target=lang.lower())
                    crop_name_translated = translator.translate(crop_name)
                 except:
                    crop_name_translated = crop_name
            else:
                crop_name_translated = crop_name
                
            st.success(f"{get_text('rec_success', lang)} **{crop_name_translated}** ({crop_name})")


            # --- Profit Comparison with Previous Year ---
            if prev_crop != 'None':
                prev_profit = CROP_PROFIT.get(prev_crop, 0)
                curr_profit = CROP_PROFIT.get(crop_name, 0)
                diff = curr_profit - prev_profit
                
                st.markdown(f"#### {get_text('profit_trend', lang)}")
                st.write(f"{get_text('prev_year', lang)} ({prev_crop}): **₹{prev_profit}** | {get_text('this_year', lang)} ({crop_name}): **₹{curr_profit}**")
                
                if diff > 0:
                    st.success(f"{get_text('profit_inc', lang)}: +₹{diff}")
                elif diff < 0:
                    st.error(f"{get_text('profit_warn', lang)}: -₹{abs(diff)}")
            
            # --- Crop Rotation Logic (Updated) ---
            st.markdown(f"### {get_text('rot_analysis', lang)}")
            if prev_crop == crop_name:
                st.warning(f"{get_text('mono_risk', lang)}: {prev_crop}")
            elif prev_crop in ['Chickpea', 'Kidneybeans', 'Pigeonpeas', 'Mothbeans', 'Mungbean', 'Blackgram', 'Lentil'] and crop_name in ['Rice', 'Maize', 'Cotton']:
                st.success(f"{get_text('exc_rot', lang)}: {prev_crop} -> {crop_name}")
            elif prev_crop != 'None':
                st.info(f"{get_text('rot_check', lang)}: {prev_crop} -> {crop_name}")
            
            # 4. Explainability (Dynamic & Static)
            st.subheader(get_text('explain', lang))
            
            # A. Rule-Based / Feature Contribution (Dynamic)
            st.markdown(f"#### {get_text('why_pred', lang)}")
            st.write(f"{get_text('analyzing_factors', lang)} **{crop_name}**:")
            
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
            display_preventive_measures(rainfall, temp, humidity, lang)

def run_yield_forecasting(preprocessor, model, lang='English'):
    st.header(get_text('yield_fore', lang))
    st.write("Predict future yield based on environmental factors.")
    
    # Yield Forecasting Auto-Detect
    if 'y_rain' not in st.session_state: st.session_state['y_rain'] = 1200.0
    if 'y_temp' not in st.session_state: st.session_state['y_temp'] = 28.0
    
    c1, c2 = st.columns([3, 1])
    with c1: loc = st.text_input(get_text('region', lang), "Punjab, India")
    with c2:
        st.write("")
        st.write("")
        if st.button(get_text('get_weather_btn', lang)):
            w = get_live_weather(loc)
            st.session_state['y_rain'] = w['rain'] * 10 
            st.session_state['y_temp'] = w['temp']
            st.toast("Fetch Complete!")

    rain = st.number_input(get_text('avg_rain_yr', lang), 0.0, 3000.0, key='y_rain')
    pest = st.number_input(get_text('pesticides', lang), 0.0, 1000.0, 50.0)
    temp = st.number_input(get_text('avg_temp', lang), 0.0, 50.0, key='y_temp')
    area_acres = st.number_input(get_text('area_acres', lang), 0.0, 100000.0, 100.0)
    
    if st.button(get_text('forecast_yield', lang)):
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

def run_rainfall_prediction(preprocessor, model, lang='English'):
    st.header(get_text('rain_pred', lang))
    st.write("Predict rainfall based on atmospheric conditions.")
    
    # Rainfall Auto-Detect
    if 'r_temp' not in st.session_state: st.session_state['r_temp'] = 30.0
    if 'r_hum' not in st.session_state: st.session_state['r_hum'] = 75.0
    if 'r_wind' not in st.session_state: st.session_state['r_wind'] = 10.0
    if 'r_press' not in st.session_state: st.session_state['r_press'] = 1010.0
    if 'r_prev_rain' not in st.session_state: st.session_state['r_prev_rain'] = 0.0
    
    c1, c2 = st.columns([3, 1])
    with c1: loc = st.text_input(get_text('geo_loc', lang), "New Delhi")
    with c2:
        st.write("")
        st.write("")
        if st.button(get_text('sense_atmos_btn', lang)):
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
        temp = st.number_input(get_text('temp_label', lang), 0.0, 50.0, key='r_temp')
        hum = st.number_input(get_text('humidity_label', lang), 0.0, 100.0, key='r_hum')
    with col2:
        wind = st.number_input(get_text('wind_speed', lang), 0.0, 100.0, key='r_wind')
        pressure = st.number_input(get_text('pressure', lang), 900.0, 1100.0, key='r_press')
        
    prev_rain = st.number_input(get_text('prev_rain', lang), 0.0, 500.0, key='r_prev_rain')
    
    if st.button(get_text('predict_rain', lang)):
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
