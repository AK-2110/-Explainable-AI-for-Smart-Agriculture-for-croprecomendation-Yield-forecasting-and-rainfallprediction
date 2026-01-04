# 🌱 Explainable AI for Smart Agriculture

An intelligent agricultural decision support system leveraging **Explainable AI (XAI)**, **Hybrid Deep Learning Models**, and **Heuristic Optimization** to provide transparent and accurate insights for farmers.

## 🚀 Key Features

### 1. Crop Recommendation (with XAI)
- **Model**: Support Vector Machine (SVM) optimized with **EBMO (Enhanced Barnacle Mating Optimization)**.
- **Explainability**:
    - **Dynamic Feature Comparison**: Visualizes your specific farm conditions vs. the "Ideal" requirements for the recommended crop.
    - **Rule-Based Explanation**: Intelligible text explaining *why* a prediction was made (e.g., "High Rainfall favored Rice").
    - **Profit Analysis**: Recommends the most profitable crop among suitable options and compares it with last year's profit.
    - **Crop Rotation Checks**: Advises on monoculture risks and beneficial rotations (e.g., Legume -> Cereal).

### 2. Yield Forecasting
- **Model**: LSTM (Long Short-Term Memory) for time-series forecasting.
- **Functionality**: Predicts crop yield (hg/ha) based on:
    - Average Rainfall
    - Pesticide usage
    - Temperature
    - Farming Area
- **Auto-Detect**: Simulates fetching live weather data for the selected region.

### 3. Rainfall Prediction
- **Model**: Transformer-based Architecture for capturing complex atmospheric dependencies.
- **Functionality**: Forecasts expected rainfall (mm) based on:
    - Temperature, Humidity, Wind Speed, Pressure
    - Geographic Location
- **Location-Aware**: Uses the specific city name provided to contextualize the result.

## 🛠️ Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/) (Interactive Web Interface)
- **Machine Learning**: Scikit-Learn (SVM), TensorFlow/Keras (LSTM, Transformer)
- **Optimization**: Custom EBMO (Evolutionary Algorithm)
- **NLP/Features**: XLNet (Feature Extraction)
- **Explainability**: SHAP, LIME, Custom Heuristics
- **Data Visualization**: Matplotlib

## 📦 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/AK-2110/-Explainable-AI-for-Smart-Agriculture-for-croprecomendation-Yield-forecasting-and-rainfallprediction.git
   cd -Explainable-AI-for-Smart-Agriculture-for-croprecomendation-Yield-forecasting-and-rainfallprediction
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Synthetic Data & Train Models**
   *(First time setup only)*
   ```bash
   python main.py
   ```
   This script will:
   - Generate synthetic agricultural datasets.
   - Train the SVM, LSTM, and Transformer models.
   - Save the trained models in the `models/` directory.

## 🏃 Usage

Run the Streamlit application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

## 📂 Project Structure
```
├── app.py                 # Streamlit Web Application Entry Point
├── main.py                # Pipeline for Data Gen, Training & Evaluation
├── requirements.txt       # Python Dependencies
├── .gitignore             # Git exclusion rules
├── src/
│   ├── data_generator.py  # Synthetic Data Creation
│   ├── feature_extraction.py # XLNet Feature Extractor
│   ├── feature_selection.py  # EBMO Implementation
│   ├── models.py          # Model Architectures (LSTM, Transformer)
│   ├── preprocessing.py   # Data cleaning & Scaling
│   └── explainability.py  # XAI Classes (SHAP/LIME)
├── models/                # Saved Model Artifacts (Ignored by Git)
├── outputs/               # Generated Plots/Visuals (Ignored by Git)
└── data/                  # Generated CSV Datasets (Ignored by Git)
```
