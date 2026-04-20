"""
Smart Farming ML - Crop Recommendation App
==========================================

Streamlit web application for crop recommendation based on soil and climate parameters.

Author: Shubham Singh
Date: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
import sys
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
try:
    from scripts.assistant import answer_query as rag_answer_query, retrieve as rag_retrieve, build_vector_store as rag_build_vector_store
except Exception:
    rag_answer_query = None
    rag_retrieve = None
    rag_build_vector_store = None

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Smart Farming - Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem 0;
    }
    .tagline {
        font-size: 1.05rem;
        color: #388E3C;
        text-align: center;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
        font-style: italic;
    }
    .prediction-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        margin: 1rem 0;
    }
    .confidence-text {
        font-size: 1.1rem;
        color: #2E7D32;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def fetch_crop_wikipedia(crop_name: str):
    """
    Fetch a short description and thumbnail image for a crop from Wikipedia REST API.

    Returns a dict: { 'ok': bool, 'description': str|None, 'thumbnail': str|None }
    """
    if not crop_name:
        return { 'ok': False, 'description': None, 'thumbnail': None }

    try:
        title = crop_name.strip().title()
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
        resp = requests.get(url, timeout=8)
        if resp.status_code == 404:
            return { 'ok': False, 'description': None, 'thumbnail': None }
        resp.raise_for_status()
        data = resp.json()

        # Prefer extract for a concise summary
        description = data.get('extract') or data.get('description') or None
        thumb_url = None
        thumb = data.get('thumbnail')
        if isinstance(thumb, dict):
            thumb_url = thumb.get('source')

        return { 'ok': True, 'description': description, 'thumbnail': thumb_url }
    except Exception:
        return { 'ok': False, 'description': None, 'thumbnail': None }


@st.cache_data(show_spinner=False)
def fetch_crop_info(crop_name: str):
    """
    Fetch crop info with robust fallbacks and synonyms.

    1) Clean and normalize the crop name (lowercase, strip, map synonyms)
    2) Try Wikipedia REST summary endpoint
    3) If not found, fallback to Wikipedia Search API (first result)
    4) If image missing, fallback to Pexels (if PEXELS_API_KEY is set)

    Returns a tuple: (description, image_url, page_url) or (None, None, None)
    """
    def normalize_name(name: str) -> str:
        synonyms = {
            'paddy': 'rice',
            'brinjal': 'eggplant',
            'maize': 'corn',
            'ladyfinger': 'okra',
            'groundnut': 'peanut',
        }
        n = (name or '').strip().lower().replace('_', ' ')
        n = ' '.join(n.split())  # collapse spaces
        n = synonyms.get(n, n)
        return n

    def wiki_summary_by_title(title: str):
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
        headers = {
            'User-Agent': 'SmartFarmingApp/1.0 (contact: support@example.com)'
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except Exception:
            return None

    def wiki_search_first(query: str):
        api = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json',
            'srlimit': 1
        }
        headers = {
            'User-Agent': 'SmartFarmingApp/1.0 (contact: support@example.com)'
        }
        r = requests.get(api, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        try:
            data = r.json()
        except Exception:
            return None
        hits = (((data or {}).get('query') or {}).get('search')) or []
        if hits:
            return hits[0].get('title')
        return None

    name_norm = normalize_name(crop_name)
    if not name_norm:
        return (None, None, None)

    # Try direct REST summary first
    title_try = name_norm.title()
    summary = None
    try:
        summary = wiki_summary_by_title(title_try)
    except Exception:
        summary = None

    # Fallback to search if summary is missing or not a valid content page
    if not summary or (summary.get('type') in {'disambiguation'} and not summary.get('extract')):
        try:
            t = wiki_search_first(name_norm)
            if t:
                summary = wiki_summary_by_title(t)
        except Exception:
            summary = summary  # keep whatever we had (likely None)

    description = None
    thumb_url = None
    page_url = None

    if summary:
        description = summary.get('extract') or summary.get('description')
        thumb = summary.get('thumbnail')
        if isinstance(thumb, dict):
            thumb_url = thumb.get('source')
        page = summary.get('content_urls') or {}
        page_desktop = page.get('desktop') or {}
        page_url = page_desktop.get('page')
        # As a fallback, construct page link from title
        if not page_url and summary.get('title'):
            page_url = f"https://en.wikipedia.org/wiki/{requests.utils.quote(summary['title'].replace(' ', '_'))}"

    # If no image, try Pexels
    if not thumb_url:
        api_key = os.getenv('PEXELS_API_KEY')
        if api_key:
            try:
                url = "https://api.pexels.com/v1/search"
                headers = { 'Authorization': api_key }
                params = { 'query': name_norm.title(), 'per_page': 1 }
                resp = requests.get(url, headers=headers, params=params, timeout=8)
                resp.raise_for_status()
                pdata = resp.json()
                photos = pdata.get('photos') or []
                if photos:
                    src = photos[0].get('src') or {}
                    thumb_url = src.get('medium') or src.get('large') or src.get('original')
            except Exception:
                pass

    if description or thumb_url or page_url:
        return (description, thumb_url, page_url)
    return (None, None, None)


def fetch_weather_data(city_name):
    """
    Fetch weather data from OpenWeatherMap API.
    
    Parameters:
    -----------
    city_name : str
        Name of the city
    
    Returns:
    --------
    dict
        Dictionary with weather data or error message
    """
    API_KEY = os.getenv('OPENWEATHER_API_KEY')
    API_BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    if not API_KEY:
        return {
            'status': 'error',
            'message': 'OPENWEATHER_API_KEY not found in environment variables. Please add it to your .env file.'
        }
    
    current_weather_url = f"{API_BASE_URL}/weather"
    params = {
        'q': city_name,
        'appid': API_KEY,
        'units': 'metric'
    }
    
    try:
        response = requests.get(current_weather_url, params=params, timeout=10)
        
        if response.status_code == 401:
            return {
                'status': 'error',
                'message': 'Invalid API key. Please check your OPENWEATHER_API_KEY in the .env file.'
            }
        elif response.status_code == 404:
            return {
                'status': 'error',
                'message': f'City "{city_name}" not found. Please check the city name.'
            }
        
        response.raise_for_status()
        data = response.json()
        
        # Extract temperature and humidity
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        
        # Extract rainfall (if available)
        rainfall = 0
        if 'rain' in data:
            if '1h' in data['rain']:
                rainfall = data['rain']['1h']
            elif '3h' in data['rain']:
                rainfall = data['rain']['3h'] / 3
        
        return {
            'status': 'success',
            'temperature': round(temperature, 2),
            'humidity': round(humidity, 2),
            'rainfall': round(rainfall, 2),
            'city': data.get('name', city_name),
            'description': data['weather'][0]['description']
        }
    
    except requests.exceptions.RequestException as e:
        return {
            'status': 'error',
            'message': f'Network error: {str(e)}. Please check your internet connection.'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Unexpected error: {str(e)}'
        }


@st.cache_data
def load_yield_model(model_path='models/best_yield_model.pkl'):
    """
    Load the trained crop yield prediction model.
    
    Parameters:
    -----------
    model_path : str
        Path to the yield model file
    
    Returns:
    --------
    dict or None
        Dictionary with model, scaler, encoders, or None if error
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        return None
    
    try:
        model_data = joblib.load(model_path)
        
        # Validate model structure
        if 'model' not in model_data:
            raise ValueError("Yield model file missing 'model' key")
        
        return model_data
    except Exception as e:
        # Silent fail - will be handled gracefully in prediction
        return None


@st.cache_data
def load_model(model_path='models/best_crop_model.pkl'):
    """
    Load the trained crop recommendation model.
    
    Parameters:
    -----------
    model_path : str
        Path to the model file
    
    Returns:
    --------
    dict or None
        Dictionary with model, scaler, encoder, or None if error
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        return None
    
    try:
        model_data = joblib.load(model_path)
        
        # Validate model structure
        if 'model' not in model_data:
            raise ValueError("Model file missing 'model' key")
        
        # Check if model supports predict_proba (required for confidence scores)
        model = model_data['model']
        if not hasattr(model, 'predict_proba'):
            st.warning("⚠️ Model does not support probability prediction. Confidence scores may not be available.")
        
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def predict_crop(model_data, N, P, K, temperature, humidity, ph, rainfall):
    """
    Predict crop recommendation based on input parameters.
    
    Parameters:
    -----------
    model_data : dict
        Dictionary containing model, scaler, encoder
    N, P, K : float
        Soil nutrient levels
    temperature : float
        Temperature in Celsius
    humidity : float
        Humidity percentage
    ph : float
        Soil pH value
    rainfall : float
        Rainfall in mm
    
    Returns:
    --------
    tuple
        (predicted_crop, confidence_score, probabilities) or None if error
    """
    try:
        model = model_data['model']
        scaler = model_data.get('scaler')
        label_encoder = model_data.get('label_encoder')
        
        # Ensure all inputs are floats
        N = float(N)
        P = float(P)
        K = float(K)
        temperature = float(temperature)
        humidity = float(humidity)
        ph = float(ph)
        rainfall = float(rainfall)
        
        # Prepare input features in the EXACT order used during training:
        # [N, P, K, temperature, humidity, ph, rainfall]
        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Validate input shape
        if input_features.shape[1] != 7:
            raise ValueError(f"Input dimension mismatch. Expected 7 features, got {input_features.shape[1]}")
        
        # Scale features if scaler is available
        if scaler:
            try:
                input_scaled = scaler.transform(input_features)
            except Exception as e:
                raise ValueError(f"Error scaling features: {str(e)}")
        else:
            input_scaled = input_features
        
        # Make prediction
        try:
            prediction_proba = model.predict_proba(input_scaled)[0]
            prediction_encoded = model.predict(input_scaled)[0]
        except Exception as e:
            raise ValueError(f"Error making prediction: {str(e)}")
        
        # Get confidence score (max probability)
        confidence_score = np.max(prediction_proba)
        
        # Decode prediction to crop name
        if label_encoder:
            try:
                predicted_crop = label_encoder.inverse_transform([prediction_encoded])[0]
                # Get all class probabilities
                classes = label_encoder.classes_
                probabilities = dict(zip(classes, prediction_proba))
            except Exception as e:
                raise ValueError(f"Error decoding prediction: {str(e)}")
        else:
            predicted_crop = f"Class {prediction_encoded}"
            probabilities = {f"Class {i}": prob for i, prob in enumerate(prediction_proba)}
        
        return predicted_crop, confidence_score, probabilities
    
    except Exception as e:
        raise RuntimeError(f"Prediction error: {str(e)}")


def predict_yield(yield_model_data, N, P, K, temperature, humidity, ph, rainfall, crop_name):
    """
    Predict crop yield based on soil/climate parameters and crop type.
    
    Parameters:
    -----------
    yield_model_data : dict
        Dictionary containing yield model, scaler, encoders
    N, P, K : float
        Soil nutrient levels
    temperature : float
        Temperature in Celsius
    humidity : float
        Humidity percentage
    ph : float
        Soil pH value
    rainfall : float
        Rainfall in mm
    crop_name : str
        Name of the predicted crop
    
    Returns:
    --------
    float or None
        Predicted yield in tons/hectare, or None if prediction fails
    """
    if yield_model_data is None:
        return None
    
    try:
        model = yield_model_data['model']
        scaler = yield_model_data.get('scaler')
        encoders = yield_model_data.get('encoders')
        feature_names = yield_model_data.get('feature_names', [])
        
        # Try to encode crop if encoder is available
        crop_encoded = None
        if encoders:
            # Look for 'Crop' encoder in the encoders dict
            if 'Crop' in encoders:
                crop_encoder = encoders['Crop']
                try:
                    crop_encoded = crop_encoder.transform([crop_name])[0]
                except (ValueError, KeyError):
                    # Crop not in encoder classes, use most common or default
                    # Try lowercase or title case
                    crop_variations = [crop_name.lower(), crop_name.title(), crop_name.upper()]
                    for var in crop_variations:
                        try:
                            crop_encoded = crop_encoder.transform([var])[0]
                            break
                        except:
                            continue
                    
                    if crop_encoded is None:
                        # Use first class as default if crop not found
                        crop_encoded = 0
        
        # Prepare input features
        # The yield model might expect different features than crop recommendation
        # We'll try to match what features the model expects
        
        # If feature_names are provided, use them to construct input
        if feature_names:
            # Create a mapping of our available features
            available_features = {
                'N': float(N),
                'P': float(P),
                'K': float(K),
                'temperature': float(temperature),
                'humidity': float(humidity),
                'ph': float(ph),
                'rainfall': float(rainfall)
            }
            
            # Build feature array based on model's expected features
            feature_array = []
            for feat_name in feature_names:
                if feat_name in available_features:
                    feature_array.append(available_features[feat_name])
                elif feat_name == 'Crop' and crop_encoded is not None:
                    feature_array.append(float(crop_encoded))
                else:
                    # Feature not available, use 0 or mean (this is a fallback)
                    feature_array.append(0.0)
            
            input_features = np.array([feature_array])
        else:
            # Default: use N, P, K, temperature, humidity, ph, rainfall + encoded crop
            if crop_encoded is not None:
                input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall, crop_encoded]])
            else:
                input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall, 0]])
        
        # Scale features if scaler is available
        if scaler:
            try:
                input_scaled = scaler.transform(input_features)
            except Exception:
                # Scaling failed, use unscaled
                input_scaled = input_features
        else:
            input_scaled = input_features
        
        # Make prediction
        predicted_yield = model.predict(input_scaled)[0]
        
        # Ensure yield is positive
        predicted_yield = max(0, float(predicted_yield))
        
        return predicted_yield
    
    except Exception as e:
        # Return None on any error - will be handled gracefully
        return None


def load_shap_plots():
    """
    Check if SHAP plots are available.
    
    Returns:
    --------
    tuple
        (feature_importance_path, shap_summary_path) or (None, None)
    """
    feature_importance_path = Path('results/feature_importance_bar.png')
    shap_summary_path = Path('results/shap_summary.png')
    
    has_feature_plot = feature_importance_path.exists()
    has_summary_plot = shap_summary_path.exists()
    
    return (feature_importance_path if has_feature_plot else None,
            shap_summary_path if has_summary_plot else None)


def main():
    """
    Main Streamlit application.
    """
    # Initialize session state for weather data
    if 'weather_data' not in st.session_state:
        st.session_state.weather_data = None
    if 'temperature' not in st.session_state:
        st.session_state.temperature = float(25.0)
    if 'humidity' not in st.session_state:
        st.session_state.humidity = float(60.0)
    if 'rainfall' not in st.session_state:
        st.session_state.rainfall = float(150.0)
    
    # Header
    st.markdown('<div class="main-header">🌾 Smart Farming - Crop Recommendation System</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="tagline">AI-powered Smart Farming Assistant</div>', 
                unsafe_allow_html=True)
    
    # Project description
    st.markdown("""
    ### Welcome to the Smart Farming ML System
    
    This application uses machine learning to recommend the most suitable crop to grow 
    based on your soil and climate conditions. Simply input the parameters below and 
    get instant recommendations!
    
    ---
    """)
    
    # Weather data fetching section
    st.markdown("### 🌤️ Get Real-Time Weather Data")
    col_city, col_fetch = st.columns([3, 1])
    
    with col_city:
        city_name = st.text_input(
            "Enter City Name",
            placeholder="e.g., Varanasi, Mumbai, Delhi...",
            key="city_input",
            help="Enter the name of the city to fetch current weather data"
        )
    
    with col_fetch:
        st.write("")  # Spacing
        fetch_button = st.button("🌡️ Fetch Weather", type="secondary", use_container_width=True)
    
    # Fetch weather data
    if fetch_button:
        if city_name:
            with st.spinner(f"Fetching weather data for {city_name}..."):
                weather_result = fetch_weather_data(city_name)
                
                if weather_result['status'] == 'success':
                    # Store weather data in session state (ensure float type)
                    fetched_temp = float(weather_result['temperature'])
                    fetched_humidity = float(weather_result['humidity'])
                    fetched_rainfall = float(weather_result['rainfall'])
                    
                    st.session_state.weather_data = weather_result
                    st.session_state.temperature = fetched_temp
                    st.session_state.humidity = fetched_humidity
                    st.session_state.rainfall = fetched_rainfall
                    
                    # Update slider keys in session_state to trigger automatic refresh
                    st.session_state.temp_slider = fetched_temp
                    st.session_state.humidity_slider = fetched_humidity
                    st.session_state.rainfall_slider = fetched_rainfall
                    
                    st.success(f"✅ Weather data fetched successfully for {weather_result['city']}!")
                    st.info(f"🌡️ Temperature: {weather_result['temperature']}°C | "
                           f"💧 Humidity: {weather_result['humidity']}% | "
                           f"🌧️ Rainfall: {weather_result['rainfall']} mm | "
                           f"☁️ Condition: {weather_result['description'].title()}")
                else:
                    # API failed - keep existing values in session_state unchanged
                    # No updates to session_state means previous values are preserved
                    st.error(f"❌ {weather_result['message']}")
                    st.info("💡 Previous values preserved. You can still manually enter weather parameters below.")
        else:
            st.warning("⚠️ Please enter a city name before fetching weather data.")
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.markdown("""
        Minimal, elegant, and responsive ML assistant for smart farming.
        """)

        st.markdown("### 👥 Team")
        st.markdown("- Shubham")

        st.markdown("### 📁 Project Info")
        st.markdown("""
        - Crop recommendation (classification)
        - Yield prediction (regression)
        - XAI insights (SHAP)
        """)

        st.markdown("---")
        st.markdown("### 🌤️ Weather API Status")

        # Check API key status
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if api_key:
            st.success("✅ Weather API Key Found")
            st.caption(f"Key: {api_key[:10]}...{api_key[-4:]}")
        else:
            st.warning("⚠️ Weather API Key Not Found")
            st.caption("Add OPENWEATHER_API_KEY to .env file")
            st.info("Get free API key: https://openweathermap.org/api")

        st.markdown("---")
        st.markdown("### 🛠️ Model Information")

        # Load and display model info
        model_data = load_model()
        if model_data:
            model_name = type(model_data['model']).__name__
            st.success(f"✅ Model Loaded: {model_name}")

            if 'cv_score' in model_data:
                st.info(f"📈 CV Accuracy: {model_data['cv_score']:.4f}")
            if 'test_accuracy' in model_data:
                st.info(f"🎯 Test Accuracy: {model_data['test_accuracy']:.4f}")
        else:
            st.error("❌ Model not found!")
            st.warning("Please ensure `models/best_crop_model.pkl` exists.")
    
    # Load model
    model_data = load_model()
    
    if model_data is None:
        st.error("""
        ## ❌ Model Not Found
        
        The trained model file (`models/best_crop_model.pkl`) was not found.
        
        Please ensure you have:
        1. Trained the model using `scripts/auto_tuning.py`
        2. The model file exists in the `models/` directory
        
        You can train the model by running:
        ```bash
        python scripts/auto_tuning.py
        ```
        """)
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Parameters")
        
        # Align nutrient inputs horizontally
        n_col, p_col, k_col = st.columns(3)
        with n_col:
            N = st.slider(
                "Nitrogen (N)",
                min_value=0,
                max_value=150,
                value=50,
                step=1,
                help="Nitrogen content in soil"
            )
        with p_col:
            P = st.slider(
                "Phosphorus (P)",
                min_value=0,
                max_value=150,
                value=50,
                step=1,
                help="Phosphorus content in soil"
            )
        with k_col:
            K = st.slider(
                "Potassium (K)",
                min_value=0,
                max_value=150,
                value=50,
                step=1,
                help="Potassium content in soil"
            )
        
    with col2:
        st.header(" ")
        st.write("")  # Spacing
        
        # Get temperature from slider (will use session_state value if set, otherwise default)
        temp_default = float(st.session_state.temperature) if 'temperature' in st.session_state else 25.0
        temperature = st.slider(
            "Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=temp_default,
            step=0.1,
            help="Average temperature in Celsius (auto-filled from weather data if fetched)",
            key="temp_slider"
        )
        # Sync slider value back to session_state (user can manually adjust)
        st.session_state.temperature = float(temperature)

        # Get humidity from slider (will use session_state value if set, otherwise default)
        humidity_default = float(st.session_state.humidity) if 'humidity' in st.session_state else 60.0
        humidity = st.slider(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=humidity_default,
            step=0.1,
            help="Relative humidity percentage (auto-filled from weather data if fetched)",
            key="humidity_slider"
        )
        # Sync slider value back to session_state (user can manually adjust)
        st.session_state.humidity = float(humidity)
        
        ph = st.slider(
            "pH",
            min_value=0.0,
            max_value=14.0,
            value=7.0,
            step=0.1,
            help="Soil pH value (0-14) - Manual input only"
        )
        
        # Get rainfall from slider (will use session_state value if set, otherwise default)
        rainfall_default = float(st.session_state.rainfall) if 'rainfall' in st.session_state else 150.0
        rainfall = st.slider(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=rainfall_default,
            step=1.0,
            help="Average rainfall in millimeters (auto-filled from weather data if fetched)",
            key="rainfall_slider"
        )
        # Sync slider value back to session_state (user can manually adjust)
        st.session_state.rainfall = float(rainfall)
    
    # Prediction button
    st.markdown("---")
    predict_button = st.button(
        "🔮 Predict Crop",
        type="primary",
        use_container_width=True
    )
    
    if predict_button:
        # Validate that model is loaded
        if model_data is None:
            st.error("""
            ## ❌ Model Not Available
            
            The crop recommendation model could not be loaded. Please ensure:
            - The model file exists at `models/best_crop_model.pkl`
            - The model file is not corrupted
            - You have run the training script to generate the model
            
            To train the model, run:
            ```bash
            python scripts/auto_tuning.py
            ```
            """)
            return
        
        # Collect all current input values in the exact order used during training
        try:
            # Ensure all values are properly collected (works with both manual and weather-fetched inputs)
            input_values = {
                'N': float(N),
                'P': float(P),
                'K': float(K),
                'temperature': float(temperature),
                'humidity': float(humidity),
                'ph': float(ph),
                'rainfall': float(rainfall)
            }
            
            # Validate inputs are within reasonable ranges
            if not (0 <= input_values['N'] <= 150):
                st.warning(f"⚠️ Nitrogen (N) value {input_values['N']} is outside typical range (0-150)")
            if not (0 <= input_values['P'] <= 150):
                st.warning(f"⚠️ Phosphorus (P) value {input_values['P']} is outside typical range (0-150)")
            if not (0 <= input_values['K'] <= 150):
                st.warning(f"⚠️ Potassium (K) value {input_values['K']} is outside typical range (0-150)")
            if not (0 <= input_values['temperature'] <= 50):
                st.warning(f"⚠️ Temperature value {input_values['temperature']}°C is outside typical range (0-50)")
            if not (0 <= input_values['humidity'] <= 100):
                st.warning(f"⚠️ Humidity value {input_values['humidity']}% is outside typical range (0-100)")
            if not (0 <= input_values['ph'] <= 14):
                st.warning(f"⚠️ pH value {input_values['ph']} is outside typical range (0-14)")
            if not (0 <= input_values['rainfall'] <= 500):
                st.warning(f"⚠️ Rainfall value {input_values['rainfall']}mm is outside typical range (0-500)")
            
            # Make prediction with inputs in the exact training order: [N, P, K, temperature, humidity, ph, rainfall]
            with st.spinner("🔄 Analyzing soil and climate conditions..."):
                predicted_crop, confidence_score, probabilities = predict_crop(
                    model_data, 
                    input_values['N'],
                    input_values['P'],
                    input_values['K'],
                    input_values['temperature'],
                    input_values['humidity'],
                    input_values['ph'],
                    input_values['rainfall']
                )
            
            # Clear any previous generated report in session
            try:
                st.session_state.pop('report_bytes', None)
                st.session_state.pop('report_name', None)
            except Exception:
                pass

            # Display results with clear success message
            st.markdown("---")
            st.success("✅ Prediction completed successfully!")
            
            # Main result section
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            st.markdown("## 🌱 Recommended Crop")
            st.markdown(f"# {predicted_crop.title()}")
            st.success(f"🌱 Recommended Crop: {predicted_crop.title()}")
            
            # Confidence score with clear display
            st.markdown("---")
            confidence_percent = confidence_score * 100
            
            # Use different styling based on confidence level
            if confidence_percent >= 80:
                st.info(f"📊 **Confidence Score:** {confidence_percent:.2f}% (Very High)")
            elif confidence_percent >= 60:
                st.info(f"📊 **Confidence Score:** {confidence_percent:.2f}% (High)")
            else:
                st.warning(f"📊 **Confidence Score:** {confidence_percent:.2f}% (Moderate)")
            
            # Progress bar for confidence
            st.progress(confidence_score)
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Soil & Fertilizer recommendations
            try:
                from scripts.fertilizer_recommender import get_fertilizer_recommendation
            except Exception:
                get_fertilizer_recommendation = None

            if get_fertilizer_recommendation is not None:
                rec = get_fertilizer_recommendation(
                    N=input_values['N'],
                    P=input_values['P'],
                    K=input_values['K'],
                    pH=input_values['ph'],
                    predicted_crop=predicted_crop,
                )

                st.markdown("### 🧪 Soil & Fertilizer Recommendations")
                box = st.container()
                with box:
                    st.info("Tailored tips based on your soil and climate inputs.")
                    status = rec.get("soil_status", "Needs Improvement")
                    items = rec.get("recommendations", [])
                    if status == "Good":
                        st.success("✅ Soil status: Good")
                    else:
                        st.warning("⚠️ Soil status: Needs Improvement")
                    for tip in items[:6]:  # keep concise
                        # Use checkmark for positive notes and warning for issues
                        icon = "✅" if any(k in tip.lower() for k in ["adequate", "optimal", "ideal"]) else "⚠️"
                        st.write(f"{icon} {tip}")
            
            # Yield Prediction
            st.markdown("---")
            st.markdown("### 📈 Yield Prediction")
            
            # Load yield model
            yield_model_data = load_yield_model('models/best_yield_model.pkl')
            predicted_yield_value = None
            
            if yield_model_data is None:
                st.info("""
                💡 **Yield prediction model not available.**
                
                To enable yield prediction, train the yield model by running:
                ```bash
                python scripts/train_yield_model.py
                ```
                
                This will create `models/best_yield_model.pkl` for yield predictions.
                """)
            else:
                # Predict yield
                with st.spinner("🔄 Predicting expected yield..."):
                    predicted_yield = predict_yield(
                        yield_model_data,
                        input_values['N'],
                        input_values['P'],
                        input_values['K'],
                        input_values['temperature'],
                        input_values['humidity'],
                        input_values['ph'],
                        input_values['rainfall'],
                        predicted_crop
                    )
                    predicted_yield_value = predicted_yield
                
                if predicted_yield is not None and predicted_yield > 0:
                    st.success(f"📈 **Expected Yield: {predicted_yield:.2f} tons/hectare**")
                    st.info(f"💡 Based on the recommended crop **{predicted_crop.title()}** and current soil/climate conditions.")
                else:
                    st.warning("""
                    ⚠️ **Yield prediction unavailable.**
                    
                    The yield model could not make a prediction. This might be because:
                    - The crop type is not recognized by the yield model
                    - Feature mismatch between crop recommendation and yield prediction models
                    - Model structure differences
                    
                    You can still use the crop recommendation above.
                    """)
            
            # Persist last results for report generation after rerun
            st.session_state['last_predicted_crop'] = predicted_crop
            st.session_state['last_predicted_yield'] = float(predicted_yield) if (predicted_yield is not None and predicted_yield > 0) else None
            st.session_state['last_fertilizer_info'] = rec if 'rec' in locals() else None

            # Wikipedia crop info
            st.markdown("---")
            st.markdown("### 📚 Learn More About The Recommended Crop")
            with st.spinner("Fetching crop information..."):
                desc, img_url, page_url = fetch_crop_info(predicted_crop)
            if desc or img_url:
                crop_title = predicted_crop.strip().title()

                # Render with Streamlit components only (no raw HTML)
                container = st.container()
                with container:
                    if img_url:
                        col_l, col_c, col_r = st.columns([1, 2, 1])
                        with col_c:
                            st.image(img_url, width=220, caption=f"{crop_title} image")
                    st.subheader(crop_title)
                    if desc:
                        st.write(desc)
                    if desc and page_url:
                        st.link_button("Learn More on Wikipedia", page_url, use_container_width=False)
            else:
                st.warning("⚠️ No Wikipedia information found for this crop.")

            # Top 3 crops (including the top prediction)
            st.markdown("### 🥈 Top 3 Crops")
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]

            for i, (crop, prob) in enumerate(sorted_probs, 1):
                col_name, col_prob, col_info = st.columns([4, 2, 1])
                with col_name:
                    st.write(f"{i}. **{crop.title()}**")
                with col_prob:
                    st.write(f"{prob*100:.2f}%")
                with col_info:
                    # Small inline info control beside each crop
                    info_key = f"info_btn_{i}_{crop}"
                    if hasattr(st, "popover"):
                        # Use popover when available (shows inline content)
                        with st.popover("ℹ️", use_container_width=False):
                            dsc, imgu, pageu = fetch_crop_info(crop)
                            if dsc or imgu:
                                if imgu:
                                    st.image(imgu, width=120)
                                if dsc:
                                    dd = dsc
                                    if len(dd) > 220:
                                        dd = dd[:220].rstrip() + "..."
                                    st.markdown(dd)
                            else:
                                st.warning("No info found for this crop.")
                    else:
                        # Fallback: toggle with a tiny button and render inline below
                        clicked = st.button("ℹ️", key=info_key)
                        if clicked:
                            st.session_state[info_key] = not st.session_state.get(info_key, False)

                # Render fallback details row (when popover unavailable)
                if not hasattr(st, "popover") and st.session_state.get(info_key, False):
                    dsc, imgu, pageu = fetch_crop_info(crop)
                    box_cols = st.columns([1, 5])
                    with box_cols[0]:
                        if imgu:
                            st.image(imgu, width=90)
                    with box_cols[1]:
                        if dsc:
                            dd = dsc
                            if len(dd) > 260:
                                dd = dd[:260].rstrip() + "..."
                            st.markdown(dd)
                        else:
                            st.warning("No info found for this crop.")
                st.progress(prob)
            
            

            # Expandable XAI section
            with st.expander("🔬 Explain Prediction (XAI)", expanded=False):
                st.markdown("""
                #### Feature Importance Analysis
                
                This section shows how different features influence the crop recommendation 
                using SHAP (SHapley Additive exPlanations) values.
                """)
                
                # Load SHAP plots
                feature_importance_path, shap_summary_path = load_shap_plots()
                
                if feature_importance_path:
                    st.markdown("##### Global Feature Importance")
                    st.image(str(feature_importance_path), use_container_width=True)
                    st.caption("This plot shows which features most influence crop predictions globally.")
                
                if shap_summary_path:
                    st.markdown("##### SHAP Summary Plot")
                    st.image(str(shap_summary_path), use_container_width=True)
                    st.caption("""
                    This plot shows:
                    - Feature importance (y-axis)
                    - SHAP values (x-axis) - how each feature pushes the prediction
                    - Feature values (color) - red=high, blue=low
                    """)
                
                if not feature_importance_path and not shap_summary_path:
                    st.warning("""
                    ⚠️ SHAP plots not found.
                    
                    To generate SHAP analysis plots, run:
                    ```bash
                    python scripts/xai_analysis.py
                    ```
                    
                    This will create feature importance and summary plots in the `results/` folder.
                    """)
            
            # Show current input summary
            with st.expander("📋 Input Summary", expanded=False):
                input_df = pd.DataFrame({
                    'Parameter': ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 
                                 'Temperature', 'Humidity', 'pH', 'Rainfall'],
                    'Value': [
                        input_values['N'],
                        input_values['P'],
                        input_values['K'],
                        input_values['temperature'],
                        input_values['humidity'],
                        input_values['ph'],
                        input_values['rainfall']
                    ],
                    'Unit': ['ppm', 'ppm', 'ppm', '°C', '%', 'pH', 'mm']
                })
                st.dataframe(input_df, use_container_width=True, hide_index=True)
        
        except ValueError as e:
            st.error(f"## ❌ Input Validation Error")
            st.error(f"**Error:** {str(e)}")
            st.info("""
            Please ensure:
            - All input values are valid numbers
            - Input dimensions match the model's expected format (7 features)
            - The model file structure is correct
            """)
        
        except RuntimeError as e:
            st.error(f"## ❌ Prediction Error")
            st.error(f"**Error:** {str(e)}")
            st.info("""
            This error occurred during prediction. Possible causes:
            - Model structure mismatch
            - Feature scaling error
            - Label encoding error
            
            Please check:
            - Model file is from the same training pipeline
            - All preprocessing components (scaler, encoder) are included
            """)
        
        except Exception as e:
            st.error(f"## ❌ Unexpected Error")
            st.error(f"**Error:** {str(e)}")
            st.info("""
            An unexpected error occurred. Please:
            - Check all input values are valid
            - Verify the model file is correct
            - Try refreshing the page or restarting the app
            """)
    
    # Always-visible Report section (uses session_state to avoid losing state on rerun)
    st.markdown("---")
    st.markdown("### 📄 Download Report")

    if not st.session_state.get('last_predicted_crop'):
        st.info("Generate a prediction first, then create the report here.")
    else:
        def _find_latest_chart_for_crop(name: str):
            try:
                base = Path('outputs') / 'charts'
                if not base.exists():
                    return None
                key = str(name).strip().lower()
                pngs = [p for p in base.glob('*.png') if key in p.name.lower()]
                pngs = sorted(pngs, key=lambda p: p.stat().st_mtime, reverse=True)
                return str(pngs[0]) if pngs else None
            except Exception:
                return None

        latest_chart_path = _find_latest_chart_for_crop(st.session_state['last_predicted_crop'])

        if st.button("📄 Generate Report", key="gen_report_btn", type="primary", use_container_width=True):
            with st.spinner("Generating report..."):
                try:
                    from fpdf import FPDF

                    output_dir = Path('outputs') / 'reports'
                    output_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
                    pdf_path = output_dir / f"Crop_Report_{ts}.pdf"

                    pdf = FPDF(orientation='P', unit='mm', format='A4')
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.add_page()

                    TITLE_COLOR = (27, 94, 32)
                    HEADER_FILL = (235, 255, 235)
                    TEXT_COLOR = (0, 0, 0)

                    # Header
                    pdf.set_fill_color(*HEADER_FILL)
                    pdf.rect(x=10, y=10, w=190, h=14, style='F')
                    pdf.set_xy(10, 10)
                    pdf.set_text_color(*TITLE_COLOR)
                    pdf.set_font('Arial', 'B', 16)
                    pdf.cell(190, 10, "Smart Farming Report", ln=True, align='C')

                    pdf.ln(4)
                    pdf.set_text_color(*TEXT_COLOR)
                    pdf.set_font('Arial', '', 12)
                    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
                    city_name_val = st.session_state.weather_data.get('city') if st.session_state.get('weather_data') else 'Unknown'
                    pdf.cell(0, 8, f"City: {city_name_val}", ln=True)

                    # Recommendation
                    pdf.ln(2)
                    pdf.set_text_color(*TITLE_COLOR)
                    pdf.set_font('Arial', 'B', 13)
                    pdf.cell(0, 8, "[Crop] Recommendation", ln=True)

                    pdf.set_text_color(*TEXT_COLOR)
                    pdf.set_font('Arial', '', 12)
                    left_w, right_w = 60, 120
                    pdf.cell(left_w, 8, "Recommended Crop:", border=0)
                    pdf.cell(right_w, 8, f"{st.session_state['last_predicted_crop'].title()}", ln=True, border=0)
                    pdf.cell(left_w, 8, "Predicted Yield:", border=0)
                    if st.session_state.get('last_predicted_yield'):
                        pdf.cell(right_w, 8, f"{float(st.session_state['last_predicted_yield']):.2f} tons/hectare", ln=True, border=0)
                    else:
                        pdf.cell(right_w, 8, "N/A", ln=True, border=0)

                    # Weather summary
                    pdf.ln(2)
                    pdf.set_text_color(*TITLE_COLOR)
                    pdf.set_font('Arial', 'B', 13)
                    pdf.cell(0, 8, "[Weather] Summary", ln=True)

                    pdf.set_text_color(*TEXT_COLOR)
                    pdf.set_font('Arial', '', 12)
                    temp_v = float(st.session_state.temperature)
                    rain_v = float(st.session_state.rainfall)
                    hum_v = float(st.session_state.humidity)
                    pdf.cell(left_w, 8, "Temperature:", border=0)
                    pdf.cell(right_w, 8, f"{temp_v:.2f} °C", ln=True, border=0)
                    pdf.cell(left_w, 8, "Rainfall:", border=0)
                    pdf.cell(right_w, 8, f"{rain_v:.2f} mm", ln=True, border=0)
                    pdf.cell(left_w, 8, "Humidity:", border=0)
                    pdf.cell(right_w, 8, f"{hum_v:.2f} %", ln=True, border=0)

                    # Fertilizer info
                    pdf.ln(2)
                    pdf.set_text_color(*TITLE_COLOR)
                    pdf.set_font('Arial', 'B', 13)
                    pdf.cell(0, 8, "[Fertilizer] Guidance", ln=True)

                    pdf.set_text_color(*TEXT_COLOR)
                    pdf.set_font('Arial', '', 12)
                    fert = st.session_state.get('last_fertilizer_info')
                    if isinstance(fert, dict):
                        status = fert.get('soil_status', 'Needs Improvement')
                        recs = fert.get('recommendations', [])
                        pdf.cell(left_w, 8, "Soil Status:", border=0)
                        pdf.cell(right_w, 8, f"{status}", ln=True, border=0)
                        if recs:
                            pdf.multi_cell(0, 6, "Recommendations:")
                            for tip in recs[:6]:
                                pdf.multi_cell(0, 6, f"- {tip}")
                        else:
                            pdf.multi_cell(0, 6, "No specific recommendations available.")
                    else:
                        pdf.multi_cell(0, 6, "No fertilizer guidance available.")

                    # Chart
                    pdf.ln(2)
                    pdf.set_text_color(*TITLE_COLOR)
                    pdf.set_font('Arial', 'B', 13)
                    pdf.cell(0, 8, "[Chart] Yield vs Weather", ln=True)

                    pdf.set_text_color(*TEXT_COLOR)
                    pdf.set_font('Arial', '', 12)
                    if latest_chart_path and Path(latest_chart_path).exists():
                        pdf.image(latest_chart_path, x=15, y=None, w=180)
                        pdf.ln(4)
                    else:
                        pdf.cell(0, 8, "Chart not available.", ln=True)

                    # Footer
                    pdf.ln(4)
                    pdf.set_font('Arial', 'I', 10)
                    pdf.cell(0, 8, "Generated by Smart Farming ML System", ln=True, align='C')

                    pdf.output(str(pdf_path))

                    with open(pdf_path, 'rb') as f:
                        st.session_state['report_bytes'] = f.read()
                    st.session_state['report_name'] = pdf_path.name
                    st.success("Report generated. Click below to download.")
                except Exception as e:
                    st.error(f"Failed to generate report: {str(e)}")

        if st.session_state.get('report_bytes') and st.session_state.get('report_name'):
            st.download_button(
                label="⬇️ Download Report",
                data=st.session_state['report_bytes'],
                file_name=st.session_state['report_name'],
                mime="application/pdf",
                use_container_width=True
            )
    
    # AI Assistant Tab
    st.markdown("---")
    _a_tabs = st.tabs(["🧠 AI Assistant"])
    with _a_tabs[0]:
        st.markdown("Ask questions. The assistant will use local docs and optionally OpenAI.")

        if 'ai_chat' not in st.session_state:
            st.session_state.ai_chat = []

        use_web = st.toggle("Use web (OpenAI) answers", value=True, help="If off, only local retrieval is used.")

        # Render chat history
        for msg in st.session_state.ai_chat:
            role = msg.get('role', 'assistant')
            with st.chat_message(role):
                st.markdown(msg.get('content', ''))
                if role == 'assistant' and msg.get('sources'):
                    src_lines = []
                    for s in msg['sources']:
                        src = s.get('source', '')
                        chk = s.get('chunk_id', '')
                        name = Path(src).name if src else src
                        src_lines.append(f"- {name} {('('+chk+')') if chk else ''}")
                    if src_lines:
                        st.caption("Sources:\n" + "\n".join(src_lines))

        prompt = st.chat_input("Type your question here…")
        if prompt:
            # Add user message
            st.session_state.ai_chat.append({"role": "user", "content": prompt})

            # Ensure vector store exists (best-effort)
            try:
                need_build = not (Path('models')/ 'vector_store.joblib').exists() or not (Path('models')/ 'chunk_metadata.json').exists()
            except Exception:
                need_build = False
            if need_build and rag_build_vector_store is not None:
                with st.spinner("Building local knowledge base…"):
                    try:
                        rag_build_vector_store([
                            str(Path('Readme.md')),
                            str(Path('README.md')),
                            str(Path('data')/'crop_info.json'),
                            str(Path('outputs')/'reports'/'*.pdf'),
                        ])
                    except Exception as e:
                        st.warning(f"Could not build vector store: {e}")

            do_web = use_web and bool(os.getenv('OPENAI_API_KEY', '').strip()) and (rag_answer_query is not None)
            with st.spinner("Thinking…"):
                try:
                    if do_web:
                        reply = rag_answer_query(prompt, chat_history=st.session_state.ai_chat)
                        ans = reply.get('answer', 'Sorry, I could not generate an answer.')
                        sources = reply.get('sources', [])
                    else:
                        if rag_retrieve is None:
                            ans = "Offline mode not available."
                            sources = []
                        else:
                            chunks = rag_retrieve(prompt, k=4) or []
                            previews = [ (c.get('text','') or '')[:350] for c in chunks ]
                            ans = ("Offline answer from local docs (top matches):\n\n" + "\n\n---\n\n".join(previews)) if previews else "No local matches found."
                            sources = [ {"source": c.get('source',''), "chunk_id": c.get('chunk_id',''), "score": c.get('score')} for c in chunks ]
                except Exception as e:
                    ans = f"Error: {e}"
                    sources = []

            st.session_state.ai_chat.append({"role": "assistant", "content": ans, "sources": sources})
            with st.chat_message("assistant"):
                st.markdown(ans)
                if sources:
                    src_lines = []
                    for s in sources:
                        src = s.get('source', '')
                        chk = s.get('chunk_id', '')
                        name = Path(src).name if src else src
                        src_lines.append(f"- {name} {('('+chk+')') if chk else ''}")
                    st.caption("Sources:\n" + "\n".join(src_lines))
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #666;'>
        <p><strong>Smart Farming ML System</strong></p>
        <p>Developed by <strong>Shubham</strong></p>
        <p>🌾 Making farming smarter with Machine Learning 🌾</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

