"""
Crop Recommendation Prediction Helper
=====================================

This script demonstrates how to load a saved model and make predictions.

Author: ML Engineer
Date: 2025
"""

import joblib
import numpy as np
from pathlib import Path


def load_crop_recommendation_model(model_path='models/crop_recommendation.pkl'):
    """
    Load the crop recommendation model with all preprocessing objects.
    
    Parameters:
    -----------
    model_path : str or Path
        Path to the saved model file
    
    Returns:
    --------
    dict
        Dictionary containing model, encoders, and scaler
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_data = joblib.load(model_path)
    return model_data


def predict_crop(N, P, K, temperature, humidity, ph, rainfall, 
                 model_path='models/crop_recommendation.pkl'):
    """
    Predict the recommended crop for given soil and climate parameters.
    
    Parameters:
    -----------
    N : float
        Nitrogen content in soil
    P : float
        Phosphorus content in soil
    K : float
        Potassium content in soil
    temperature : float
        Temperature in Celsius
    humidity : float
        Humidity percentage
    ph : float
        Soil pH value
    rainfall : float
        Rainfall in mm
    model_path : str
        Path to the saved model file
    
    Returns:
    --------
    str
        Predicted crop name
    """
    # Load model data
    model_data = load_crop_recommendation_model(model_path)
    
    model = model_data['model']
    scaler = model_data.get('scaler')
    encoders = model_data.get('encoders')
    
    # Prepare input as array (features in order: N, P, K, temperature, humidity, ph, rainfall)
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Apply scaling if scaler is available
    if scaler:
        input_scaled = scaler.transform(input_features)
    else:
        input_scaled = input_features
    
    # Make prediction
    prediction_encoded = model.predict(input_scaled)
    
    # Decode prediction back to crop name if encoder is available
    if encoders and 'label' in encoders:
        label_encoder = encoders['label']
        crop_name = label_encoder.inverse_transform(prediction_encoded)[0]
        return crop_name
    else:
        return prediction_encoded[0]


if __name__ == "__main__":
    """
    Example usage
    """
    print("="*60)
    print("CROP RECOMMENDATION PREDICTION")
    print("="*60)
    
    # Example: Predict crop for given soil and climate conditions
    N = 90
    P = 42
    K = 43
    temperature = 20.87
    humidity = 82.00
    ph = 6.5
    rainfall = 202.93
    
    print(f"\nInput Parameters:")
    print(f"  N (Nitrogen): {N}")
    print(f"  P (Phosphorus): {P}")
    print(f"  K (Potassium): {K}")
    print(f"  Temperature: {temperature}°C")
    print(f"  Humidity: {humidity}%")
    print(f"  pH: {ph}")
    print(f"  Rainfall: {rainfall} mm")
    
    try:
        predicted_crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        print(f"\n{'='*60}")
        print(f"Recommended Crop: {predicted_crop}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have trained the model first by running:")
        print("  python scripts/crop_recommendation_model.py")

