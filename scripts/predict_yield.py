"""
Crop Yield Prediction Helper
=============================

This script demonstrates how to load a saved yield prediction model and make predictions.

Author: ML Engineer
Date: 2025
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path


def load_yield_prediction_model(model_path='models/crop_yield_predictor.pkl'):
    """
    Load the crop yield prediction model with all preprocessing objects.
    
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


def predict_yield(soil_type=None, rainfall=None, temperature=None, 
                  fertilizer=None, year=None, crop=None,
                  State_Name=None, District_Name=None, Season=None,
                  model_path='models/crop_yield_predictor.pkl'):
    """
    Predict crop yield for given parameters.
    
    Note: The exact features depend on what was used during training.
    This function tries to handle common features from crop_production.csv.
    
    Parameters:
    -----------
    soil_type : str, optional
        Type of soil
    rainfall : float, optional
        Rainfall in mm
    temperature : float, optional
        Temperature in Celsius
    fertilizer : float, optional
        Fertilizer amount (kg/acre)
    year : int, optional
        Year
    crop : str, optional
        Crop name
    State_Name : str, optional
        State name (from crop_production dataset)
    District_Name : str, optional
        District name (from crop_production dataset)
    Season : str, optional
        Season (from crop_production dataset)
    model_path : str
        Path to the saved model file
    
    Returns:
    --------
    float
        Predicted yield (tons per hectare)
    """
    # Load model data
    model_data = load_yield_prediction_model(model_path)
    
    model = model_data['model']
    scaler = model_data.get('scaler')
    encoders = model_data.get('encoders')
    
    # Get feature names from the model (if available)
    # This is a simplified approach - you may need to adjust based on your actual model
    
    print("Warning: This function needs to be customized based on your actual model features.")
    print("Please check what features your model was trained with.")
    
    # Example: Create a DataFrame with available features
    # Adjust this based on your actual preprocessing pipeline
    features = {}
    
    # Add numerical features if provided
    if rainfall is not None:
        features['rainfall'] = rainfall
    if temperature is not None:
        features['temperature'] = temperature
    if fertilizer is not None:
        features['fertilizer'] = fertilizer
    if year is not None:
        features['year'] = year
    
    # Create DataFrame
    if features:
        input_df = pd.DataFrame([features])
        
        # Apply encoding for categorical features if encoders exist
        # This is simplified - you may need to adjust
        if encoders:
            for col, encoder in encoders.items():
                if col in input_df.columns:
                    input_df[col] = encoder.transform(input_df[col].astype(str))
        
        # Apply scaling
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        return prediction
    else:
        raise ValueError("Please provide at least some input features")


if __name__ == "__main__":
    """
    Example usage
    """
    print("="*60)
    print("CROP YIELD PREDICTION")
    print("="*60)
    print("\nNote: Please customize this function based on your model's features.")
    print("Check the preprocessing pipeline to see what features are used.\n")

