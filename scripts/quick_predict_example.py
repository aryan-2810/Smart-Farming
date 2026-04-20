"""
Quick Prediction Example
=========================

Correct way to load and use the saved crop recommendation model.
Copy and paste this code into Python.
"""

from joblib import load
import numpy as np

# Load the model data (it's a dictionary!)
model_data = load('models/crop_recommendation.pkl')

# Extract the actual model, scaler, and encoders from the dictionary
model = model_data['model']          # The trained RandomForest model
scaler = model_data['scaler']        # StandardScaler for feature normalization
encoders = model_data['encoders']    # LabelEncoders for categorical variables

# Your input features: [N, P, K, temperature, humidity, ph, rainfall]
test_input = np.array([[90, 42, 43, 20.87, 82.00, 6.5, 202.93]])

# IMPORTANT: Scale the input using the same scaler that was used during training
test_input_scaled = scaler.transform(test_input)

# Make prediction (this returns encoded label)
prediction_encoded = model.predict(test_input_scaled)

# Decode the prediction back to crop name using the label encoder
label_encoder = encoders['label']
crop_name = label_encoder.inverse_transform(prediction_encoded)[0]

print(f"Predicted crop: {crop_name}")
print(f"Encoded value: {prediction_encoded[0]}")

