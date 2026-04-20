"""
Weather API Integration
========================

This script integrates with OpenWeatherMap API to fetch weather data
for the Smart Farming ML project. It retrieves temperature, humidity,
and rainfall information for any city.

Author: ML Engineer
Date: 2025
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# OpenWeatherMap API configuration
API_BASE_URL = "https://api.openweathermap.org/data/2.5"
API_KEY = os.getenv('OPENWEATHER_API_KEY')


def get_weather(city_name):
    """
    Fetch current weather data for a given city from OpenWeatherMap API.
    
    Parameters:
    -----------
    city_name : str
        Name of the city to get weather for
    
    Returns:
    --------
    dict
        Dictionary with keys: 'temperature', 'humidity', 'rainfall', 'status'
        If successful: contains weather data
        If error: contains error message in 'status' and default values
    """
    if not API_KEY:
        return {
            'temperature': None,
            'humidity': None,
            'rainfall': None,
            'status': 'error',
            'message': 'OPENWEATHER_API_KEY not found in .env file. Please add your API key.'
        }
    
    # Prepare API request for current weather
    current_weather_url = f"{API_BASE_URL}/weather"
    params = {
        'q': city_name,
        'appid': API_KEY,
        'units': 'metric'  # Use metric units (Celsius, meters, etc.)
    }
    
    try:
        # Fetch current weather data
        response = requests.get(current_weather_url, params=params, timeout=10)
        
        # Check for API errors before raising
        if response.status_code == 401:
            raise requests.exceptions.HTTPError(f"401 Unauthorized - Invalid API Key", response=response)
        elif response.status_code == 404:
            raise requests.exceptions.HTTPError(f"404 Not Found - City not found", response=response)
        
        response.raise_for_status()  # Raises an HTTPError for other bad responses
        
        data = response.json()
        
        # Extract temperature and humidity from current weather
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        
        # Get rainfall from current weather (if available)
        # Note: OpenWeatherMap current weather API doesn't always include rain
        rainfall = 0  # Default value
        if 'rain' in data:
            if '1h' in data['rain']:
                rainfall = data['rain']['1h']  # Rainfall in last hour (mm)
            elif '3h' in data['rain']:
                rainfall = data['rain']['3h'] / 3  # Approximate hourly rate
        
        # If no rain in current weather, try to get from forecast
        if rainfall == 0:
            rainfall = get_forecast_rainfall(city_name)
        
        return {
            'temperature': round(temperature, 2),
            'humidity': round(humidity, 2),
            'rainfall': round(rainfall, 2),
            'status': 'success',
            'city': data.get('name', city_name),
            'description': data['weather'][0]['description']
        }
    
    except requests.exceptions.HTTPError as e:
        # Handle HTTP errors (401, 404, etc.)
        status_code = e.response.status_code if hasattr(e, 'response') else None
        
        if status_code == 401:
            return {
                'temperature': None,
                'humidity': None,
                'rainfall': None,
                'status': 'error',
                'message': 'Invalid API key. Please check your OPENWEATHER_API_KEY in the .env file.\nGet a free API key from: https://openweathermap.org/api'
            }
        elif status_code == 404:
            return {
                'temperature': None,
                'humidity': None,
                'rainfall': None,
                'status': 'error',
                'message': f'City "{city_name}" not found. Please check the city name.'
            }
        else:
            return {
                'temperature': None,
                'humidity': None,
                'rainfall': None,
                'status': 'error',
                'message': f'API request failed with status {status_code}: {str(e)}'
            }
    
    except requests.exceptions.RequestException as e:
        # Handle network errors, timeout, etc.
        return {
            'temperature': None,
            'humidity': None,
            'rainfall': None,
            'status': 'error',
            'message': f'Network error: {str(e)}\nPlease check your internet connection.'
        }
    
    except KeyError as e:
        # Handle missing data in API response
        return {
            'temperature': None,
            'humidity': None,
            'rainfall': None,
            'status': 'error',
            'message': f'Unexpected API response format: {str(e)}'
        }
    
    except Exception as e:
        # Handle any other errors
        return {
            'temperature': None,
            'humidity': None,
            'rainfall': None,
            'status': 'error',
            'message': f'Unexpected error: {str(e)}'
        }


def get_forecast_rainfall(city_name):
    """
    Get average hourly rainfall from 5-day forecast.
    
    Parameters:
    -----------
    city_name : str
        Name of the city
    
    Returns:
    --------
    float
        Average hourly rainfall in mm (0 if unavailable)
    """
    if not API_KEY:
        return 0
    
    forecast_url = f"{API_BASE_URL}/forecast"
    params = {
        'q': city_name,
        'appid': API_KEY,
        'units': 'metric'
    }
    
    try:
        response = requests.get(forecast_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Calculate average rainfall from forecast list
        total_rainfall = 0
        count = 0
        
        if 'list' in data:
            for forecast in data['list']:
                if 'rain' in forecast:
                    if '3h' in forecast['rain']:
                        total_rainfall += forecast['rain']['3h']
                        count += 1
        
        # Return average hourly rainfall
        if count > 0:
            return (total_rainfall / count) / 3  # Convert 3h to hourly average
        return 0
    
    except Exception:
        # If forecast fails, return 0 (no rainfall data available)
        return 0


def format_weather_output(weather_data):
    """
    Format weather data for neat display.
    
    Parameters:
    -----------
    weather_data : dict
        Weather data dictionary from get_weather()
    
    Returns:
    --------
    str
        Formatted string for display
    """
    if weather_data.get('status') == 'success':
        output = f"""
{'='*60}
WEATHER DATA - {weather_data.get('city', 'Unknown City').upper()}
{'='*60}
Temperature: {weather_data['temperature']}°C
Humidity:    {weather_data['humidity']}%
Rainfall:    {weather_data['rainfall']} mm
Condition:   {weather_data.get('description', 'N/A').title()}
{'='*60}
"""
        return output
    else:
        output = f"""
{'='*60}
ERROR RETRIEVING WEATHER DATA
{'='*60}
{weather_data.get('message', 'Unknown error occurred')}
{'='*60}
"""
        return output


if __name__ == "__main__":
    """
    Test the weather API function with a sample city.
    """
    print("\n" + "="*60)
    print("WEATHER API TEST - SMART FARMING SYSTEM")
    print("="*60)
    
    # Check if API key is configured
    if not API_KEY:
        print("\n[WARNING] OPENWEATHER_API_KEY not found!")
        print("Please create a .env file in the project root with:")
        print("  OPENWEATHER_API_KEY=your_api_key_here")
        print("\nGet your free API key from: https://openweathermap.org/api")
        print("="*60 + "\n")
    else:
        print(f"\n[OK] API Key found: {API_KEY[:10]}...{API_KEY[-4:]}")
    
    # Test with Varanasi
    test_city = "Varanasi"
    print(f"\nFetching weather data for: {test_city}")
    print("-" * 60)
    
    weather_data = get_weather(test_city)
    print(format_weather_output(weather_data))
    
    # Additional test with error handling example
    print("\nTesting with invalid city name...")
    print("-" * 60)
    invalid_weather = get_weather("InvalidCityName12345")
    print(format_weather_output(invalid_weather))
    
    print("\n" + "="*60)
    print("API TEST COMPLETE")
    print("="*60 + "\n")

