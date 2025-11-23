"""
Example usage of the Gemini Service for the Disaster Early Warning System.

This script demonstrates how to use the GeminiService class to:
1. Generate risk explanations
2. Chat with the AI assistant
3. Explain feature importance
4. Generate alert messages

Prerequisites:
- Set GEMINI_API_KEY in .env file
- Install google-generativeai: pip install google-generativeai
"""

import sys
import os
import pandas as pd

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.services.gemini_service import GeminiService, get_gemini_service


def example_generate_explanation():
    """Example: Generate explanation for a disaster prediction."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Generate Risk Explanation")
    print("="*80)
    
    # Create sample weather data
    weather_data = pd.DataFrame({
        'timestamp': ['2024-01-15 12:00:00'],
        'temperature': [28.5],
        'pressure': [995.2],
        'humidity': [85.0],
        'wind_speed': [45.3],
        'rainfall_24h': [125.5]
    })
    
    # Sample prediction
    prediction = {
        'risk_score': 78.5,
        'disaster_type': 'storm',
        'confidence': 82.3
    }
    
    try:
        service = get_gemini_service()
        explanation = service.generate_explanation(weather_data, prediction)
        
        print("\nWeather Conditions:")
        print(f"  Temperature: {weather_data['temperature'][0]}°C")
        print(f"  Pressure: {weather_data['pressure'][0]} hPa")
        print(f"  Wind Speed: {weather_data['wind_speed'][0]} mph")
        print(f"  24h Rainfall: {weather_data['rainfall_24h'][0]} mm")
        
        print("\nPrediction:")
        print(f"  Risk Score: {prediction['risk_score']}%")
        print(f"  Disaster Type: {prediction['disaster_type']}")
        print(f"  Confidence: {prediction['confidence']}%")
        
        print("\nAI Explanation:")
        print(f"  {explanation}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Make sure GEMINI_API_KEY is set in your .env file")


def example_chat_response():
    """Example: Chat with the AI assistant."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Chat with AI Assistant")
    print("="*80)
    
    # Sample context
    context = {
        'location': 'Da Nang, Vietnam',
        'risk_score': 65.0,
        'disaster_type': 'flood',
        'temperature': 26.5,
        'pressure': 1002.3
    }
    
    # Sample conversation
    messages = [
        "What should I do to prepare for the current flood risk?",
        "How long will this weather pattern last?",
        "What are the signs that conditions are getting worse?"
    ]
    
    try:
        service = get_gemini_service()
        
        print("\nContext:")
        print(f"  Location: {context['location']}")
        print(f"  Risk Score: {context['risk_score']}%")
        print(f"  Disaster Type: {context['disaster_type']}")
        
        conversation_history = []
        
        for message in messages:
            print(f"\nUser: {message}")
            
            response = service.chat_response(
                message=message,
                context=context,
                conversation_history=conversation_history
            )
            
            print(f"Assistant: {response}")
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": message})
            conversation_history.append({"role": "assistant", "content": response})
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Make sure GEMINI_API_KEY is set in your .env file")


def example_feature_importance():
    """Example: Explain feature importance."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Feature Importance Explanation")
    print("="*80)
    
    # Sample feature importance from ML model
    feature_importance = {
        'pressure_drop_7d': 28.5,
        'rain_accumulation_7d': 22.3,
        'wind_spike_max': 18.7,
        'pressure_velocity': 12.4,
        'wind_gust_ratio': 9.2,
        'humidity_trend': 5.8,
        'temp_deviation': 3.1
    }
    
    try:
        service = get_gemini_service()
        explanation = service.generate_feature_importance_explanation(feature_importance)
        
        print("\nFeature Importance Rankings:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {importance:.1f}%")
        
        print("\nAI Explanation:")
        print(f"  {explanation}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Make sure GEMINI_API_KEY is set in your .env file")


def example_alert_message():
    """Example: Generate alert message."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Generate Alert Message")
    print("="*80)
    
    # Sample predictions with different risk levels
    predictions = [
        {
            'risk_score': 85.0,
            'disaster_type': 'flood',
            'confidence': 88.5
        },
        {
            'risk_score': 72.0,
            'disaster_type': 'storm',
            'confidence': 75.2
        },
        {
            'risk_score': 91.5,
            'disaster_type': 'hurricane',
            'confidence': 93.8
        }
    ]
    
    try:
        service = get_gemini_service()
        
        for i, prediction in enumerate(predictions, 1):
            print(f"\nPrediction {i}:")
            print(f"  Risk Score: {prediction['risk_score']}%")
            print(f"  Disaster Type: {prediction['disaster_type']}")
            print(f"  Confidence: {prediction['confidence']}%")
            
            alert_message = service.generate_alert_message(prediction)
            print(f"  Alert Message: {alert_message}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Make sure GEMINI_API_KEY is set in your .env file")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("GEMINI SERVICE EXAMPLES")
    print("Disaster Early Warning System")
    print("="*80)
    
    # Check if API key is set
    if not os.getenv('GEMINI_API_KEY'):
        print("\n⚠️  WARNING: GEMINI_API_KEY not found in environment variables")
        print("Please set it in your .env file to run these examples")
        print("\nExample .env entry:")
        print("GEMINI_API_KEY=your_api_key_here")
        return
    
    # Run examples
    example_generate_explanation()
    example_chat_response()
    example_feature_importance()
    example_alert_message()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
