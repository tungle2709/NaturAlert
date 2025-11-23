"""
Example: Using the Prediction Engine

This script demonstrates how to use the PredictionEngine to generate
disaster risk predictions from weather data.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.services.prediction_engine import PredictionEngine

def main():
    """Example usage of the Prediction Engine"""
    
    print("=" * 70)
    print("Disaster Early Warning System - Prediction Engine Example")
    print("=" * 70)
    
    # Initialize the prediction engine
    print("\n1. Initializing Prediction Engine...")
    
    # Get paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    db_path = os.path.join(project_root, 'disaster_data.db')
    models_dir = os.path.join(project_root, 'models')
    
    try:
        engine = PredictionEngine(
            db_path=db_path,
            models_dir=models_dir
        )
        print("   ✓ Prediction engine initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return
    
    # Generate prediction for default location
    print("\n2. Generating Prediction...")
    try:
        prediction = engine.get_current_prediction(location_id='default')
        print("   ✓ Prediction generated successfully")
    except Exception as e:
        print(f"   ✗ Prediction failed: {e}")
        return
    
    # Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    
    print(f"\nLocation: {prediction['location_id']}")
    print(f"Timestamp: {prediction['last_updated']}")
    
    print(f"\n📊 Risk Assessment:")
    print(f"   Risk Score: {prediction['risk_score']:.2f}%")
    print(f"   Disaster Type: {prediction['disaster_type']}")
    print(f"   Confidence: {prediction['confidence']:.2f}%")
    print(f"   Confidence Interval: [{prediction['confidence_interval']['lower']:.2f}%, "
          f"{prediction['confidence_interval']['upper']:.2f}%]")
    
    # Risk level interpretation
    risk_score = prediction['risk_score']
    if risk_score >= 70:
        risk_level = "🔴 HIGH"
    elif risk_score >= 50:
        risk_level = "🟡 MODERATE"
    else:
        risk_level = "🟢 LOW"
    
    print(f"\n   Risk Level: {risk_level}")
    
    print(f"\n🌤️  Current Weather Conditions:")
    ws = prediction['weather_snapshot']
    print(f"   Temperature: {ws['temperature']:.1f}°C")
    print(f"   Pressure: {ws['pressure']:.1f} hPa")
    print(f"   Humidity: {ws['humidity']:.1f}%")
    print(f"   Wind Speed: {ws['wind_speed']:.1f} mph")
    print(f"   24h Rainfall: {ws['rainfall_24h']:.1f} mm")
    
    print(f"\n🔧 Model Information:")
    print(f"   Model Version: {prediction['model_version']}")
    
    # Display AI explanation if available
    if prediction.get('ai_explanation'):
        print(f"\n🤖 AI Explanation:")
        print(f"   {prediction['ai_explanation']}")
    else:
        print(f"\n🤖 AI Explanation: Not available (Gemini API not configured)")
    
    print("\n" + "=" * 70)
    print("✓ Example completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
