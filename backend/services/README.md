# Backend Services

This directory contains the core backend services for the Disaster Early Warning System.

## Services Overview

### 1. Prediction Engine (`prediction_engine.py`)

The Prediction Engine is responsible for generating disaster risk predictions using trained ML models.

**Key Features:**
- Loads trained scikit-learn models (binary classification and disaster type)
- Fetches weather data from SQLite database
- Calculates engineered features from raw weather data
- Runs ML predictions and generates risk scores
- Integrates with Gemini AI for natural language explanations
- Stores prediction logs in database

**Usage Example:**
```python
from backend.services.prediction_engine import PredictionEngine

# Initialize engine
engine = PredictionEngine(
    db_path='disaster_data.db',
    models_dir='models',
    use_gemini=True  # Enable AI explanations
)

# Generate prediction
prediction = engine.get_current_prediction(location_id='default')

print(f"Risk Score: {prediction['risk_score']:.1f}%")
print(f"Disaster Type: {prediction['disaster_type']}")
print(f"AI Explanation: {prediction['ai_explanation']}")
```

**Calculated Features:**
- `temperature`: Current temperature (°C)
- `pressure`: Current atmospheric pressure (hPa)
- `wind_speed`: Current wind speed (mph)
- `humidity`: Current humidity (%)
- `pressure_drop_7d`: Maximum pressure drop over 7 days (hPa)
- `wind_spike_max`: Maximum wind speed in 7-day window (mph)
- `humidity_trend`: Linear regression slope of humidity
- `temp_deviation`: Temperature standard deviation
- `pressure_velocity`: Rate of pressure change (hPa/hour)
- `wind_gust_ratio`: Ratio of max to mean wind speed

### 2. Gemini Service (`gemini_service.py`)

The Gemini Service provides AI-powered natural language generation using Google's Gemini API.

**Key Features:**
- Generate risk explanations from weather data and predictions
- Interactive chat functionality with context awareness
- Explain ML feature importance in simple terms
- Generate concise, actionable alert messages
- Robust error handling with fallback messages

**Setup:**

1. Install the required package:
```bash
pip install google-generativeai
```

2. Set your API key in `.env`:
```bash
GEMINI_API_KEY=your_api_key_here
```

3. Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

**Usage Example:**
```python
from backend.services.gemini_service import get_gemini_service
import pandas as pd

# Get service instance
gemini = get_gemini_service()

# Generate explanation
weather_data = pd.DataFrame({...})
prediction = {'risk_score': 75.0, 'disaster_type': 'storm', 'confidence': 80.0}

explanation = gemini.generate_explanation(weather_data, prediction)
print(explanation)

# Chat with AI
response = gemini.chat_response(
    message="What should I do to prepare?",
    context={'risk_score': 75.0, 'location': 'Da Nang'}
)
print(response)

# Explain features
feature_importance = {
    'pressure_drop_7d': 28.5,
    'rain_accumulation_7d': 22.3,
    'wind_spike_max': 18.7
}
explanation = gemini.generate_feature_importance_explanation(feature_importance)
print(explanation)

# Generate alert
alert = gemini.generate_alert_message(prediction)
print(alert)
```

**Methods:**

#### `generate_explanation(weather_data, prediction)`
Generates a 2-3 sentence explanation of why the risk is at the current level.

**Args:**
- `weather_data`: pandas DataFrame with recent weather observations
- `prediction`: Dict with `risk_score`, `disaster_type`, `confidence`

**Returns:** String with natural language explanation

#### `chat_response(message, context, conversation_history)`
Generates contextual chat responses for user questions.

**Args:**
- `message`: User's question
- `context`: Dict with current conditions (location, risk_score, etc.)
- `conversation_history`: List of previous messages (optional)

**Returns:** String with AI response

#### `generate_feature_importance_explanation(feature_importance)`
Explains ML feature importance in simple terms.

**Args:**
- `feature_importance`: Dict mapping feature names to importance percentages

**Returns:** String with explanations for each feature

#### `generate_alert_message(prediction)`
Creates concise, actionable alert messages.

**Args:**
- `prediction`: Dict with `risk_score`, `disaster_type`, `confidence`

**Returns:** String with 1-2 sentence alert message

**Error Handling:**

The Gemini Service includes robust error handling:
- Graceful degradation if API key is missing
- Fallback messages if API calls fail
- Safe initialization that won't crash the application
- Detailed error logging for debugging

**Testing:**

Run the example script to test all Gemini features:
```bash
python examples/gemini_service_example.py
```

## Integration

Both services are designed to work together:

1. **Prediction Engine** generates ML predictions
2. **Gemini Service** adds natural language explanations
3. Results are stored in database with both numerical and textual insights

This provides users with both quantitative risk scores and qualitative explanations they can understand.

## Environment Variables

Required environment variables (set in `.env`):

```bash
# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Database
DATABASE_PATH=disaster_data.db

# Models
MODEL_VERSION=1.0.0
BINARY_MODEL_PATH=models/disaster_prediction_model.pkl
TYPE_MODEL_PATH=models/disaster_type_model.pkl
```

## Dependencies

```bash
# Core ML
scikit-learn>=1.3.0
joblib>=1.3.0
pandas>=2.1.0
numpy>=1.24.0

# AI Integration
google-generativeai>=0.3.0

# Database
sqlalchemy>=2.0.0

# Utilities
python-dotenv>=1.0.0
```

## Development

### Running Examples

```bash
# Test prediction engine
python examples/prediction_engine_example.py

# Test Gemini service
python examples/gemini_service_example.py
```

### Adding New Features

To add new Gemini-powered features:

1. Add a new method to `GeminiService` class
2. Create a prompt template for your use case
3. Add error handling with fallback message
4. Update this README with usage examples
5. Add tests to verify functionality

### Troubleshooting

**"GEMINI_API_KEY not found"**
- Make sure `.env` file exists in project root
- Verify the key is set correctly: `GEMINI_API_KEY=your_key`
- Check that `python-dotenv` is installed

**"google-generativeai package not installed"**
- Install it: `pip install google-generativeai`
- Verify installation: `pip show google-generativeai`

**"Failed to initialize Gemini API"**
- Check your API key is valid
- Verify you have internet connection
- Check API quota limits in Google AI Studio

**"Insufficient weather data"**
- Ensure database has at least 2 weather records
- Run data preprocessing notebooks to populate database
- Check location_id matches database records

## Future Enhancements

Planned improvements:
- [ ] Conversation history persistence in database
- [ ] Multi-language support for explanations
- [ ] Custom prompt templates per user preference
- [ ] Caching of AI responses to reduce API calls
- [ ] A/B testing of different prompt strategies
- [ ] Integration with other AI models (Claude, GPT-4)
