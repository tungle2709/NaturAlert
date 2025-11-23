# Disaster Prediction Integration

## Overview
The system now automatically analyzes weather conditions (current + last 3 days) using ML models and Gemini AI to predict disaster risks whenever a user searches for a location.

## How It Works

### Automatic Prediction Flow

```
User Searches Location
         ↓
Get Coordinates
         ↓
Fetch Current Weather (Open-Meteo API)
         ↓
Fetch Historical Weather (Last 3 Days)
         ↓
Send to Backend ML Model
         ↓
ML Model Analyzes Data
         ↓
Predict Disaster Type & Risk Level
         ↓
Gemini AI Generates Explanation
         ↓
Display Results to User
```

## Disaster Types Detected

The system can predict various disaster types:
- **Flood** - Heavy rainfall, river overflow
- **Storm** - Severe weather systems
- **Hurricane** - Tropical cyclones
- **Extreme Rainfall** - Intense precipitation
- **Tsunami** - (requires seismic data integration)
- **Earthquake** - (requires seismic data integration)
- **Heat Wave** - Extreme temperatures
- **Cold Snap** - Extreme cold

## Risk Levels

### Risk Score (0-100%)
- **0-30%**: Low Risk (Green)
- **30-70%**: Moderate Risk (Yellow)
- **70-100%**: High Risk (Red)

### Display Logic
```javascript
if (riskScore < 30) {
  // Display: "LOW RISK - Normal weather conditions"
  // Show: Weather data only
} else if (riskScore < 70) {
  // Display: "MODERATE RISK - Monitor conditions"
  // Show: Disaster type + weather data + AI explanation
} else {
  // Display: "HIGH RISK - Take precautions"
  // Show: Disaster type + risk level + AI explanation + recommendations
}
```

## Implementation

### Frontend Integration

**File**: `frontend/src/App.jsx`

```javascript
const handleSelectLocation = async (locationData) => {
  // ... get coordinates ...
  
  // 1. Fetch current weather
  await api.getLocationWeather(latitude, longitude);
  
  // 2. Fetch historical weather (last 3 days)
  const historical = await api.getHistoricalWeather(latitude, longitude);
  setHistoricalWeather(historical);
  
  // 3. Automatically fetch risk assessment
  // This calls ML model + Gemini AI
  await fetchRiskData(locationId);
};
```

### Backend Processing

**File**: `backend/app.py`

```python
@app.get("/api/v1/risk/current")
def get_current_risk(location_id: str):
    # 1. Get weather data from database
    weather_data = get_weather_data(location_id)
    
    # 2. Calculate features (7-day rolling windows)
    features = calculate_features(weather_data)
    
    # 3. Run ML prediction
    prediction = ml_model.predict(features)
    risk_score = prediction['risk_score']
    disaster_type = prediction['disaster_type']
    
    # 4. Generate AI explanation with Gemini
    explanation = gemini_service.generate_explanation(
        weather_data, 
        prediction
    )
    
    # 5. Return results
    return {
        "risk_score": risk_score,
        "disaster_type": disaster_type,
        "confidence": prediction['confidence'],
        "ai_explanation": explanation,
        "weather_snapshot": current_weather
    }
```

### ML Model Features

The prediction engine analyzes:

**Current Conditions:**
- Temperature
- Pressure
- Humidity
- Wind speed
- Rainfall

**Historical Patterns (7-day window):**
- `pressure_drop_7d` - Pressure changes
- `wind_spike_max` - Maximum wind gusts
- `rain_accumulation_7d` - Total rainfall
- `humidity_trend` - Humidity changes
- `temp_deviation` - Temperature anomalies
- `pressure_velocity` - Rate of pressure change
- `wind_gust_ratio` - Wind variability

## UI Display

### No Disaster Risk (Low Risk)
```
┌─────────────────────────────────────┐
│ DISASTER RISK                       │
│ 15.2%                               │
│ LOW RISK                            │
│                                     │
│ Current Weather                     │
│ Temperature: 18°C                   │
│ Pressure: 1013 hPa                  │
│ Humidity: 65%                       │
│ Wind: 12 km/h                       │
└─────────────────────────────────────┘
```

### Disaster Risk Detected (High Risk)
```
┌─────────────────────────────────────┐
│ DISASTER RISK                       │
│ 78.5%                               │
│ HIGH RISK                           │
│ Type: FLOOD                         │
│ Confidence: 85.3%                   │
│                                     │
│ AI Explanation                      │
│ Heavy rainfall over the past 3 days │
│ combined with low pressure indicates│
│ high flood risk. Accumulated        │
│ precipitation: 125mm. River levels  │
│ likely rising. Take precautions.    │
│                                     │
│ Last 3 Days Weather                 │
│ Day 1: Rain 45mm, Wind 25km/h      │
│ Day 2: Rain 52mm, Wind 30km/h      │
│ Day 3: Rain 28mm, Wind 22km/h      │
└─────────────────────────────────────┘
```

## Gemini AI Integration

### Prompt Structure

```python
prompt = f"""
Analyze the following weather data and disaster prediction:

Current Weather:
- Temperature: {temp}°C
- Pressure: {pressure} hPa
- Humidity: {humidity}%
- Wind Speed: {wind_speed} km/h
- Rainfall: {rainfall} mm

Historical Weather (Last 3 Days):
{historical_data}

ML Prediction:
- Risk Score: {risk_score}%
- Disaster Type: {disaster_type}
- Confidence: {confidence}%

Provide a clear, concise explanation of:
1. Why this disaster risk exists
2. What weather patterns indicate this risk
3. What precautions should be taken (if high risk)

Keep response under 150 words.
"""
```

### AI Response Examples

**Low Risk:**
```
Current weather conditions are normal with stable pressure 
and moderate temperatures. No significant weather patterns 
indicate disaster risk. Continue normal activities.
```

**High Risk - Flood:**
```
Heavy rainfall over the past 3 days (125mm accumulated) 
combined with low atmospheric pressure (985 hPa) indicates 
high flood risk. Soil saturation is likely high, and 
continued rain could cause river overflow. Avoid low-lying 
areas, monitor local alerts, and prepare emergency supplies.
```

**High Risk - Storm:**
```
Rapid pressure drop (30 hPa in 24 hours) and increasing 
wind speeds (gusts up to 85 km/h) indicate an approaching 
severe storm system. Secure outdoor items, stay indoors, 
and monitor weather updates closely.
```

## Data Flow Diagram

```
┌─────────────────┐
│  User Action    │
│  (Search City)  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Get Coordinates │
│  (Lat, Lng)     │
└────────┬────────┘
         │
         ├──────────────────────┐
         ↓                      ↓
┌─────────────────┐    ┌─────────────────┐
│ Current Weather │    │ Historical      │
│ (Open-Meteo)    │    │ Weather (3 days)│
└────────┬────────┘    └────────┬────────┘
         │                      │
         └──────────┬───────────┘
                    ↓
         ┌─────────────────────┐
         │  Backend API        │
         │  /api/v1/risk/      │
         │  current            │
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         ↓                     ↓
┌─────────────────┐   ┌─────────────────┐
│  ML Model       │   │  Gemini AI      │
│  Prediction     │   │  Explanation    │
│  Engine         │   │  Generator      │
└────────┬────────┘   └────────┬────────┘
         │                     │
         └──────────┬──────────┘
                    ↓
         ┌─────────────────────┐
         │  Risk Response      │
         │  - Risk Score       │
         │  - Disaster Type    │
         │  - Confidence       │
         │  - AI Explanation   │
         └──────────┬──────────┘
                    ↓
         ┌─────────────────────┐
         │  Display to User    │
         └─────────────────────┘
```

## Starting the System

### 1. Start Backend Server

```bash
cd backend
python3 app.py
```

Server starts on: `http://localhost:8000`

### 2. Start Frontend

```bash
cd frontend
npm run dev
```

Frontend starts on: `http://localhost:5173`

### 3. Test the System

1. Open `http://localhost:5173`
2. Search for a location (e.g., "London")
3. Select from dropdown
4. System automatically:
   - Fetches current weather
   - Fetches historical weather (3 days)
   - Runs ML prediction
   - Generates AI explanation
   - Displays results

## API Endpoints

### Get Risk Assessment
```
GET /api/v1/risk/current?location_id={lat},{lng}
```

**Response:**
```json
{
  "location_id": "51.5,-0.1",
  "risk_score": 78.5,
  "disaster_type": "flood",
  "confidence": 85.3,
  "confidence_interval": {
    "lower": 72.1,
    "upper": 84.9
  },
  "model_version": "v1.0",
  "timestamp": "2024-11-23T10:30:00",
  "last_updated": "2024-11-23T10:30:00",
  "weather_snapshot": {
    "temperature": 12.5,
    "pressure": 985.2,
    "humidity": 88,
    "wind_speed": 25,
    "rainfall_24h": 45.2,
    "timestamp": "2024-11-23T10:30:00"
  },
  "ai_explanation": "Heavy rainfall over the past 3 days..."
}
```

## Configuration

### Environment Variables

**Backend** (`.env`):
```bash
GEMINI_API_KEY=your_api_key_here
DATABASE_PATH=disaster_data.db
MODEL_PATH=models/
```

### Model Files Required

```
models/
├── disaster_prediction_model.pkl  # Binary classifier
└── disaster_type_model.pkl        # Multi-class classifier
```

## Error Handling

### No Disaster Risk
```javascript
if (riskScore < 30) {
  // Show only weather conditions
  // No disaster warning needed
}
```

### Model Not Available
```javascript
if (!prediction_engine) {
  showMessage("! Prediction service unavailable");
  // Show weather data only
}
```

### API Errors
```javascript
try {
  await fetchRiskData(locationId);
} catch (error) {
  showMessage("! Failed to fetch risk assessment");
  // Continue showing weather data
}
```

## Testing

### Test Locations

**High Risk Areas (for testing):**
- Coastal regions during storm season
- River valleys during heavy rain
- Areas with recent extreme weather

**Normal Risk Areas:**
- Inland cities with stable weather
- Regions with moderate climate

### Manual Testing

```bash
# Test risk assessment API
curl "http://localhost:8000/api/v1/risk/current?location_id=51.5,-0.1"

# Test with coordinates
curl "http://localhost:8000/api/v1/risk/current?location_id=40.7,-74.0"
```

## Future Enhancements

### 1. Real-Time Seismic Data
```python
# Integrate earthquake/tsunami detection
seismic_data = get_seismic_data(location)
if seismic_data.magnitude > 5.0:
    disaster_type = "earthquake"
    risk_score = calculate_earthquake_risk(seismic_data)
```

### 2. Multi-Disaster Detection
```python
# Detect multiple concurrent risks
risks = {
    "flood": 75.2,
    "storm": 62.8,
    "landslide": 45.3
}
primary_risk = max(risks, key=risks.get)
```

### 3. Predictive Forecasting
```python
# Predict risk for next 24-72 hours
forecast_risk = predict_future_risk(
    current_weather,
    historical_patterns,
    forecast_data
)
```

### 4. Alert Notifications
```python
# Send alerts when risk exceeds threshold
if risk_score > 70:
    send_alert(user_email, disaster_type, risk_score)
```

## Summary

The system now provides comprehensive disaster prediction by:

✅ **Automatically analyzing** weather data when location is selected
✅ **Using ML models** to predict disaster type and risk level
✅ **Leveraging Gemini AI** to explain predictions in natural language
✅ **Displaying results** with clear risk levels and recommendations
✅ **Showing historical context** with 3-day weather patterns
✅ **Supporting multiple disaster types** (flood, storm, hurricane, etc.)

The integration is seamless - users simply search for a location and get instant disaster risk assessment!
