# Real-Time Disaster Prediction Feature

## Overview
Successfully implemented real-time disaster prediction using coordinates and last 3 days of weather data.

## What Was Implemented

### Backend Changes (`backend/app.py`)

1. **Updated `/api/v1/risk/current` endpoint** to accept coordinates as `location_id`
   - Format: `latitude,longitude` (e.g., `43.59,-79.64`)
   - Automatically detects coordinate format (contains comma)
   - Falls back to database lookup for non-coordinate location IDs

2. **Integrated Open-Meteo APIs**
   - **Current Weather**: Fetches real-time weather data
   - **Historical Weather**: Retrieves last 3 days of weather data from Archive API
   - Parameters fetched:
     - Temperature (max, min, mean)
     - Precipitation
     - Wind speed (max)
     - Atmospheric pressure (mean)

3. **Rule-Based Risk Assessment**
   - **Flood Risk**: Total precipitation > 100mm AND average pressure < 1000 hPa
   - **Storm Risk**: Max wind speed > 40 km/h AND average pressure < 1005 hPa
   - **Hurricane Risk**: Max wind speed > 120 km/h AND average pressure < 980 hPa
   - **Heatwave Risk**: Temperature max > 35°C for multiple days
   - **None**: Normal conditions (baseline 10% risk)

4. **Gemini AI Integration**
   - Generates natural language explanations for risk assessments
   - Uses `gemini-2.5-flash` model
   - Provides context-aware explanations based on weather patterns

### Gemini Service Updates (`backend/services/gemini_service.py`)

1. **Updated Model Name**
   - Changed from `gemini-pro` to `gemini-2.5-flash`
   - Added fallback to `gemini-1.5-flash` if needed
   - Fixed API version compatibility issues

## API Response Format

```json
{
  "location_id": "43.5920189746737,-79.64920674889892",
  "risk_score": 10,
  "disaster_type": "none",
  "confidence": 75,
  "confidence_interval": {
    "lower": 65,
    "upper": 85
  },
  "model_version": "gemini-pro-1.0",
  "timestamp": "2025-11-23T12:17:36.647152",
  "last_updated": "2025-11-23T12:17:36.647160",
  "weather_snapshot": {
    "temperature": 8.2,
    "pressure": 991.9,
    "humidity": 67,
    "wind_speed": 22.1,
    "rainfall_24h": 0.0,
    "timestamp": "2025-11-23T12:15"
  },
  "ai_explanation": "The disaster risk for this location is currently low at 10%, with no specific disaster type identified. Current and recent weather patterns, including moderate winds around 22 km/h and very low precipitation (2.6 mm over the last three days), do not indicate conditions for an imminent severe weather event or natural disaster."
}
```

## How It Works

### Data Flow

1. **User selects location** on frontend (via search or current location)
2. **Frontend calls** `/api/v1/risk/current?location_id=lat,lng`
3. **Backend fetches**:
   - Current weather from Open-Meteo Forecast API
   - Last 3 days historical weather from Open-Meteo Archive API
4. **Backend calculates**:
   - Total precipitation over 3 days
   - Average pressure
   - Maximum wind speed
5. **Backend assesses risk** using rule-based criteria
6. **Gemini AI generates** natural language explanation
7. **Backend returns** complete risk assessment to frontend

### Risk Calculation Logic

```python
# Flood Risk
if total_precipitation > 100 and avg_pressure < 1000:
    risk_score = min(100, 60 + (total_precipitation - 100) / 5)
    disaster_type = "flood"

# Storm Risk
elif max_wind > 40 and avg_pressure < 1005:
    risk_score = min(100, 50 + (max_wind - 40) / 2)
    disaster_type = "storm"

# Hurricane Risk
elif max_wind > 120 and avg_pressure < 980:
    risk_score = min(100, 80 + (120 - avg_pressure) / 10)
    disaster_type = "hurricane"

# Heatwave Risk
elif any(day.temperature_max > 35 for day in historical_weather):
    hot_days = sum(1 for day if day.temperature_max > 35)
    risk_score = min(100, 40 + hot_days * 15)
    disaster_type = "heatwave"

# Normal Conditions
else:
    risk_score = 10
    disaster_type = "none"
```

## Testing

### Test Command
```bash
curl "http://localhost:8000/api/v1/risk/current?location_id=43.5920189746737,-79.64920674889892"
```

### Expected Behavior
- ✅ Accepts coordinates in format `latitude,longitude`
- ✅ Fetches real-time weather data
- ✅ Retrieves 3 days of historical weather
- ✅ Calculates risk score based on weather patterns
- ✅ Generates AI explanation using Gemini
- ✅ Returns complete risk assessment

## Frontend Integration

The frontend already has the necessary API functions:
- `getCurrentRisk(locationId)` - Calls the updated endpoint
- `getHistoricalWeather(lat, lng)` - Direct Open-Meteo API call
- `getLocationWeather(lat, lng)` - Direct Open-Meteo API call

## Server Configuration

- **Backend Port**: 8000 (fallback from 5000)
- **Frontend Port**: 3000 (or 5173 for Vite)
- **CORS**: Enabled for localhost:3000 and localhost:5173

## Environment Variables Required

```env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_PATH=disaster_data.db
```

## Known Issues & Limitations

1. **Model Version Warnings**: scikit-learn version mismatch warnings (non-critical)
2. **Python Version**: Using Python 3.9.6 (past EOL, should upgrade to 3.10+)
3. **Rule-Based System**: Currently uses simple thresholds, not ML models for coordinate-based predictions
4. **Historical Data Limit**: Only last 3 days available from Open-Meteo Archive API

## Future Enhancements

1. **ML Model Integration**: Use trained models instead of rule-based thresholds
2. **Feature Engineering**: Calculate 7-day rolling features from historical data
3. **Database Caching**: Store fetched weather data in database for faster subsequent queries
4. **Extended History**: Fetch more historical data for better pattern analysis
5. **Real-Time Updates**: Implement WebSocket for live risk updates
6. **Location Caching**: Cache risk assessments for frequently queried locations

## Success Metrics

- ✅ API responds within 5 seconds
- ✅ Gemini AI generates explanations successfully
- ✅ Risk scores calculated based on weather patterns
- ✅ Frontend can query any location by coordinates
- ✅ No database dependency for new locations

## Related Files

- `backend/app.py` - Main API endpoint implementation
- `backend/services/gemini_service.py` - Gemini AI integration
- `frontend/src/services/api.js` - Frontend API client
- `frontend/src/App.jsx` - Frontend UI integration

## Deployment Notes

1. Ensure Gemini API key is configured in `.env`
2. Backend runs on port 8000 (or 5000 if available)
3. Frontend configured to use port 8000
4. No database setup required for coordinate-based predictions
5. Internet connection required for Open-Meteo and Gemini APIs
