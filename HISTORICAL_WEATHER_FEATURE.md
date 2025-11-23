# Historical Weather Feature

## Overview
Added automatic retrieval of historical weather data for the last 3 days whenever a user searches for a location. This provides context about recent weather patterns to help assess disaster risk.

## Features

### Automatic Historical Data Retrieval
- **Triggered On**: Location selection or "Use Current Location"
- **Data Range**: Last 3 days
- **Data Source**: Open-Meteo Historical Weather Archive API
- **Display**: Shows in dashboard below current weather

### Data Points Collected

For each of the last 3 days:
- **Temperature Max**: Highest temperature of the day (°C)
- **Temperature Min**: Lowest temperature of the day (°C)
- **Temperature Mean**: Average temperature (°C)
- **Precipitation**: Total precipitation (mm)
- **Rain**: Total rainfall (mm)
- **Wind Speed Max**: Maximum wind speed (km/h)
- **Wind Direction**: Dominant wind direction (degrees)

## Implementation

### API Function

**File**: `frontend/src/services/api.js`

```javascript
export async function getHistoricalWeather(latitude, longitude) {
  // Calculate dates for last 3 days
  const endDate = new Date();
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - 3);
  
  const response = await fetch(
    `https://archive-api.open-meteo.com/v1/archive?latitude=${latitude}&longitude=${longitude}&start_date=${formatDate(startDate)}&end_date=${formatDate(endDate)}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,rain_sum,wind_speed_10m_max,wind_direction_10m_dominant&timezone=auto`
  );
  
  // Returns formatted historical data
}
```

### Integration Points

**1. Location Selection**
```javascript
const handleSelectLocation = async (locationData) => {
  // ... existing code ...
  
  // Fetch historical weather data for last 3 days
  const historical = await api.getHistoricalWeather(
    locationData.latitude, 
    locationData.longitude
  );
  setHistoricalWeather(historical);
};
```

**2. Current Location**
```javascript
const handleUseCurrentLocation = () => {
  // ... get coordinates ...
  
  // Fetch historical weather data
  const historical = await api.getHistoricalWeather(latitude, longitude);
  setHistoricalWeather(historical);
};
```

### UI Display

**Location**: Dashboard sidebar, below AI Explanation section

**Design**:
```
┌─────────────────────────────────────┐
│ Last 3 Days Weather                 │
├─────────────────────────────────────┤
│ 2024-11-21                          │
│ Temp: 0.6°C - 3.0°C  Avg: 1.5°C   │
│ Rain: 2.5 mm          Wind: 25 km/h│
├─────────────────────────────────────┤
│ 2024-11-22                          │
│ Temp: 0.9°C - 6.3°C  Avg: 3.2°C   │
│ Rain: 0.0 mm          Wind: 18 km/h│
├─────────────────────────────────────┤
│ 2024-11-23                          │
│ Temp: 1.3°C - 14.4°C Avg: 7.8°C   │
│ Rain: 1.2 mm          Wind: 22 km/h│
└─────────────────────────────────────┘
```

## API Details

### Open-Meteo Historical Weather Archive API

**Endpoint:**
```
https://archive-api.open-meteo.com/v1/archive
```

**Parameters:**
- `latitude` - Latitude coordinate
- `longitude` - Longitude coordinate
- `start_date` - Start date (YYYY-MM-DD)
- `end_date` - End date (YYYY-MM-DD)
- `daily` - Daily weather variables (comma-separated)
- `timezone` - Timezone (auto)

**Daily Variables:**
- `temperature_2m_max` - Maximum temperature at 2m height
- `temperature_2m_min` - Minimum temperature at 2m height
- `temperature_2m_mean` - Mean temperature at 2m height
- `precipitation_sum` - Total precipitation
- `rain_sum` - Total rainfall
- `wind_speed_10m_max` - Maximum wind speed at 10m height
- `wind_direction_10m_dominant` - Dominant wind direction

**Response Format:**
```json
{
  "latitude": 51.5,
  "longitude": -0.1,
  "timezone": "Europe/London",
  "daily": {
    "time": ["2024-11-21", "2024-11-22", "2024-11-23"],
    "temperature_2m_max": [3.0, 6.3, 14.4],
    "temperature_2m_min": [-0.4, 0.9, 1.3],
    "temperature_2m_mean": [0.6, 3.2, 7.8],
    "precipitation_sum": [2.5, 0.0, 1.2],
    "rain_sum": [2.5, 0.0, 1.2],
    "wind_speed_10m_max": [25, 18, 22],
    "wind_direction_10m_dominant": [270, 180, 225]
  }
}
```

## Use Cases

### 1. Disaster Risk Assessment
Historical weather patterns help identify:
- Recent extreme weather events
- Accumulating precipitation (flood risk)
- Temperature trends (heat wave/cold snap)
- Wind pattern changes (storm risk)

### 2. Pattern Recognition
Compare last 3 days with:
- Current conditions
- Historical averages
- Disaster precursors

### 3. Context for AI Analysis
Historical data provides context for:
- Gemini AI explanations
- Risk score interpretation
- Trend analysis

## Data Flow

```
User Action (Search/Current Location)
         ↓
Get Coordinates (lat, lng)
         ↓
Calculate Date Range (today - 3 days)
         ↓
Call Open-Meteo Archive API
         ↓
Format Response Data
         ↓
Store in State (historicalWeather)
         ↓
Display in Dashboard UI
```

## Error Handling

### API Errors
```javascript
try {
  const historical = await api.getHistoricalWeather(lat, lng);
  setHistoricalWeather(historical);
} catch (error) {
  console.error('Historical weather fetch error:', error);
  // Continues without historical data
}
```

### Missing Data
- Handles null/undefined values gracefully
- Uses optional chaining: `day.temperature_max?.toFixed(1)`
- Displays "N/A" or skips missing fields

## Performance

### API Response Time
- **Average**: 100-300ms
- **Cached**: Data is cached by Open-Meteo
- **Parallel**: Fetches alongside current weather

### Data Size
- **Per Request**: ~1-2 KB
- **3 Days**: Minimal bandwidth impact
- **No Pagination**: Single request for all 3 days

## Browser Compatibility

### API Support
- ✅ All modern browsers (Fetch API)
- ✅ No API key required
- ✅ CORS enabled
- ✅ HTTPS only

### Date Handling
- Uses native JavaScript Date object
- ISO 8601 format (YYYY-MM-DD)
- Timezone-aware

## Future Enhancements

### 1. Configurable Date Range
```javascript
// Allow user to select range
getHistoricalWeather(lat, lng, days = 3)
```

### 2. Weather Charts
```javascript
// Visualize trends with charts
<LineChart data={historicalWeather} />
```

### 3. Comparison View
```javascript
// Compare with historical averages
compareWithAverage(historicalWeather, historicalAverage)
```

### 4. Export Data
```javascript
// Download as CSV/JSON
exportHistoricalData(historicalWeather)
```

### 5. Anomaly Detection
```javascript
// Highlight unusual patterns
detectAnomalies(historicalWeather)
```

## Testing

### Manual Test
```bash
# Test API directly
curl "https://archive-api.open-meteo.com/v1/archive?latitude=51.5&longitude=-0.1&start_date=2024-11-20&end_date=2024-11-23&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,rain_sum,wind_speed_10m_max,wind_direction_10m_dominant&timezone=auto"
```

### In Application
1. Search for a location (e.g., "London")
2. Select from dropdown
3. Check console for: `Historical weather data: {...}`
4. Verify "Last 3 Days Weather" section appears
5. Confirm data displays correctly

### Test Locations
- **London, UK**: `51.5, -0.1`
- **New York, US**: `40.7, -74.0`
- **Tokyo, Japan**: `35.7, 139.7`
- **Sydney, Australia**: `-33.9, 151.2`

## API Limits

### Open-Meteo Archive API
- **Rate Limit**: 10,000 requests/day (free tier)
- **Data Range**: 1940 - present
- **Update Frequency**: Daily
- **No API Key**: Required

### Recommendations
- Cache results client-side
- Debounce rapid location changes
- Consider backend caching for production

## Summary

The historical weather feature provides valuable context for disaster risk assessment by automatically retrieving and displaying weather conditions from the last 3 days whenever a user searches for a location. This helps users understand recent weather patterns and trends that may contribute to disaster risk.

**Key Benefits:**
- ✅ Automatic retrieval on location search
- ✅ No user action required
- ✅ Free API with no key needed
- ✅ Clean, integrated UI display
- ✅ Useful context for risk assessment
- ✅ Supports AI analysis and explanations
