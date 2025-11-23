# Location Search Integration with Open-Meteo API

## Overview
Successfully integrated Open-Meteo Geocoding and Weather APIs to enable location search functionality in the Disaster Early Warning System.

## What Was Implemented

### Direct Open-Meteo API Integration

The frontend now calls Open-Meteo APIs directly, bypassing the backend for location search and weather data. This reduces latency and simplifies the architecture.

#### 1. Location Search (Geocoding API)
- **API:** `https://geocoding-api.open-meteo.com/v1/search`
- **Parameters:**
  - `name` (required): Location name to search
  - `count` (optional, default: 10): Maximum number of results
  - `language` (optional): Language code (default: en)
  - `format` (optional): Response format (default: json)
- **Returns:** List of matching locations with:
  - Name, coordinates (lat/lng)
  - Country, country code
  - Administrative areas (state/province, county/district)
  - Timezone, population
  - Elevation, feature code

**Example:**
```bash
curl "https://geocoding-api.open-meteo.com/v1/search?name=London&count=3&language=en&format=json"
```

#### 2. Weather Forecast API
- **API:** `https://api.open-meteo.com/v1/forecast`
- **Parameters:**
  - `latitude` (required): Latitude coordinate (-90 to 90)
  - `longitude` (required): Longitude coordinate (-180 to 180)
  - `current` (required): Current weather variables (comma-separated)
  - `timezone` (optional): Timezone (default: auto)
- **Current Weather Variables:**
  - `temperature_2m` - Temperature at 2 meters (°C)
  - `relative_humidity_2m` - Relative humidity (%)
  - `precipitation` - Precipitation (mm)
  - `surface_pressure` - Surface pressure (hPa)
  - `wind_speed_10m` - Wind speed at 10 meters (km/h)
  - `wind_direction_10m` - Wind direction (°)

**Example:**
```bash
curl "https://api.open-meteo.com/v1/forecast?latitude=51.50853&longitude=-0.12574&current=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m&timezone=auto"
```

### Backend API Endpoints (Optional)

The backend still provides proxy endpoints if needed:
- `GET /api/v1/location/search` - Proxies to Open-Meteo Geocoding API
- `GET /api/v1/location/weather` - Proxies to Open-Meteo Weather API

### Frontend Integration

#### Updated Files

1. **frontend/src/services/api.js**
   - Added `searchLocation(query, count)` function
   - Added `getLocationWeather(latitude, longitude)` function
   - Updated API base URL to port 8000

2. **frontend/src/App.jsx**
   - Added location search state management
   - Implemented autocomplete dropdown for location search
   - Added debounced search (300ms delay)
   - Display selected location with coordinates
   - Integrated weather data fetching on location selection

#### Features

- **Autocomplete Search:** Type-ahead search with dropdown results
- **Debouncing:** Prevents excessive API calls (300ms delay)
- **Rich Location Display:** Shows city name, state/province, country, and population
- **Coordinate Display:** Shows latitude and longitude for selected location
- **Weather Integration:** Fetches real-time weather data for selected location

### Backend Changes

1. **backend/app.py**
   - Added location search and weather endpoints
   - Fixed import paths (removed `backend.` prefix)
   - Added automatic port fallback (5000 → 8000 if occupied)
   - Integrated `requests` library for API calls

2. **requirements.txt**
   - Confirmed `requests>=2.31.0` is included

## API Sources

- **Geocoding:** [Open-Meteo Geocoding API](https://open-meteo.com/en/docs/geocoding-api)
- **Weather:** [Open-Meteo Weather API](https://open-meteo.com/en/docs)

Both APIs are free, require no API key, and have generous rate limits.

## Testing

### Test File
Created `test_location_search.html` for standalone testing:
- Open in browser: `file:///path/to/test_location_search.html`
- Search for any city
- Click results to see weather data

### Manual Testing
1. Start backend: `cd backend && python3 app.py`
2. Test search: `curl "http://localhost:8000/api/v1/location/search?query=Tokyo"`
3. Test weather: `curl "http://localhost:8000/api/v1/location/weather?latitude=35.6762&longitude=139.6503"`

## Server Configuration

- **Backend Port:** 8000 (fallback from 5000)
- **Frontend Port:** 3000 (Vite default) or 5173
- **CORS:** Configured for localhost:3000 and localhost:5173

## Usage in Frontend

The frontend now calls Open-Meteo APIs directly:

```javascript
import * as api from './services/api';

// Search for locations (direct API call to Open-Meteo)
const results = await api.searchLocation('London', 10);
// Returns: { results: [...], count: 10, query: 'London' }

// Get weather for coordinates (direct API call to Open-Meteo)
const weather = await api.getLocationWeather(51.5085, -0.1257);
// Returns: { weather: { temperature, humidity, pressure, ... } }
```

### Direct API Calls (No Backend Required)

```javascript
// Geocoding API
const geoResponse = await fetch(
  'https://geocoding-api.open-meteo.com/v1/search?name=London&count=10&language=en&format=json'
);
const geoData = await geoResponse.json();

// Weather API
const weatherResponse = await fetch(
  'https://api.open-meteo.com/v1/forecast?latitude=51.5&longitude=-0.1&current=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m&timezone=auto'
);
const weatherData = await weatherResponse.json();
```

## Next Steps

1. **Integrate with Risk Assessment:** Use coordinates to fetch disaster risk predictions
2. **Save Locations:** Allow users to save favorite locations with coordinates
3. **Map Integration:** Display selected location on the globe visualization
4. **Historical Data:** Link location coordinates to historical disaster data
5. **Multi-language Support:** Add language parameter to geocoding API

## Notes

- Open-Meteo APIs are free and don't require authentication
- Rate limits are generous for development use
- Weather data updates every 15 minutes
- Geocoding database includes 7+ million locations worldwide
