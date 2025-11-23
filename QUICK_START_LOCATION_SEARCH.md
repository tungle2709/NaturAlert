# Quick Start: Location Search Feature

## Direct API Access (No Backend Required!)

The frontend now calls Open-Meteo APIs directly. No backend server needed for location search!

## Test the API

### 1. Search for a Location (Direct Open-Meteo API)

```bash
curl "https://geocoding-api.open-meteo.com/v1/search?name=Tokyo&count=5&language=en&format=json"
```

**Response:**
```json
{
  "results": [
    {
      "id": 1850144,
      "name": "Tokyo",
      "latitude": 35.6895,
      "longitude": 139.69171,
      "country": "Japan",
      "country_code": "JP",
      "admin1": "Tokyo",
      "timezone": "Asia/Tokyo",
      "population": 8336599,
      "display_name": "Tokyo, Tokyo, Japan"
    }
  ],
  "count": 1,
  "query": "Tokyo",
  "timestamp": "2025-11-23T..."
}
```

### 2. Get Weather for Location (Direct Open-Meteo API)

```bash
curl "https://api.open-meteo.com/v1/forecast?latitude=35.6895&longitude=139.69171&current=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m&timezone=auto"
```

**Response:**
```json
{
  "latitude": 35.69,
  "longitude": 139.69,
  "timezone": "Asia/Tokyo",
  "current": {
    "time": "2025-11-23T12:00",
    "temperature_2m": 15.2,
    "relative_humidity_2m": 65,
    "precipitation": 0.0,
    "surface_pressure": 1013.5,
    "wind_speed_10m": 12.5,
    "wind_direction_10m": 180
  }
}
```

## Frontend Usage (Direct API Calls)

### In Your React Component

```javascript
import * as api from './services/api';

// Search for locations (calls Open-Meteo directly)
const handleSearch = async (query) => {
  const results = await api.searchLocation(query, 10);
  console.log(results.results);
  // Results include: name, latitude, longitude, country, admin1, population, etc.
};

// Get weather (calls Open-Meteo directly)
const handleGetWeather = async (lat, lng) => {
  const weather = await api.getLocationWeather(lat, lng);
  console.log(weather.weather);
  // Weather includes: temperature, humidity, pressure, wind_speed, etc.
};
```

### Direct Fetch (Without API Service)

```javascript
// Geocoding
const searchResponse = await fetch(
  `https://geocoding-api.open-meteo.com/v1/search?name=${query}&count=10&language=en&format=json`
);
const locations = await searchResponse.json();

// Weather
const weatherResponse = await fetch(
  `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lng}&current=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m&timezone=auto`
);
const weather = await weatherResponse.json();
```

### Example: Autocomplete Search

```javascript
const [searchQuery, setSearchQuery] = useState('');
const [locations, setLocations] = useState([]);

// Debounced search
useEffect(() => {
  if (searchQuery.length < 2) return;
  
  const timer = setTimeout(async () => {
    const results = await api.searchLocation(searchQuery);
    setLocations(results.results);
  }, 300);
  
  return () => clearTimeout(timer);
}, [searchQuery]);
```

## Test in Browser

1. Open `test_location_search.html` in your browser
2. Type a city name (e.g., "London", "Paris", "New York")
3. Click on a result to see weather data

## API Documentation

- **Open-Meteo Geocoding:** https://open-meteo.com/en/docs/geocoding-api
- **Open-Meteo Weather:** https://open-meteo.com/en/docs
- **Backend API (optional):** http://localhost:8000/docs (Swagger UI)

## Common Locations to Test

- **London:** `query=London`
- **Tokyo:** `query=Tokyo`
- **New York:** `query=New York`
- **Paris:** `query=Paris`
- **Sydney:** `query=Sydney`
- **Mumbai:** `query=Mumbai`
- **São Paulo:** `query=Sao Paulo`

## Troubleshooting

### CORS Errors
Open-Meteo APIs support CORS by default, so no CORS issues when calling from the browser!

### Backend Not Required
The location search and weather features work entirely client-side. The backend is only needed for:
- Disaster risk predictions
- Gemini AI features
- Historical disaster data

### No Results
- Check spelling
- Try broader search terms (e.g., "Lond" instead of "London, UK")
- Minimum 2 characters required

## Rate Limits

Open-Meteo APIs are free with generous limits:
- **Geocoding:** 10,000 requests/day
- **Weather:** 10,000 requests/day

For production, consider caching results.
