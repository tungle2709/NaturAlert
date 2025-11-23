# Quick Reference - Globe Location Search

## 🚀 Quick Start

### Test the Feature
```bash
# Open standalone test
open test_globe_movement.html

# Or run the full app
cd frontend && npm run dev
```

## 🔍 Search for Locations

### In the App
1. Type city name in search box
2. Select from dropdown
3. Globe moves to location automatically
4. Red marker appears

### Popular Test Locations
- London
- Tokyo
- New York
- Paris
- Sydney
- Mumbai
- São Paulo

## 📡 API Endpoints

### Geocoding (Location Search)
```
https://geocoding-api.open-meteo.com/v1/search?name=London&count=10&language=en&format=json
```

### Weather Data
```
https://api.open-meteo.com/v1/forecast?latitude=51.5&longitude=-0.1&current=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m&timezone=auto
```

## 💻 Code Snippets

### Search Location
```javascript
import * as api from './services/api';

const results = await api.searchLocation('Tokyo', 10);
console.log(results.results);
```

### Get Weather
```javascript
const weather = await api.getLocationWeather(35.6895, 139.6917);
console.log(weather.weather);
```

### Move Globe
```javascript
globeInstance.current.pointOfView({
  lat: 51.5074,
  lng: -0.1278,
  altitude: 1.5
}, 1000);
```

### Add Marker
```javascript
globeInstance.current.pointsData([{
  lat: 51.5074,
  lng: -0.1278,
  name: 'London',
  color: '#ff0000'
}]);
```

## ⚙️ Configuration

### Animation Speed
- Fast: `500ms`
- Normal: `1000ms` (default)
- Slow: `2000ms`

### Camera Altitude
- Very close: `1.0`
- Close: `1.5` (default)
- Medium: `2.0`
- Far: `2.5`

### Marker Size
- Small: `0.8`
- Medium: `1.2` (default)
- Large: `1.8`

## 📁 Files

### Main Files
- `frontend/src/App.jsx` - Main app with globe
- `frontend/src/services/api.js` - API service layer

### Test Files
- `test_globe_movement.html` - Globe movement test
- `test_direct_api.html` - API integration test
- `test_location_search.html` - Simple search test

### Documentation
- `GLOBE_INTEGRATION_SUMMARY.md` - Complete summary
- `GLOBE_MOVEMENT_FEATURE.md` - Feature details
- `FRONTEND_DIRECT_API_INTEGRATION.md` - API integration
- `LOCATION_SEARCH_INTEGRATION.md` - Search overview
- `QUICK_START_LOCATION_SEARCH.md` - Quick start guide

## 🐛 Troubleshooting

### Globe doesn't move
- Check console for errors
- Verify `globeInstance.current` exists
- Check coordinates are valid

### Marker doesn't appear
- Verify `pointsData` is called
- Check marker color is set
- Increase `pointRadius` size

### Search returns no results
- Check spelling
- Try broader terms
- Minimum 2 characters required

## 📊 API Limits

- **Geocoding**: 10,000 requests/day
- **Weather**: 10,000 requests/day
- **No API key required**
- **CORS enabled**

## 🎯 Features

✅ Real-time location search
✅ Autocomplete dropdown
✅ Smooth globe animation
✅ Visual markers
✅ Weather data integration
✅ No backend required
✅ Free API access

## 📞 Support

- **Globe.GL Docs**: https://github.com/vasturiano/globe.gl
- **Open-Meteo Docs**: https://open-meteo.com/en/docs
- **API Status**: https://status.open-meteo.com/
