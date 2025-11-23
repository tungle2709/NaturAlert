# Globe Movement Feature

## Overview
The globe now automatically moves to show the selected location when a user searches and selects a city from the location search dropdown.

## Features

### 🌍 Automatic Globe Navigation
- **Smooth Animation**: Globe smoothly rotates to the selected location over 1 second
- **Optimal Viewing**: Camera automatically adjusts to altitude 1.5 for best view
- **Visual Marker**: Red marker appears at the selected location
- **Location Label**: Hover over marker to see location name

### 🎯 Implementation Details

#### Globe Configuration
```javascript
Globe()
  .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
  .bumpImageUrl('//unpkg.com/three-globe/example/img/earth-topology.png')
  .backgroundImageUrl('//unpkg.com/three-globe/example/img/night-sky.png')
  .pointOfView({ altitude: 2.5 })
  .pointsData([])
  .pointAltitude(0.01)
  .pointRadius(1.2)
  .pointColor('color')
  .pointLabel('name')
  .pointsMerge(false)
```

#### Location Selection Handler
```javascript
const handleSelectLocation = async (locationData) => {
  // Move globe to selected location
  if (globeInstance.current) {
    // Animate globe to the selected location
    globeInstance.current.pointOfView({
      lat: locationData.latitude,
      lng: locationData.longitude,
      altitude: 1.5
    }, 1000); // 1000ms animation duration
    
    // Add a marker at the selected location
    globeInstance.current.pointsData([{
      lat: locationData.latitude,
      lng: locationData.longitude,
      name: locationData.display_name,
      size: 0.5,
      color: '#ff0000'
    }]);
  }
  
  // ... rest of the handler
};
```

## User Flow

1. **Search**: User types a city name in the search box
2. **Results**: Autocomplete dropdown shows matching locations
3. **Select**: User clicks on a location from the dropdown
4. **Animation**: Globe smoothly rotates to show the selected location
5. **Marker**: Red marker appears at the location
6. **Info**: Location details displayed in the sidebar

## API Integration

### Geocoding API
```javascript
// Search for locations
const response = await fetch(
  `https://geocoding-api.open-meteo.com/v1/search?name=${query}&count=10&language=en&format=json`
);
const data = await response.json();

// data.results contains:
// - name: City name
// - latitude: Latitude coordinate
// - longitude: Longitude coordinate
// - country: Country name
// - admin1: State/Province
// - population: Population count
```

### Globe.GL Methods

#### pointOfView()
Moves the camera to a specific location:
```javascript
globe.pointOfView({
  lat: 51.5074,      // Latitude
  lng: -0.1278,      // Longitude
  altitude: 1.5      // Camera altitude (1.5 = close view)
}, 1000);            // Animation duration in ms
```

#### pointsData()
Adds markers to the globe:
```javascript
globe.pointsData([
  {
    lat: 51.5074,
    lng: -0.1278,
    name: 'London, England, United Kingdom',
    color: '#ff0000'
  }
]);
```

## Configuration Options

### Animation Duration
Adjust the speed of globe rotation:
```javascript
// Fast (500ms)
globe.pointOfView({ lat, lng, altitude: 1.5 }, 500);

// Normal (1000ms) - Default
globe.pointOfView({ lat, lng, altitude: 1.5 }, 1000);

// Slow (2000ms)
globe.pointOfView({ lat, lng, altitude: 1.5 }, 2000);
```

### Camera Altitude
Control how close the camera zooms:
```javascript
// Very close
globe.pointOfView({ lat, lng, altitude: 1.0 }, 1000);

// Close (default)
globe.pointOfView({ lat, lng, altitude: 1.5 }, 1000);

// Medium
globe.pointOfView({ lat, lng, altitude: 2.0 }, 1000);

// Far
globe.pointOfView({ lat, lng, altitude: 2.5 }, 1000);
```

### Marker Appearance
Customize marker size and color:
```javascript
globe
  .pointRadius(1.2)           // Marker size
  .pointColor('color')        // Use color from data
  .pointAltitude(0.01)        // Height above surface
  .pointLabel('name');        // Tooltip text
```

## Testing

### Test File
Open `test_globe_movement.html` in your browser to test the feature:
```bash
open test_globe_movement.html
```

### Test Locations
Try searching for these cities:
- **London** - Europe
- **Tokyo** - Asia
- **New York** - North America
- **Sydney** - Australia
- **São Paulo** - South America
- **Cairo** - Africa

### Expected Behavior
1. Type city name in search box
2. See autocomplete results appear
3. Click on a result
4. Globe smoothly rotates to show the location
5. Red marker appears at the location
6. Hover over marker to see location name

## Troubleshooting

### Globe doesn't move
- Check browser console for errors
- Verify `globeInstance.current` is initialized
- Ensure coordinates are valid (-90 to 90 for lat, -180 to 180 for lng)

### Marker doesn't appear
- Check that `pointsData` is being called with valid data
- Verify marker color is set correctly
- Check `pointRadius` is large enough to be visible

### Animation is jerky
- Reduce animation duration (try 500ms)
- Check browser performance
- Ensure no other heavy operations during animation

## Future Enhancements

### Multiple Markers
Show multiple locations at once:
```javascript
globe.pointsData([
  { lat: 51.5, lng: -0.1, name: 'London', color: '#ff0000' },
  { lat: 35.7, lng: 139.7, name: 'Tokyo', color: '#00ff00' },
  { lat: 40.7, lng: -74.0, name: 'New York', color: '#0000ff' }
]);
```

### Risk-Based Colors
Color markers based on disaster risk:
```javascript
const getMarkerColor = (riskScore) => {
  if (riskScore >= 70) return '#ff0000'; // Red - High risk
  if (riskScore >= 50) return '#ffaa00'; // Orange - Medium risk
  return '#00ff00'; // Green - Low risk
};

globe.pointsData([{
  lat: location.latitude,
  lng: location.longitude,
  name: location.name,
  color: getMarkerColor(riskData.risk_score)
}]);
```

### Arcs Between Locations
Show connections between locations:
```javascript
globe.arcsData([
  {
    startLat: 51.5,
    startLng: -0.1,
    endLat: 35.7,
    endLng: 139.7,
    color: '#ffffff'
  }
]);
```

### Custom Marker Shapes
Use custom images for markers:
```javascript
globe
  .htmlElementsData(data)
  .htmlElement(d => {
    const el = document.createElement('div');
    el.innerHTML = '📍';
    el.style.fontSize = '20px';
    return el;
  });
```

## Resources

- **Globe.GL Documentation**: https://github.com/vasturiano/globe.gl
- **Three.js**: https://threejs.org/
- **Open-Meteo Geocoding**: https://open-meteo.com/en/docs/geocoding-api
- **Example Gallery**: https://github.com/vasturiano/globe.gl/tree/master/example

## Code References

- **Frontend App**: `frontend/src/App.jsx`
- **API Service**: `frontend/src/services/api.js`
- **Test File**: `test_globe_movement.html`
