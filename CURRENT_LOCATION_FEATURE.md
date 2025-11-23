# Current Location Feature

## Overview
Added "Use Current Location" button that automatically detects the user's current location using the browser's Geolocation API and displays disaster risk predictions for their location.

## Features

### 📍 Automatic Location Detection
- **One-Click Access**: Single button click to get current location
- **Browser Geolocation**: Uses HTML5 Geolocation API
- **High Accuracy**: Requests high-accuracy GPS coordinates
- **Globe Animation**: Automatically moves globe to user's location
- **Green Marker**: Displays green marker (vs red for searched locations)
- **Auto-Prediction**: Automatically fetches risk assessment for current location

### 🔍 Updated Search Display
- **Removed Population**: Cleaner search results without population data
- **Simplified Display**: Shows only city name, state/province, and country
- **Better Readability**: More focused information display

## Implementation

### New Function: `handleUseCurrentLocation()`

```javascript
const handleUseCurrentLocation = () => {
  if (!navigator.geolocation) {
    showMessage('⚠️ Geolocation is not supported by your browser');
    return;
  }

  showMessage('📍 Getting your location...');
  
  navigator.geolocation.getCurrentPosition(
    async (position) => {
      const { latitude, longitude } = position.coords;
      
      // Create location data
      const locationData = {
        latitude,
        longitude,
        name: 'Current Location',
        display_name: `Current Location (${latitude.toFixed(4)}, ${longitude.toFixed(4)})`
      };
      
      // Move globe with green marker
      globeInstance.current.pointOfView({
        lat: latitude,
        lng: longitude,
        altitude: 1.5
      }, 1000);
      
      globeInstance.current.pointsData([{
        lat: latitude,
        lng: longitude,
        name: 'Your Location',
        color: '#00ff00' // Green marker
      }]);
      
      // Fetch weather and risk data
      await api.getLocationWeather(latitude, longitude);
      const locationId = `${latitude},${longitude}`;
      await fetchRiskData(locationId);
    },
    (error) => {
      // Handle errors
    },
    {
      enableHighAccuracy: true,
      timeout: 10000,
      maximumAge: 0
    }
  );
};
```

### UI Changes

**Before:**
```jsx
<input type="text" placeholder="Search..." />
// Results show: City, State, Country • Pop: 1,234,567
```

**After:**
```jsx
<button>📍 Use Current Location</button>
<input type="text" placeholder="Search..." />
// Results show: City, State, Country
```

## User Flow

### Using Current Location
1. User clicks "📍 Use Current Location" button
2. Browser requests location permission (first time only)
3. User grants permission
4. System gets GPS coordinates
5. Globe smoothly moves to user's location
6. Green marker appears at user's location
7. Weather data is fetched
8. Risk assessment is automatically displayed

### Searching for Location
1. User types city name
2. Autocomplete dropdown appears
3. Results show: City, State, Country (no population)
4. User selects a location
5. Globe moves to selected location
6. Red marker appears
7. User clicks "Get Risk Assessment" for predictions

## Marker Colors

- **Green (#00ff00)**: Current location (GPS-detected)
- **Red (#ff0000)**: Searched location (user-selected)

## Error Handling

### Permission Denied
```
⚠️ Location permission denied. Please enable location access.
```

### Position Unavailable
```
⚠️ Location information unavailable.
```

### Timeout
```
⚠️ Location request timed out.
```

### Browser Not Supported
```
⚠️ Geolocation is not supported by your browser
```

## Browser Compatibility

### Supported Browsers
- ✅ Chrome 5+
- ✅ Firefox 3.5+
- ✅ Safari 5+
- ✅ Edge 12+
- ✅ Opera 10.6+
- ✅ iOS Safari 3.2+
- ✅ Android Browser 2.1+

### Requirements
- **HTTPS**: Geolocation requires secure context (HTTPS or localhost)
- **User Permission**: User must grant location access
- **GPS/Network**: Device must have GPS or network-based location

## Configuration

### Geolocation Options
```javascript
{
  enableHighAccuracy: true,  // Use GPS if available
  timeout: 10000,            // 10 second timeout
  maximumAge: 0              // Don't use cached position
}
```

### Accuracy Levels
- **High Accuracy (GPS)**: ±5-10 meters
- **Network-based**: ±100-1000 meters
- **IP-based**: ±5-50 kilometers

## Testing

### Test File
Open `test_globe_movement.html` to test:
```bash
open test_globe_movement.html
```

### Test Steps
1. Click "📍 Use Current Location"
2. Grant location permission when prompted
3. Verify globe moves to your location
4. Verify green marker appears
5. Check coordinates are displayed

### Manual Testing
```javascript
// Test geolocation support
if (navigator.geolocation) {
  console.log('Geolocation supported');
} else {
  console.log('Geolocation not supported');
}

// Get current position
navigator.geolocation.getCurrentPosition(
  (position) => {
    console.log('Lat:', position.coords.latitude);
    console.log('Lng:', position.coords.longitude);
    console.log('Accuracy:', position.coords.accuracy, 'meters');
  }
);
```

## Privacy & Security

### User Privacy
- Location is only accessed when user clicks the button
- Location is not stored or transmitted to external servers
- Location is only used for local risk assessment
- User can deny permission at any time

### Security Requirements
- **HTTPS Required**: Geolocation only works on HTTPS (or localhost)
- **User Consent**: Browser always asks for permission
- **Temporary Access**: Permission can be revoked anytime

### Best Practices
- Clear explanation of why location is needed
- Respect user's privacy choices
- Don't repeatedly request permission
- Provide alternative (manual search)

## Future Enhancements

### 1. Location Tracking
```javascript
// Watch position for continuous updates
const watchId = navigator.geolocation.watchPosition(
  (position) => {
    // Update location in real-time
  }
);
```

### 2. Reverse Geocoding
```javascript
// Get city name from coordinates
const response = await fetch(
  `https://geocoding-api.open-meteo.com/v1/reverse?latitude=${lat}&longitude=${lng}`
);
```

### 3. Location History
```javascript
// Save recent locations
const recentLocations = [
  { lat: 51.5, lng: -0.1, name: 'London', timestamp: Date.now() },
  // ...
];
```

### 4. Nearby Locations
```javascript
// Find locations within radius
const nearbyLocations = locations.filter(loc => {
  const distance = calculateDistance(userLat, userLng, loc.lat, loc.lng);
  return distance < 50; // 50km radius
});
```

## Troubleshooting

### Location Not Working
1. Check browser console for errors
2. Verify HTTPS or localhost
3. Check location permissions in browser settings
4. Try different browser
5. Check device location services are enabled

### Inaccurate Location
1. Enable high accuracy mode
2. Wait for GPS lock (may take 10-30 seconds)
3. Move to area with better GPS signal
4. Check device location settings

### Permission Denied
1. Check browser location settings
2. Clear site permissions and try again
3. Use manual search as alternative

## Code References

- **Main App**: `frontend/src/App.jsx`
- **Test File**: `test_globe_movement.html`
- **API Service**: `frontend/src/services/api.js`

## Resources

- **MDN Geolocation API**: https://developer.mozilla.org/en-US/docs/Web/API/Geolocation_API
- **Can I Use**: https://caniuse.com/geolocation
- **W3C Specification**: https://www.w3.org/TR/geolocation-API/
