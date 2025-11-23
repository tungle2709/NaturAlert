# Location Service Fix Guide

## Issue
`CoreLocationProvider: CoreLocation framework reported a kCLErrorLocationUnknown failure`

This is a macOS system-level issue, not a code problem.

## Quick Solution: Use Location Search Instead ✅

The app has a **location search feature** that works perfectly:

1. Type any city name in the search box (e.g., "Toronto", "London", "Tokyo")
2. Select from the dropdown
3. The app will fetch real-time weather and analyze disaster risk

**This is the recommended way to use the app!**

## Why Current Location Fails

macOS CoreLocation can fail for several reasons:
- Location Services disabled
- No GPS/WiFi signal
- Browser doesn't have location permission
- System can't determine location at that moment

## Fix macOS Location Services (Optional)

If you want to fix the "Current Location" button:

### Step 1: Enable Location Services
1. Open **System Settings** (or System Preferences)
2. Go to **Privacy & Security**
3. Click **Location Services**
4. Ensure "Location Services" is **ON**

### Step 2: Enable for Your Browser
1. In Location Services, scroll down to your browser (Chrome, Safari, Firefox, etc.)
2. Make sure it's checked/enabled

### Step 3: Grant Browser Permission
1. Open your browser
2. Go to `http://localhost:5173`
3. When prompted, click **Allow** for location access

### Step 4: Check WiFi
- Make sure WiFi is enabled (macOS uses WiFi for location)
- Connect to a WiFi network if possible

## Code Improvements Made

I've improved the app to handle location errors better:

### 1. Better Error Messages
```javascript
// Now shows helpful suggestions
"Location information unavailable. Try searching for your city instead"
```

### 2. Auto-Focus Search Box
When location fails, the search box automatically gets focus so you can type immediately.

### 3. Enhanced Geolocation Options
```javascript
{
  enableHighAccuracy: true,  // Use GPS if available
  timeout: 10000,            // 10 second timeout
  maximumAge: 0              // Don't use cached position
}
```

### 4. Detailed Error Handling
- Permission denied → Shows how to enable
- Position unavailable → Suggests search
- Timeout → Offers retry or search
- Unknown error → Directs to search

## Testing the Fix

### Test Location Search (Recommended)
1. Open `http://localhost:5173`
2. Type "Toronto" in the search box
3. Select "Toronto, Ontario, Canada"
4. ✅ Should show weather and disaster risk

### Test Current Location (Optional)
1. Follow the macOS fix steps above
2. Click "Use Current Location" button
3. Allow location access when prompted
4. Should detect your location

## Alternative: Use Coordinates

You can also test with direct coordinates:
```bash
curl "http://localhost:8000/api/v1/risk/current?location_id=43.65,-79.38"
```

## Summary

✅ **Location search works perfectly** - Use this!
⚠️ **Current location is optional** - macOS system issue
🔧 **Error handling improved** - Better user experience

The app is fully functional using the location search feature. The "Current Location" button is just a convenience feature that depends on macOS system services.

---

**Bottom Line**: Use the search box to find any location worldwide. It's faster and more reliable than geolocation!
