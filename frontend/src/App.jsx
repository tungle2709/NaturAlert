import React, { useState, useEffect, useRef } from 'react';
import { initializeApp } from 'firebase/app';
import { getAuth, signInAnonymously, onAuthStateChanged } from 'firebase/auth';
import { getFirestore, collection, addDoc, onSnapshot, query, orderBy, limit } from 'firebase/firestore';
import Globe from 'globe.gl';
import * as api from './services/api';

const App = () => {
  // Disable all console logging in production
  if (typeof window !== 'undefined') {
    console.log = () => {};
    console.error = () => {};
    console.warn = () => {};
    console.info = () => {};
    console.debug = () => {};
  }

  const [page, setPage] = useState('dashboard');
  const [user, setUser] = useState(null);
  const [location, setLocation] = useState('default');
  const [riskData, setRiskData] = useState(null);
  const [trendData, setTrendData] = useState(null);
  const [hourlyPredictions, setHourlyPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [email, setEmail] = useState('');
  const [sms, setSms] = useState('');
  const [sosLocation, setSosLocation] = useState('');
  const [peopleCount, setPeopleCount] = useState('');
  const [savedLocations, setSavedLocations] = useState([]);
  const [sosAlerts, setSosAlerts] = useState([]);
  const [currentUserLocation, setCurrentUserLocation] = useState(null);
  const [nearbySOSAlerts, setNearbySOSAlerts] = useState([]);
  const [mySOSAlerts, setMySOSAlerts] = useState([]);
  const [sosRadiusFilter, setSosRadiusFilter] = useState(10); // Default 10km
  const [message, setMessage] = useState('');
  const [newLocation, setNewLocation] = useState('');
  const [chatMessage, setChatMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [locationSearch, setLocationSearch] = useState('');
  const [locationResults, setLocationResults] = useState([]);
  const [showLocationDropdown, setShowLocationDropdown] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [historicalWeather, setHistoricalWeather] = useState(null);
  const globeEl = useRef();
  const globeInstance = useRef();
  const searchTimeoutRef = useRef(null);

  // Initialize Firebase
  useEffect(() => {
    const app = initializeApp(window.__firebase_config);
    const auth = getAuth(app);
    const db = getFirestore(app);

    (async () => {
      try {
        await signInAnonymously(auth);
      } catch (e) {
        console.error('Auth error:', e);
      }
    })();

    onAuthStateChanged(auth, (u) => {
      setUser(u);
    });

    // Load saved locations from localStorage
    const loadSavedLocations = () => {
      try {
        const saved = JSON.parse(localStorage.getItem('savedLocations') || '[]');
        setSavedLocations(saved);
      } catch (err) {
        console.error('Error loading saved locations:', err);
        setSavedLocations([]);
      }
    };
    
    // Load SOS alerts from localStorage
    const loadSOSAlerts = () => {
      try {
        const alerts = JSON.parse(localStorage.getItem('sosAlerts') || '[]');
        setSosAlerts(alerts);
        
        const myAlerts = JSON.parse(localStorage.getItem('mySOSAlerts') || '[]');
        setMySOSAlerts(myAlerts);
      } catch (err) {
        console.error('Error loading SOS alerts:', err);
        setSosAlerts([]);
        setMySOSAlerts([]);
      }
    };
    
    loadSavedLocations();
    loadSOSAlerts();

    // Initialize Globe
    if (globeEl.current && !globeInstance.current) {
      globeInstance.current = Globe()(globeEl.current)
        .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
        .bumpImageUrl('//unpkg.com/three-globe/example/img/earth-topology.png')
        .backgroundImageUrl('//unpkg.com/three-globe/example/img/night-sky.png')
        .pointOfView({ altitude: 2.5 })
        .htmlElementsData([])
        .htmlElement(d => {
          const el = document.createElement('div');
          el.innerHTML = `
            <div style="
              color: ${d.color};
              font-size: 24px;
              text-shadow: 0 0 4px rgba(0,0,0,0.8);
              cursor: pointer;
              transform: translateY(-12px);
            " title="${d.name}">
              📍
            </div>
          `;
          return el;
        });
    }
  }, []);

  // Fetch risk data from backend API
  const fetchRiskData = async (locationId = 'default') => {
    setLoading(true);
    setError(null);
    
    try {
      // Fetch current risk
      const risk = await api.getCurrentRisk(locationId);
      setRiskData(risk);
      
      // Fetch trends
      const trends = await api.getRiskTrends(locationId);
      setTrendData(trends);
      
      // Fetch hourly predictions
      const hourly = await api.getHourlyPredictions(locationId, 24);
      setHourlyPredictions(hourly);
      
      setSosLocation(locationId);
      
      showMessage('✓ Risk data updated successfully');
    } catch (err) {
      setError(err.message);
      // Silently handle error - no message displayed
    } finally {
      setLoading(false);
    }
  };

  // Handle location search with debouncing
  const handleLocationSearch = async (searchQuery) => {
    setLocationSearch(searchQuery);
    
    if (searchQuery.length < 2) {
      setLocationResults([]);
      setShowLocationDropdown(false);
      return;
    }
    
    // Clear previous timeout
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }
    
    // Debounce search
    searchTimeoutRef.current = setTimeout(async () => {
      try {
        const results = await api.searchLocation(searchQuery, 10);
        setLocationResults(results.results || []);
        setShowLocationDropdown(true);
      } catch (err) {
        console.error('Location search failed:', err);
        setLocationResults([]);
      }
    }, 300);
  };

  const handleUseCurrentLocation = () => {
    if (!navigator.geolocation) {
      showMessage('⚠ Geolocation is not supported by your browser');
      return;
    }

    showMessage('📍 Getting your location...');
    setLocationSearch('Getting your location...');
    setLoading(true);
    
    // Enhanced geolocation options
    const geoOptions = {
      enableHighAccuracy: true,  // Use GPS if available
      timeout: 15000,            // 15 second timeout (increased)
      maximumAge: 300000         // Accept cached position up to 5 minutes old
    };
    
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;
        
        try {
          showMessage('🌍 Found your location, getting weather data...');
          
          // Use reverse geocoding to get location name
          const response = await fetch(
            `https://geocoding-api.open-meteo.com/v1/search?latitude=${latitude}&longitude=${longitude}&count=1&language=en&format=json`
          );
          
          let locationName = 'Current Location';
          let admin1 = '';
          let country = '';
          
          if (response.ok) {
            const data = await response.json();
            if (data.results && data.results.length > 0) {
              const result = data.results[0];
              locationName = result.name || 'Current Location';
              admin1 = result.admin1 || '';
              country = result.country || '';
            }
          }
          
          // Create location data object
          const locationData = {
            latitude,
            longitude,
            name: locationName,
            admin1: admin1,
            country: country,
            display_name: `${locationName}${admin1 ? ', ' + admin1 : ''}${country ? ', ' + country : ''}`
          };
          
          setSelectedLocation(locationData);
          setLocationSearch(locationData.display_name);
          
          // Move globe to current location
          if (globeInstance.current) {
            globeInstance.current.pointOfView({
              lat: latitude,
              lng: longitude,
              altitude: 1.5
            }, 1000);
            
            globeInstance.current.htmlElementsData([{
              lat: latitude,
              lng: longitude,
              name: locationData.display_name,
              size: 0.5,
              color: '#00ff00'
            }]);
          }
          
          showMessage(`🔍 Analyzing weather data for ${locationName}...`);
          
          // Use the main risk assessment endpoint which fetches everything
          const locationId = `${latitude},${longitude}`;
          const risk = await api.getCurrentRisk(locationId);
          
          setRiskData(risk);
          setLocation(locationId);
          showMessage(`✅ Analysis complete: ${locationName}`);
        } catch (err) {
          console.error('Location analysis error:', err);
          showMessage(`⚠ Error analyzing location: ${err.message}`);
          setLocationSearch('');
        } finally {
          setLoading(false);
        }
      },
      (error) => {
        let errorMessage = 'Unable to get your location';
        let suggestion = 'Please use the location search instead.';
        
        switch (error.code) {
          case error.PERMISSION_DENIED:
            errorMessage = '🚫 Location permission denied';
            suggestion = 'Please enable location access in your browser settings';
            break;
          case error.POSITION_UNAVAILABLE:
            errorMessage = '📍 Location information unavailable';
            suggestion = 'Try searching for your city instead (e.g., "Toronto", "New York")';
            break;
          case error.TIMEOUT:
            errorMessage = '⏱ Location request timed out';
            suggestion = 'Please try again or search for your location manually';
            break;
          default:
            errorMessage = '⚠ Location service error';
            suggestion = 'Use the search box to find your location';
        }
        
        showMessage(`${errorMessage}. ${suggestion}`);
        setLocationSearch('');
        setLoading(false);
        
        // Focus on search input as fallback
        setTimeout(() => {
          const searchInput = document.querySelector('input[type="text"]');
          if (searchInput) searchInput.focus();
        }, 100);
      },
      geoOptions
    );
  };

  const handleSelectLocation = async (locationData) => {
    setSelectedLocation(locationData);
    setLocationSearch(locationData.display_name);
    setShowLocationDropdown(false);
    setLoading(true);
    
    // Move globe to selected location
    if (globeInstance.current) {
      // Animate globe to the selected location
      globeInstance.current.pointOfView({
        lat: locationData.latitude,
        lng: locationData.longitude,
        altitude: 1.5
      }, 1000); // 1000ms animation duration
      
      // Add a marker at the selected location
      globeInstance.current.htmlElementsData([{
        lat: locationData.latitude,
        lng: locationData.longitude,
        name: locationData.display_name,
        size: 0.5,
        color: '#ff0000'
      }]);
    }
    
    // Fetch weather data for the selected location
    try {
      showMessage(`✓ Analyzing weather data...`);
      
      // Use the main risk assessment endpoint which fetches everything
      const locationId = `${locationData.latitude},${locationData.longitude}`;
      const risk = await api.getCurrentRisk(locationId);
      
      setRiskData(risk);
      setLocation(locationId);
      
      showMessage(`✓ Analysis complete: ${locationData.display_name}`);
      
    } catch (err) {
      // Silently handle error - no message displayed
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!location) return;
    await fetchRiskData(location);
  };

  const handleAlertSubscribe = async () => {
    if (!email || !selectedLocation) {
      showMessage('! Please enter email and select a location');
      return;
    }
    
    try {
      const response = await api.subscribeToAlerts({
        email: email,
        location_name: selectedLocation.display_name,
        latitude: selectedLocation.latitude,
        longitude: selectedLocation.longitude,
        sms: sms || null
      });
      
      showMessage(`✓ Successfully subscribed to alerts for ${selectedLocation.name}! Check your email for weather analysis.`);
      setEmail('');
      setSms('');
      setSelectedLocation(null);
      setLocationSearch('');
    } catch (err) {
      showMessage(`! Failed to subscribe: ${err.message}`);
    }
  };

  const handleSOS = () => {
    if (!currentUserLocation) {
      showMessage('⚠ Please enable location access first');
      requestLocationForSOS();
      return;
    }

    try {
      // Get existing SOS alerts from localStorage
      const existingAlerts = JSON.parse(localStorage.getItem('sosAlerts') || '[]');
      
      // Create new SOS alert
      const newAlert = {
        id: Date.now().toString(),
        location: sosLocation || 'Emergency Location',
        latitude: currentUserLocation.latitude,
        longitude: currentUserLocation.longitude,
        peopleCount: parseInt(peopleCount) || 1,
        timestamp: Date.now(),
        status: 'active',
        isMine: true // Mark as user's own alert
      };
      
      // Add to beginning of array
      const updatedAlerts = [newAlert, ...existingAlerts];
      
      // Save to localStorage
      localStorage.setItem('sosAlerts', JSON.stringify(updatedAlerts));
      
      // Update state
      setSosAlerts(updatedAlerts);
      
      // Track user's own SOS alerts
      const myAlerts = JSON.parse(localStorage.getItem('mySOSAlerts') || '[]');
      const updatedMyAlerts = [newAlert, ...myAlerts];
      localStorage.setItem('mySOSAlerts', JSON.stringify(updatedMyAlerts));
      setMySOSAlerts(updatedMyAlerts);
      
      showMessage('✓ SOS Alert Sent! Help is on the way.');
      
      // Refresh nearby alerts and update globe
      if (currentUserLocation) {
        const nearby = filterNearbyAlerts(currentUserLocation.latitude, currentUserLocation.longitude);
        
        // Update globe markers
        if (globeInstance.current) {
          const markers = [
            // User's current location (green)
            {
              lat: currentUserLocation.latitude,
              lng: currentUserLocation.longitude,
              name: 'Your Location',
              size: 0.8,
              color: '#00ff00'
            }
          ];
          
          // Add nearby SOS alerts (red)
          nearby.forEach(alert => {
            if (alert.latitude && alert.longitude) {
              markers.push({
                lat: alert.latitude,
                lng: alert.longitude,
                name: `SOS: ${alert.location || 'Emergency'} - ${alert.peopleCount} people`,
                size: 0.6,
                color: '#ff0000'
              });
            }
          });
          
          globeInstance.current.htmlElementsData(markers);
        }
      }
      
      // Reset form
      setSosLocation('');
      setPeopleCount('');
    } catch (err) {
      showMessage('⚠ Failed to send SOS alert');
      console.error('SOS error:', err);
    }
  };

  const handleSaveLocation = async () => {
    if (!user || !newLocation) return;
    const db = getFirestore();
    await addDoc(collection(db, `artifacts/${window.__app_id}/users/${user.uid}/saved_locations`), {
      name: newLocation, timestamp: Date.now()
    });
    setNewLocation('');
  };

  const handleSaveCurrentLocation = () => {
    if (!selectedLocation || !riskData) {
      showMessage('⚠ Please select a location and get risk data first');
      return;
    }
    
    try {
      // Get existing saved locations from localStorage
      const existingLocations = JSON.parse(localStorage.getItem('savedLocations') || '[]');
      
      // Create new location object
      const newLocation = {
        id: Date.now().toString(), // Use timestamp as unique ID
        name: selectedLocation.display_name || selectedLocation.name,
        location_id: location,
        latitude: selectedLocation.latitude,
        longitude: selectedLocation.longitude,
        risk_score: riskData.risk_score,
        disaster_type: riskData.disaster_type,
        confidence: riskData.confidence,
        has_risk: riskData.has_risk,
        explanation: riskData.ai_explanation || riskData.explanation,
        recommendations: riskData.recommendations,
        weather_snapshot: riskData.weather_snapshot,
        timestamp: Date.now()
      };
      
      // Add new location to the beginning of the array
      const updatedLocations = [newLocation, ...existingLocations];
      
      // Save back to localStorage
      localStorage.setItem('savedLocations', JSON.stringify(updatedLocations));
      
      // Update state to reflect the new saved location
      setSavedLocations(updatedLocations);
      
      showMessage('✓ Location and prediction saved successfully!');
      
      // Navigate to saved page after a short delay
      setTimeout(() => {
        setPage('saved');
      }, 1500);
      
    } catch (err) {
      showMessage('⚠ Failed to save location');
      console.error('Save error:', err);
    }
  };

  const handleCheckPrediction = async (loc) => {
    setLocation(loc);
    await fetchRiskData(loc);
    setPage('dashboard');
  };

  const handleDeleteLocation = (locationId) => {
    try {
      // Filter out the location to delete
      const updatedLocations = savedLocations.filter(loc => loc.id !== locationId);
      
      // Update localStorage
      localStorage.setItem('savedLocations', JSON.stringify(updatedLocations));
      
      // Update state
      setSavedLocations(updatedLocations);
      
      showMessage('✓ Location deleted successfully');
    } catch (err) {
      showMessage('⚠ Failed to delete location');
      console.error('Delete error:', err);
    }
  };

  const handleMarkAsSafe = (alertId) => {
    try {
      // Remove from my SOS alerts
      const updatedMyAlerts = mySOSAlerts.filter(alert => alert.id !== alertId);
      localStorage.setItem('mySOSAlerts', JSON.stringify(updatedMyAlerts));
      setMySOSAlerts(updatedMyAlerts);
      
      // Remove from all SOS alerts
      const allAlerts = JSON.parse(localStorage.getItem('sosAlerts') || '[]');
      const updatedAllAlerts = allAlerts.filter(alert => alert.id !== alertId);
      localStorage.setItem('sosAlerts', JSON.stringify(updatedAllAlerts));
      setSosAlerts(updatedAllAlerts);
      
      // Update globe markers if location is available
      if (currentUserLocation) {
        const nearby = filterNearbyAlerts(currentUserLocation.latitude, currentUserLocation.longitude);
        
        if (globeInstance.current) {
          const markers = [
            {
              lat: currentUserLocation.latitude,
              lng: currentUserLocation.longitude,
              name: 'Your Location',
              size: 0.8,
              color: '#00ff00'
            }
          ];
          
          nearby.forEach(alert => {
            if (alert.latitude && alert.longitude) {
              markers.push({
                lat: alert.latitude,
                lng: alert.longitude,
                name: `SOS: ${alert.location || 'Emergency'} - ${alert.peopleCount} people`,
                size: 0.6,
                color: '#ff0000'
              });
            }
          });
          
          globeInstance.current.htmlElementsData(markers);
        }
      }
      
      showMessage('✓ Marked as safe. SOS alert removed.');
    } catch (err) {
      showMessage('⚠ Failed to mark as safe');
      console.error('Mark as safe error:', err);
    }
  };

  const handleChatSend = async () => {
    if (!chatMessage.trim()) return;
    
    // Add user message to history
    const userMsg = { role: 'user', content: chatMessage, timestamp: new Date().toISOString() };
    setChatHistory(prev => [...prev, userMsg]);
    
    const currentMessage = chatMessage;
    setChatMessage('');
    
    try {
      // Prepare context from current risk data
      const context = riskData ? {
        location: location,
        risk_score: riskData.risk_score,
        disaster_type: riskData.disaster_type,
        temperature: riskData.weather_snapshot?.temperature,
        pressure: riskData.weather_snapshot?.pressure,
      } : null;
      
      // Send to Gemini
      const response = await api.chatWithGemini(currentMessage, context, conversationId);
      
      // Update conversation ID
      if (!conversationId) {
        setConversationId(response.conversation_id);
      }
      
      // Add assistant response to history
      const assistantMsg = { role: 'assistant', content: response.response, timestamp: response.timestamp };
      setChatHistory(prev => [...prev, assistantMsg]);
      
    } catch (err) {
      // Silently handle error - no message displayed
    }
  };

  const showMessage = (msg) => {
    setMessage(msg);
    setTimeout(() => setMessage(''), 5000);
  };

  const getRiskColor = (score) => {
    if (score >= 70) return 'text-red-500';
    if (score >= 50) return 'text-yellow-500';
    return 'text-green-500';
  };

  const getRiskLabel = (score) => {
    if (score >= 70) return 'HIGH RISK';
    if (score >= 50) return 'MODERATE RISK';
    return 'LOW RISK';
  };

  // Calculate distance between two coordinates using Haversine formula
  const calculateDistance = (lat1, lon1, lat2, lon2) => {
    const R = 6371; // Earth's radius in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = 
      Math.sin(dLat/2) * Math.sin(dLat/2) +
      Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
      Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c; // Distance in km
  };

  // Get current location for SOS
  const requestLocationForSOS = () => {
    if (!navigator.geolocation) {
      showMessage('⚠ Geolocation is not supported by your browser');
      return;
    }

    showMessage('Requesting location permission...');

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        setCurrentUserLocation({ latitude, longitude });
        showMessage('✓ Location access granted');
        
        // Filter nearby SOS alerts
        const nearby = filterNearbyAlerts(latitude, longitude);
        
        // Zoom globe to user's location
        if (globeInstance.current) {
          globeInstance.current.pointOfView({
            lat: latitude,
            lng: longitude,
            altitude: 1.5
          }, 1000);
          
          // Create markers for user location and nearby SOS alerts
          const markers = [
            // User's current location (green)
            {
              lat: latitude,
              lng: longitude,
              name: 'Your Location',
              size: 0.8,
              color: '#00ff00'
            }
          ];
          
          // Add nearby SOS alerts (red)
          nearby.forEach(alert => {
            if (alert.latitude && alert.longitude) {
              markers.push({
                lat: alert.latitude,
                lng: alert.longitude,
                name: `SOS: ${alert.location || 'Emergency'} - ${alert.peopleCount} people`,
                size: 0.6,
                color: '#ff0000'
              });
            }
          });
          
          globeInstance.current.htmlElementsData(markers);
        }
      },
      (error) => {
        let errorMessage = 'Unable to get your location';
        switch (error.code) {
          case error.PERMISSION_DENIED:
            errorMessage = '⚠ Location permission denied. Please enable location access.';
            break;
          case error.POSITION_UNAVAILABLE:
            errorMessage = '⚠ Location information unavailable';
            break;
          case error.TIMEOUT:
            errorMessage = '⚠ Location request timed out';
            break;
          default:
            errorMessage = '⚠ Location service error';
        }
        showMessage(errorMessage);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0
      }
    );
  };

  // Filter SOS alerts within specified radius
  const filterNearbyAlerts = (userLat, userLon, radius = sosRadiusFilter) => {
    if (!sosAlerts || sosAlerts.length === 0) {
      setNearbySOSAlerts([]);
      return [];
    }

    const nearby = sosAlerts.filter(alert => {
      if (!alert.latitude || !alert.longitude) return false;
      const distance = calculateDistance(userLat, userLon, alert.latitude, alert.longitude);
      return distance <= radius; // Within specified radius
    }).map(alert => ({
      ...alert,
      distance: calculateDistance(userLat, userLon, alert.latitude, alert.longitude)
    })).sort((a, b) => a.distance - b.distance); // Sort by distance

    setNearbySOSAlerts(nearby);
    return nearby;
  };

  return (
    <div className="relative min-h-screen overflow-hidden">
      <div ref={globeEl} className="fixed inset-0 w-full h-full"></div>
      
      <div className="relative z-10">
        {/* iOS-style Navigation Bar with Ultra Blur - Ultra Transparent */}
        <nav className="fixed top-6 left-1/2 -translate-x-1/2 bg-white/5 backdrop-blur-3xl border border-white/20 rounded-[28px] px-6 py-2.5 shadow-[0_8px_32px_rgba(0,0,0,0.08)]">
          <div className="flex gap-1.5 items-center">
            <button onClick={() => setPage('dashboard')} className={`px-5 py-2 rounded-[20px] transition-all duration-300 font-medium text-sm ${page === 'dashboard' ? 'bg-white/95 text-gray-900 shadow-[0_2px_8px_rgba(0,0,0,0.15)]' : 'text-white/90 hover:bg-white/15'}`}>Home</button>
            <button onClick={() => setPage('alerts')} className={`px-5 py-2 rounded-[20px] transition-all duration-300 font-medium text-sm ${page === 'alerts' ? 'bg-white/95 text-gray-900 shadow-[0_2px_8px_rgba(0,0,0,0.15)]' : 'text-white/90 hover:bg-white/15'}`}>Alerts</button>
            <button onClick={() => setPage('saved')} className={`px-5 py-2 rounded-[20px] transition-all duration-300 font-medium text-sm ${page === 'saved' ? 'bg-white/95 text-gray-900 shadow-[0_2px_8px_rgba(0,0,0,0.15)]' : 'text-white/90 hover:bg-white/15'}`}>Saved</button>
            <button onClick={() => setPage('chat')} className={`px-5 py-2 rounded-[20px] transition-all duration-300 font-medium text-sm ${page === 'chat' ? 'bg-white/95 text-gray-900 shadow-[0_2px_8px_rgba(0,0,0,0.15)]' : 'text-white/90 hover:bg-white/15'}`}>AI Chat</button>
            <button onClick={() => setPage('sos')} className={`px-5 py-2 rounded-[20px] transition-all duration-300 font-medium text-sm ${page === 'sos' ? 'bg-red-500/95 text-white shadow-[0_2px_8px_rgba(239,68,68,0.4)]' : 'text-white/90 hover:bg-red-500/20'}`}>SOS</button>
          </div>
        </nav>

        {/* iOS-style User Badge - Ultra Transparent */}
        {user && (
          <div className="fixed top-6 right-6 bg-white/5 backdrop-blur-3xl border border-white/20 rounded-[20px] px-5 py-2.5 shadow-[0_8px_32px_rgba(0,0,0,0.08)]">
            <span className="text-white/90 text-sm font-medium">User: {user.uid.slice(0, 8)}</span>
          </div>
        )}

        {/* iOS-style Toast Message - Ultra Transparent */}
        {message && (
          <div className="fixed top-24 right-6 z-50 animate-fade-in">
            <div className="bg-white/8 backdrop-blur-3xl border border-white/20 text-white px-6 py-3 rounded-[20px] shadow-[0_8px_32px_rgba(0,0,0,0.08)] font-medium">{message}</div>
          </div>
        )}

        {page === 'dashboard' && (
          <div className="fixed left-6 top-24 w-96 space-y-3 max-h-[calc(100vh-120px)] overflow-y-auto scrollbar-hide">
            {/* iOS-style Search Card - Ultra Transparent */}
            <div className="bg-white/5 backdrop-blur-3xl rounded-[28px] p-6 border border-white/20 shadow-[0_8px_32px_rgba(0,0,0,0.08)]">
              <h2 className="text-xl font-semibold mb-4 text-white/95">Search Location</h2>
              <div className="space-y-3">
                <div className="relative">
                  <input
                    type="text"
                    value={locationSearch}
                    onChange={(e) => handleLocationSearch(e.target.value)}
                    onFocus={() => setShowLocationDropdown(true)}
                    placeholder="Search city, country... (e.g., London, Tokyo)"
                    className="w-full px-5 py-3.5 bg-white/10 backdrop-blur-2xl border border-white/20 rounded-[18px] text-white placeholder-white/60 text-sm focus:outline-none focus:ring-2 focus:ring-white/30 focus:border-white/40 transition-all"
                  />
                  
                  {showLocationDropdown && (locationResults.length > 0 || locationSearch.length < 2) && (
                    <div className="absolute z-50 w-full mt-2 bg-white border border-gray-200 rounded-[20px] shadow-[0_8px_32px_rgba(0,0,0,0.15)] max-h-[180px] overflow-y-auto">
                      <button
                        onClick={() => {
                          handleUseCurrentLocation();
                          setShowLocationDropdown(false);
                        }}
                        className="w-full text-left px-4 py-3 hover:bg-gray-50 border-b border-gray-200 transition-all rounded-t-[20px]"
                      >
                        <div className="font-semibold text-gray-900 flex items-center gap-2">
                           Use Current Location
                        </div>
                        <div className="text-xs text-gray-600">
                          Detect your location automatically
                        </div>
                      </button>
                      
                      {locationResults.length > 0 ? (
                        locationResults.map((loc, idx) => (
                          <button
                            key={idx}
                            onClick={() => handleSelectLocation(loc)}
                            className="w-full text-left px-4 py-3 hover:bg-gray-50 border-b border-gray-100 last:border-b-0 transition-all last:rounded-b-[20px]"
                          >
                            <div className="font-semibold text-gray-900">{loc.name}</div>
                            <div className="text-xs text-gray-600">
                              {loc.admin1 && `${loc.admin1}, `}{loc.country}
                            </div>
                          </button>
                        ))
                      ) : locationSearch.length >= 2 ? (
                        <div className="px-4 py-3 text-gray-600 text-sm text-center">
                          Searching...
                        </div>
                      ) : (
                        <div className="px-4 py-3 text-gray-600 text-sm text-center">
                          Type to search locations
                        </div>
                      )}
                    </div>
                  )}
                </div>
                
                {selectedLocation && (
                  <div className="bg-white/8 backdrop-blur-2xl rounded-[16px] p-3 border border-white/20">
                    <div className="text-white/95 text-sm">
                      <div className="font-semibold">• {selectedLocation.name}</div>
                      {selectedLocation.admin1 && selectedLocation.country && (
                        <div className="text-xs text-white/70 mt-1">
                          {selectedLocation.admin1}, {selectedLocation.country}
                        </div>
                      )}
                    </div>
                  </div>
                )}
                
                <button 
                  onClick={handlePredict} 
                  disabled={loading || !location}
                  className="w-full py-3.5 bg-white/10 backdrop-blur-2xl border border-white/20 text-white rounded-[18px] hover:bg-white/15 font-semibold shadow-[0_4px_16px_rgba(0,0,0,0.08)] disabled:opacity-50 transition-all"
                >
                  {loading ? 'Analyzing...' : 'Get Risk Assessment'}
                </button>
              </div>
            </div>

            {riskData && (
              <>
                {/* iOS-style Risk Card with Gradient */}
                <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 backdrop-blur-3xl rounded-[28px] p-6 border border-white/30 shadow-[0_8px_32px_rgba(0,0,0,0.12)]">
                  <h3 className="text-xs font-semibold text-white/70 mb-2 tracking-wider">DISASTER RISK</h3>
                  <div className={`text-5xl font-bold ${getRiskColor(riskData.risk_score)}`}>
                    {riskData.risk_score.toFixed(1)}%
                  </div>
                  <div className="text-white/95 text-lg mt-2 font-medium">
                    {getRiskLabel(riskData.risk_score)}
                  </div>
                  <div className="text-white/80 text-sm mt-2">
                    Type: {riskData.disaster_type.replace('_', ' ').toUpperCase()}
                  </div>
                  <div className="text-white/60 text-xs mt-2">
                    Confidence: {riskData.confidence.toFixed(1)}%
                  </div>
                  
                  {/* Save Button */}
                  <button
                    onClick={handleSaveCurrentLocation}
                    className="w-full mt-4 py-3 bg-white/10 backdrop-blur-2xl border border-white/20 text-white rounded-[18px] hover:bg-white/15 font-semibold shadow-[0_4px_16px_rgba(0,0,0,0.08)] transition-all"
                  >
                    Save Location & Prediction
                  </button>
                </div>

                {riskData.weather_snapshot && (
                  <div className="bg-white/5 backdrop-blur-3xl rounded-[28px] p-6 border border-white/20 shadow-[0_8px_32px_rgba(0,0,0,0.08)]">
                    <h3 className="text-lg font-semibold mb-4 text-white/95">Current Weather</h3>
                    <div className="grid grid-cols-2 gap-2.5">
                      <div className="bg-white/8 backdrop-blur-xl rounded-[16px] p-3 border border-white/15">
                        <div className="text-white/70 text-xs">Temperature</div>
                        <div className="text-white/95 text-lg font-semibold">{riskData.weather_snapshot.temperature.toFixed(1)}°C</div>
                      </div>
                      <div className="bg-white/8 backdrop-blur-xl rounded-[16px] p-3 border border-white/15">
                        <div className="text-white/70 text-xs">Pressure</div>
                        <div className="text-white/95 text-lg font-semibold">{riskData.weather_snapshot.pressure.toFixed(0)} hPa</div>
                      </div>
                      <div className="bg-white/8 backdrop-blur-xl rounded-[16px] p-3 border border-white/15">
                        <div className="text-white/70 text-xs">Humidity</div>
                        <div className="text-white/95 text-lg font-semibold">{riskData.weather_snapshot.humidity.toFixed(0)}%</div>
                      </div>
                      <div className="bg-white/8 backdrop-blur-xl rounded-[16px] p-3 border border-white/15">
                        <div className="text-white/70 text-xs">Wind Speed</div>
                        <div className="text-white/95 text-lg font-semibold">{riskData.weather_snapshot.wind_speed.toFixed(0)} mph</div>
                      </div>
                    </div>
                  </div>
                )}

                {riskData.ai_explanation && (
                  <div className="bg-white/5 backdrop-blur-3xl rounded-[28px] p-6 border border-white/20 shadow-[0_8px_32px_rgba(0,0,0,0.08)]">
                    <h3 className="text-lg font-semibold mb-3 text-white/95">AI Analysis</h3>
                    <p className="text-white/90 text-sm leading-relaxed">{riskData.ai_explanation}</p>
                  </div>
                )}

                {riskData.recommendations && riskData.has_risk && (
                  <div className="bg-orange-500/20 backdrop-blur-3xl rounded-[28px] p-6 border border-orange-400/40 shadow-[0_8px_32px_rgba(0,0,0,0.12)]">
                    <h3 className="text-lg font-semibold mb-3 text-white/95">! Safety Recommendations</h3>
                    <p className="text-white/90 text-sm leading-relaxed">{riskData.recommendations}</p>
                  </div>
                )}
              </>
            )}

            {historicalWeather && (
              <div className="bg-black/40 backdrop-blur-2xl rounded-3xl p-6 border border-white/20 shadow-2xl">
                <h3 className="text-lg font-semibold mb-4 text-white">Last 3 Days Weather</h3>
                <div className="space-y-3">
                  {historicalWeather.historical_data.map((day, idx) => (
                    <div key={idx} className="bg-white/10 rounded-xl p-3">
                      <div className="text-white/90 text-sm font-semibold mb-2">{day.date}</div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="text-white/70">
                          Temp: {day.temperature_min?.toFixed(1)}°C - {day.temperature_max?.toFixed(1)}°C
                        </div>
                        <div className="text-white/70">
                          Avg: {day.temperature_mean?.toFixed(1)}°C
                        </div>
                        <div className="text-white/70">
                          Rain: {day.precipitation?.toFixed(1)} mm
                        </div>
                        <div className="text-white/70">
                          Wind: {day.wind_speed_max?.toFixed(0)} km/h
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}


          </div>
        )}

        {page === 'alerts' && (
          <div className="fixed left-6 top-24 w-96">
            <div className="bg-black/40 backdrop-blur-2xl rounded-3xl p-6 border border-white/20 shadow-2xl">
              <h2 className="text-2xl font-bold mb-4 text-white">Subscribe to Alerts</h2>
              <div className="space-y-3">
                <div className="relative">
                  <input
                    type="text"
                    value={locationSearch}
                    onChange={(e) => handleLocationSearch(e.target.value)}
                    onFocus={() => setShowLocationDropdown(true)}
                    placeholder="Search location for alerts..."
                    className="w-full px-5 py-3 bg-white/90 backdrop-blur-md border-0 rounded-2xl text-base focus:ring-2 focus:ring-blue-400"
                  />
                  
                  {showLocationDropdown && locationResults.length > 0 && (
                    <div className="absolute z-50 w-full mt-2 bg-white rounded-2xl shadow-2xl max-h-64 overflow-y-auto">
                      {locationResults.map((loc, idx) => (
                        <button
                          key={idx}
                          onClick={() => {
                            setSelectedLocation(loc);
                            setLocationSearch(loc.display_name);
                            setShowLocationDropdown(false);
                          }}
                          className="w-full text-left px-4 py-3 hover:bg-blue-50 border-b border-gray-100 last:border-b-0 transition"
                        >
                          <div className="font-semibold text-gray-800">{loc.name}</div>
                          <div className="text-xs text-gray-500">
                            {loc.admin1 && `${loc.admin1}, `}{loc.country}
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                
                {selectedLocation && (
                  <div className="bg-blue-500/20 rounded-xl p-3 border border-blue-400/30">
                    <div className="text-white text-sm">
                      <div className="font-semibold">📍 {selectedLocation.name}</div>
                      {selectedLocation.admin1 && selectedLocation.country && (
                        <div className="text-xs text-white/70 mt-1">
                          {selectedLocation.admin1}, {selectedLocation.country}
                        </div>
                      )}
                    </div>
                  </div>
                )}
                
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Email Address"
                  className="w-full px-5 py-3.5 bg-white/10 backdrop-blur-2xl border border-white/20 rounded-[18px] text-white placeholder-white/60 text-sm focus:outline-none focus:ring-2 focus:ring-white/30"
                />
                <input
                  type="tel"
                  value={sms}
                  onChange={(e) => setSms(e.target.value)}
                  placeholder="SMS (optional)"
                  className="w-full px-5 py-3.5 bg-white/10 backdrop-blur-2xl border border-white/20 rounded-[18px] text-white placeholder-white/60 text-sm focus:outline-none focus:ring-2 focus:ring-white/30"
                />
                <button onClick={handleAlertSubscribe} className="w-full py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-2xl hover:from-blue-600 hover:to-purple-700 font-semibold shadow-lg">Subscribe</button>
              </div>
            </div>
          </div>
        )}

        {page === 'saved' && (
          <div className="fixed left-6 top-24 w-96 max-h-[calc(100vh-120px)] overflow-y-auto scrollbar-hide space-y-3">
            <div className="bg-white/5 backdrop-blur-3xl rounded-[28px] p-6 border border-white/20 shadow-[0_8px_32px_rgba(0,0,0,0.08)]">
              <h2 className="text-xl font-semibold mb-4 text-white/95">Saved Locations</h2>
              
              {savedLocations.length === 0 ? (
                <div className="text-center py-8">
                  <div className="text-white/60 text-sm mb-2">No saved locations yet</div>
                  <div className="text-white/40 text-xs">Search and save locations from the Home page</div>
                </div>
              ) : (
                <div className="space-y-3">
                  {savedLocations.map((loc) => (
                    <div key={loc.id} className="bg-white/8 backdrop-blur-xl rounded-[20px] p-4 border border-white/15">
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex-1">
                          <div className="text-white/95 font-semibold text-sm">{loc.name}</div>
                          {loc.risk_score !== undefined && (
                            <div className={`text-lg font-bold mt-1 ${getRiskColor(loc.risk_score)}`}>
                              {loc.risk_score.toFixed(1)}% Risk
                            </div>
                          )}
                        </div>
                        <div className="flex gap-2">
                          <button 
                            onClick={() => handleCheckPrediction(loc.location_id || loc.name)} 
                            className="px-3 py-1.5 bg-white/90 text-gray-900 rounded-full text-xs font-semibold hover:bg-white transition-all"
                          >
                            View
                          </button>
                          <button 
                            onClick={() => handleDeleteLocation(loc.id)} 
                            className="px-3 py-1.5 bg-red-500/90 text-white rounded-full text-xs font-semibold hover:bg-red-600/90 transition-all"
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                      
                      {loc.disaster_type && loc.disaster_type !== 'none' && (
                        <div className="text-white/70 text-xs mt-2">
                          Type: {loc.disaster_type.replace('_', ' ').toUpperCase()}
                        </div>
                      )}
                      
                      {loc.timestamp && (
                        <div className="text-white/50 text-xs mt-2">
                          Predicted: {new Date(loc.timestamp).toLocaleDateString()} at {new Date(loc.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {page === 'chat' && (
          <div className="fixed left-6 top-24 w-96 h-[calc(100vh-120px)]">
            <div className="bg-white/5 backdrop-blur-3xl rounded-[28px] border border-white/20 shadow-[0_8px_32px_rgba(0,0,0,0.08)] h-full flex flex-col">
              <div className="p-6 border-b border-white/20 flex-shrink-0">
                <h2 className="text-xl font-semibold text-white/95">AI Assistant</h2>
                <p className="text-white/70 text-sm mt-1">Ask about weather and disaster risks</p>
              </div>
              
              <div className="flex-1 overflow-y-auto p-6 space-y-3 min-h-0" style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(255,255,255,0.3) transparent' }}>
                {chatHistory.length === 0 ? (
                  <div className="text-center py-8">
                    <div className="text-white/60 text-sm mb-2">No messages yet</div>
                    <div className="text-white/40 text-xs">Start a conversation with the AI assistant</div>
                  </div>
                ) : (
                  chatHistory.map((msg, idx) => (
                    <div key={idx} className={`${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
                      <div className={`inline-block max-w-[80%] p-3 rounded-[18px] ${
                        msg.role === 'user' 
                          ? 'bg-white/15 backdrop-blur-2xl border border-white/20 text-white' 
                          : 'bg-white/8 backdrop-blur-2xl border border-white/15 text-white/95'
                      }`}>
                        <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                      </div>
                    </div>
                  ))
                )}
              </div>
              
              <div className="p-4 border-t border-white/20 flex-shrink-0">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={chatMessage}
                    onChange={(e) => setChatMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleChatSend()}
                    placeholder="Ask a question..."
                    className="flex-1 px-4 py-3 bg-white/10 backdrop-blur-2xl border border-white/20 rounded-[18px] text-white placeholder-white/60 text-sm focus:outline-none focus:ring-2 focus:ring-white/30"
                  />
                  <button 
                    onClick={handleChatSend}
                    className="px-6 py-3 bg-white/10 backdrop-blur-2xl border border-white/20 text-white rounded-[18px] hover:bg-white/15 font-semibold transition-all"
                  >
                    Send
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {page === 'sos' && (
          <div className="fixed left-6 top-24 w-96 max-h-[calc(100vh-120px)] overflow-y-auto scrollbar-hide space-y-3">
            {/* SOS Send Card */}
            <div className="bg-white/5 backdrop-blur-3xl rounded-[28px] p-6 border border-red-400/30 shadow-[0_8px_32px_rgba(0,0,0,0.08)]">
              <h2 className="text-xl font-semibold mb-4 text-white/95">Emergency SOS</h2>
              
              {!currentUserLocation ? (
                <div className="space-y-3">
                  <div className="bg-white/10 backdrop-blur-xl rounded-[18px] p-4 border border-white/20">
                    <p className="text-white/90 text-sm mb-2">Location access required to send SOS alerts</p>
                    <p className="text-white/70 text-xs">We need your location to help others find you in an emergency</p>
                  </div>
                  <button 
                    onClick={requestLocationForSOS}
                    className="w-full py-3.5 bg-white/15 backdrop-blur-2xl border border-white/30 text-white rounded-[18px] hover:bg-white/20 font-semibold transition-all"
                  >
                    Enable Location Access
                  </button>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="bg-white/10 backdrop-blur-xl rounded-[18px] p-3 border border-white/20">
                    <p className="text-white/90 text-xs">
                      Location: {currentUserLocation.latitude.toFixed(4)}, {currentUserLocation.longitude.toFixed(4)}
                    </p>
                  </div>
                  
                  <input
                    type="text"
                    value={sosLocation}
                    onChange={(e) => setSosLocation(e.target.value)}
                    placeholder="Description (optional)"
                    className="w-full px-4 py-3 bg-white/10 backdrop-blur-2xl border border-white/20 rounded-[18px] text-white placeholder-white/60 text-sm focus:outline-none focus:ring-2 focus:ring-red-400/50"
                  />
                  
                  <input
                    type="number"
                    value={peopleCount}
                    onChange={(e) => setPeopleCount(e.target.value)}
                    placeholder="How many people need rescue?"
                    min="1"
                    className="w-full px-4 py-3 bg-white/10 backdrop-blur-2xl border border-white/20 rounded-[18px] text-white placeholder-white/60 text-sm focus:outline-none focus:ring-2 focus:ring-red-400/50"
                  />
                  
                  <button 
                    onClick={handleSOS}
                    className="w-full py-4 bg-red-500/15 backdrop-blur-2xl border border-red-400/30 text-white text-lg font-bold rounded-[18px] hover:bg-red-500/25 shadow-[0_4px_16px_rgba(0,0,0,0.08)] transition-all"
                  >
                    SEND SOS ALERT
                  </button>
                </div>
              )}
            </div>

            {/* Nearby SOS Alerts */}
            <div className="bg-white/5 backdrop-blur-3xl rounded-[28px] p-6 border border-white/20 shadow-[0_8px_32px_rgba(0,0,0,0.08)]">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-white/95">Nearby SOS Alerts</h3>
                {currentUserLocation && (
                  <button 
                    onClick={() => filterNearbyAlerts(currentUserLocation.latitude, currentUserLocation.longitude, sosRadiusFilter)}
                    className="text-xs text-white/70 hover:text-white/90 transition-all"
                  >
                    Refresh
                  </button>
                )}
              </div>
              
              {/* Radius Filter */}
              {currentUserLocation && (
                <div className="mb-4">
                  <label className="text-white/70 text-xs mb-2 block">Search Radius</label>
                  <select
                    value={sosRadiusFilter}
                    onChange={(e) => {
                      const newRadius = parseInt(e.target.value);
                      setSosRadiusFilter(newRadius);
                      filterNearbyAlerts(currentUserLocation.latitude, currentUserLocation.longitude, newRadius);
                    }}
                    className="w-full px-4 py-2 bg-white/10 backdrop-blur-2xl border border-white/20 rounded-[18px] text-white text-sm focus:outline-none focus:ring-2 focus:ring-white/30"
                  >
                    <option value="5" className="bg-gray-800">5 km</option>
                    <option value="10" className="bg-gray-800">10 km</option>
                    <option value="20" className="bg-gray-800">20 km</option>
                    <option value="50" className="bg-gray-800">50 km</option>
                    <option value="100" className="bg-gray-800">100 km</option>
                  </select>
                </div>
              )}
              
              {!currentUserLocation ? (
                <div className="text-center py-8">
                  <div className="text-white/60 text-sm mb-2">Enable location to see nearby alerts</div>
                  <div className="text-white/40 text-xs">Within {sosRadiusFilter}km radius</div>
                </div>
              ) : nearbySOSAlerts.length === 0 ? (
                <div className="text-center py-8">
                  <div className="text-white/60 text-sm mb-2">No SOS alerts nearby</div>
                  <div className="text-white/40 text-xs">Within {sosRadiusFilter}km radius</div>
                </div>
              ) : (
                <div className="space-y-2">
                  {nearbySOSAlerts.map((alert) => (
                    <div key={alert.id} className="bg-red-500/15 backdrop-blur-xl rounded-[18px] p-4 border border-red-400/30">
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex-1">
                          <p className="text-white/95 text-sm font-semibold">
                            {alert.location || 'Emergency Location'}
                          </p>
                          <p className="text-white/70 text-xs mt-1">
                            {alert.peopleCount} {alert.peopleCount === 1 ? 'person' : 'people'} need rescue
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="text-red-400 text-xs font-semibold">
                            {alert.distance.toFixed(1)} km away
                          </p>
                          <p className="text-white/50 text-xs mt-1">
                            {new Date(alert.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </p>
                        </div>
                      </div>
                      {alert.latitude && alert.longitude && (
                        <p className="text-white/50 text-xs mt-2">
                          Coordinates: {alert.latitude.toFixed(4)}, {alert.longitude.toFixed(4)}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* My SOS Alerts */}
            {mySOSAlerts.length > 0 && (
              <div className="bg-white/5 backdrop-blur-3xl rounded-[28px] p-6 border border-white/20 shadow-[0_8px_32px_rgba(0,0,0,0.08)]">
                <h3 className="text-lg font-semibold text-white/95 mb-4">My SOS Alerts</h3>
                <div className="space-y-2">
                  {mySOSAlerts.map((alert) => (
                    <div key={alert.id} className="bg-green-500/10 backdrop-blur-xl rounded-[18px] p-4 border border-green-400/30">
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex-1">
                          <p className="text-white/95 text-sm font-semibold">
                            {alert.location || 'Emergency Location'}
                          </p>
                          <p className="text-white/70 text-xs mt-1">
                            {alert.peopleCount} {alert.peopleCount === 1 ? 'person' : 'people'} need rescue
                          </p>
                          <p className="text-white/50 text-xs mt-1">
                            Sent: {new Date(alert.timestamp).toLocaleString()}
                          </p>
                        </div>
                        <button
                          onClick={() => handleMarkAsSafe(alert.id)}
                          className="px-3 py-1.5 bg-green-500/20 backdrop-blur-xl border border-green-400/40 text-white rounded-full text-xs font-semibold hover:bg-green-500/30 transition-all"
                        >
                          I'm Safe
                        </button>
                      </div>
                      {alert.latitude && alert.longitude && (
                        <p className="text-white/50 text-xs mt-2">
                          Location: {alert.latitude.toFixed(4)}, {alert.longitude.toFixed(4)}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
