import React, { useState, useEffect, useRef } from 'react';
import { initializeApp } from 'firebase/app';
import { getAuth, signInAnonymously, onAuthStateChanged } from 'firebase/auth';
import { getFirestore, collection, addDoc, onSnapshot, query, orderBy, limit } from 'firebase/firestore';
import Globe from 'globe.gl';
import * as api from './services/api';

const App = () => {
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
  const [peopleCount, setPeopleCount] = useState(1);
  const [savedLocations, setSavedLocations] = useState([]);
  const [sosAlerts, setSosAlerts] = useState([]);
  const [message, setMessage] = useState('');
  const [newLocation, setNewLocation] = useState('');
  const [chatMessage, setChatMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [locationSearch, setLocationSearch] = useState('');
  const [locationResults, setLocationResults] = useState([]);
  const [showLocationDropdown, setShowLocationDropdown] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState(null);
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
      if (u) {
        const savedRef = collection(db, `artifacts/${window.__app_id}/users/${u.uid}/saved_locations`);
        onSnapshot(savedRef, (snap) => {
          setSavedLocations(snap.docs.map(d => ({ id: d.id, ...d.data() })));
        });

        const sosRef = query(collection(db, `artifacts/${window.__app_id}/public/data/emergency_alerts`), orderBy('timestamp', 'desc'), limit(10));
        onSnapshot(sosRef, (snap) => {
          setSosAlerts(snap.docs.map(d => ({ id: d.id, ...d.data() })));
        });
      }
    });

    // Initialize Globe
    if (globeEl.current && !globeInstance.current) {
      globeInstance.current = Globe()(globeEl.current)
        .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
        .bumpImageUrl('//unpkg.com/three-globe/example/img/earth-topology.png')
        .backgroundImageUrl('//unpkg.com/three-globe/example/img/night-sky.png')
        .pointOfView({ altitude: 2.5 })
        .pointsData([])
        .pointAltitude(0.01)
        .pointRadius(1.2)
        .pointColor('color')
        .pointLabel('name')
        .pointsMerge(false);
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
      
      showMessage('✅ Risk data updated successfully');
    } catch (err) {
      setError(err.message);
      showMessage('⚠️ Failed to fetch risk data: ' + err.message);
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

  const handleSelectLocation = async (locationData) => {
    setSelectedLocation(locationData);
    setLocationSearch(locationData.display_name);
    setShowLocationDropdown(false);
    
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
    
    // Fetch weather data for the selected location
    try {
      const weatherData = await api.getLocationWeather(locationData.latitude, locationData.longitude);
      showMessage(`✅ Selected: ${locationData.display_name}`);
      
      // Update location ID with coordinates for backend
      const locationId = `${locationData.latitude},${locationData.longitude}`;
      setLocation(locationId);
      
      // Optionally fetch risk data immediately
      // await fetchRiskData(locationId);
    } catch (err) {
      showMessage('⚠️ Failed to fetch weather data: ' + err.message);
    }
  };

  const handlePredict = async () => {
    if (!location) return;
    await fetchRiskData(location);
  };

  const handleAlertSubscribe = async () => {
    if (!user || !email) return;
    const db = getFirestore();
    await addDoc(collection(db, `artifacts/${window.__app_id}/public/data/alerts_subscription_queue`), {
      email, sms, location, status: 'PENDING', timestamp: Date.now()
    });
    showMessage(`Successfully subscribed to alerts for ${location}!`);
    setEmail('');
    setSms('');
  };

  const handleSOS = async () => {
    if (!user || !sosLocation) return;
    const db = getFirestore();
    await addDoc(collection(db, `artifacts/${window.__app_id}/public/data/emergency_alerts`), {
      location: sosLocation, peopleCount, userId: user.uid, timestamp: Date.now()
    });
    showMessage('🚨 SOS Alert Sent! Emergency services notified.');
  };

  const handleSaveLocation = async () => {
    if (!user || !newLocation) return;
    const db = getFirestore();
    await addDoc(collection(db, `artifacts/${window.__app_id}/users/${user.uid}/saved_locations`), {
      name: newLocation, timestamp: Date.now()
    });
    setNewLocation('');
  };

  const handleCheckPrediction = async (loc) => {
    setLocation(loc);
    await fetchRiskData(loc);
    setPage('dashboard');
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
      showMessage('⚠️ Chat failed: ' + err.message);
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

  return (
    <div className="relative min-h-screen overflow-hidden">
      <div ref={globeEl} className="fixed inset-0 w-full h-full"></div>
      
      <div className="relative z-10">
        <nav className="fixed top-6 left-1/2 -translate-x-1/2 bg-black/40 backdrop-blur-2xl border border-white/20 rounded-full px-8 py-3 shadow-2xl">
          <div className="flex gap-2 items-center">
            <button onClick={() => setPage('dashboard')} className={`px-6 py-2 rounded-full transition font-medium ${page === 'dashboard' ? 'bg-white text-black shadow-lg' : 'text-white hover:bg-white/20'}`}>🏠 Home</button>
            <button onClick={() => setPage('alerts')} className={`px-6 py-2 rounded-full transition font-medium ${page === 'alerts' ? 'bg-white text-black shadow-lg' : 'text-white hover:bg-white/20'}`}>🔔 Alerts</button>
            <button onClick={() => setPage('saved')} className={`px-6 py-2 rounded-full transition font-medium ${page === 'saved' ? 'bg-white text-black shadow-lg' : 'text-white hover:bg-white/20'}`}>⭐ Saved</button>
            <button onClick={() => setPage('chat')} className={`px-6 py-2 rounded-full transition font-medium ${page === 'chat' ? 'bg-white text-black shadow-lg' : 'text-white hover:bg-white/20'}`}>💬 AI Chat</button>
            <button onClick={() => setPage('sos')} className={`px-6 py-2 rounded-full transition font-medium ${page === 'sos' ? 'bg-red-500 text-white shadow-lg' : 'text-white hover:bg-red-500/30'}`}>🚨 SOS</button>
          </div>
        </nav>

        {user && (
          <div className="fixed top-6 right-6 bg-black/40 backdrop-blur-2xl border border-white/20 rounded-full px-6 py-3 shadow-2xl">
            <span className="text-white text-sm font-medium">👤 {user.uid.slice(0, 8)}</span>
          </div>
        )}

        {message && (
          <div className="fixed top-24 right-6 z-50 animate-fade-in">
            <div className="bg-green-500/90 backdrop-blur-md text-white px-6 py-3 rounded-2xl shadow-2xl font-medium">{message}</div>
          </div>
        )}

        {page === 'dashboard' && (
          <div className="fixed left-6 top-24 w-96 space-y-4 max-h-[calc(100vh-120px)] overflow-y-auto">
            <div className="bg-black/40 backdrop-blur-2xl rounded-3xl p-6 border border-white/20 shadow-2xl">
              <h2 className="text-2xl font-bold mb-4 text-white">🌍 Search Location</h2>
              <div className="space-y-3">
                <div className="relative">
                  <input
                    type="text"
                    value={locationSearch}
                    onChange={(e) => handleLocationSearch(e.target.value)}
                    onFocus={() => locationResults.length > 0 && setShowLocationDropdown(true)}
                    placeholder="Search city, country... (e.g., London, Tokyo)"
                    className="w-full px-5 py-3 bg-white/90 backdrop-blur-md border-0 rounded-2xl text-base focus:ring-2 focus:ring-blue-400"
                  />
                  
                  {showLocationDropdown && locationResults.length > 0 && (
                    <div className="absolute z-50 w-full mt-2 bg-white rounded-2xl shadow-2xl max-h-64 overflow-y-auto">
                      {locationResults.map((loc, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleSelectLocation(loc)}
                          className="w-full text-left px-4 py-3 hover:bg-blue-50 border-b border-gray-100 last:border-b-0 transition"
                        >
                          <div className="font-semibold text-gray-800">{loc.name}</div>
                          <div className="text-xs text-gray-500">
                            {loc.admin1 && `${loc.admin1}, `}{loc.country}
                            {loc.population && ` • Pop: ${loc.population.toLocaleString()}`}
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
                      <div className="text-xs text-white/70 mt-1">
                        Lat: {selectedLocation.latitude.toFixed(4)}, Lng: {selectedLocation.longitude.toFixed(4)}
                      </div>
                    </div>
                  </div>
                )}
                
                <button 
                  onClick={handlePredict} 
                  disabled={loading || !location}
                  className="w-full py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-2xl hover:from-blue-600 hover:to-purple-700 font-semibold shadow-lg disabled:opacity-50"
                >
                  {loading ? 'Loading...' : 'Get Risk Assessment'}
                </button>
              </div>
            </div>

            {riskData && (
              <>
                <div className="bg-gradient-to-r from-purple-500/40 to-pink-500/40 backdrop-blur-2xl rounded-3xl p-6 border-2 border-purple-400/50 shadow-2xl">
                  <h3 className="text-sm font-semibold text-white/70 mb-2">DISASTER RISK</h3>
                  <div className={`text-4xl font-bold ${getRiskColor(riskData.risk_score)}`}>
                    {riskData.risk_score.toFixed(1)}%
                  </div>
                  <div className="text-white text-lg mt-2">
                    {getRiskLabel(riskData.risk_score)}
                  </div>
                  <div className="text-white/80 text-sm mt-2">
                    Type: {riskData.disaster_type.replace('_', ' ').toUpperCase()}
                  </div>
                  <div className="text-white/60 text-xs mt-2">
                    Confidence: {riskData.confidence.toFixed(1)}%
                  </div>
                </div>

                {riskData.weather_snapshot && (
                  <div className="bg-black/40 backdrop-blur-2xl rounded-3xl p-6 border border-white/20 shadow-2xl">
                    <h3 className="text-lg font-semibold mb-4 text-white">🌡️ Current Weather</h3>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="bg-white/10 rounded-xl p-3">
                        <div className="text-white/70 text-xs">Temperature</div>
                        <div className="text-white text-lg font-semibold">{riskData.weather_snapshot.temperature.toFixed(1)}°C</div>
                      </div>
                      <div className="bg-white/10 rounded-xl p-3">
                        <div className="text-white/70 text-xs">Pressure</div>
                        <div className="text-white text-lg font-semibold">{riskData.weather_snapshot.pressure.toFixed(0)} hPa</div>
                      </div>
                      <div className="bg-white/10 rounded-xl p-3">
                        <div className="text-white/70 text-xs">Humidity</div>
                        <div className="text-white text-lg font-semibold">{riskData.weather_snapshot.humidity.toFixed(0)}%</div>
                      </div>
                      <div className="bg-white/10 rounded-xl p-3">
                        <div className="text-white/70 text-xs">Wind Speed</div>
                        <div className="text-white text-lg font-semibold">{riskData.weather_snapshot.wind_speed.toFixed(0)} mph</div>
                      </div>
                    </div>
                  </div>
                )}

                {riskData.ai_explanation && (
                  <div className="bg-black/40 backdrop-blur-2xl rounded-3xl p-6 border border-white/20 shadow-2xl">
                    <h3 className="text-lg font-semibold mb-3 text-white">🤖 AI Explanation</h3>
                    <p className="text-white/90 text-sm leading-relaxed">{riskData.ai_explanation}</p>
                  </div>
                )}
              </>
            )}

            {error && (
              <div className="bg-red-500/40 backdrop-blur-2xl rounded-3xl p-6 border-2 border-red-400/50 shadow-2xl">
                <h3 className="text-white font-semibold">⚠️ Error</h3>
                <p className="text-white/90 text-sm mt-2">{error}</p>
              </div>
            )}
          </div>
        )}

        {page === 'alerts' && (
          <div className="fixed left-6 top-24 w-96">
            <div className="bg-black/40 backdrop-blur-2xl rounded-3xl p-6 border border-white/20 shadow-2xl">
              <h2 className="text-2xl font-bold mb-4 text-white">🔔 Subscribe</h2>
              <div className="space-y-3">
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Email Address"
                  className="w-full px-5 py-3 bg-white/90 rounded-2xl text-base"
                />
                <input
                  type="tel"
                  value={sms}
                  onChange={(e) => setSms(e.target.value)}
                  placeholder="SMS (optional)"
                  className="w-full px-5 py-3 bg-white/90 rounded-2xl text-base"
                />
                <button onClick={handleAlertSubscribe} className="w-full py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-2xl hover:from-blue-600 hover:to-purple-700 font-semibold shadow-lg">Subscribe</button>
              </div>
            </div>
          </div>
        )}

        {page === 'saved' && (
          <div className="fixed left-6 top-24 w-96 max-h-[calc(100vh-120px)] overflow-y-auto">
            <div className="bg-black/40 backdrop-blur-2xl rounded-3xl p-6 border border-white/20 shadow-2xl">
              <h2 className="text-2xl font-bold mb-4 text-white">⭐ Saved Locations</h2>
              <div className="space-y-3 mb-4">
                <input
                  type="text"
                  value={newLocation}
                  onChange={(e) => setNewLocation(e.target.value)}
                  placeholder="Location name"
                  className="w-full px-5 py-3 bg-white/90 rounded-2xl text-base"
                />
                <button onClick={handleSaveLocation} className="w-full py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-2xl hover:from-blue-600 hover:to-purple-700 font-semibold shadow-lg">Save</button>
              </div>
              <div className="space-y-2">
                {savedLocations.map((loc) => (
                  <div key={loc.id} className="flex justify-between items-center p-3 bg-white/10 rounded-xl border border-white/20">
                    <span className="text-white text-sm font-medium">{loc.name}</span>
                    <button onClick={() => handleCheckPrediction(loc.name)} className="px-4 py-1 bg-white text-black rounded-full text-xs font-semibold hover:bg-white/90">Check</button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {page === 'chat' && (
          <div className="fixed left-6 top-24 w-96 max-h-[calc(100vh-120px)] flex flex-col">
            <div className="bg-black/40 backdrop-blur-2xl rounded-3xl border border-white/20 shadow-2xl flex-1 flex flex-col">
              <div className="p-6 border-b border-white/20">
                <h2 className="text-2xl font-bold text-white">💬 AI Assistant</h2>
                <p className="text-white/60 text-sm mt-1">Ask about weather and disaster risks</p>
              </div>
              
              <div className="flex-1 overflow-y-auto p-6 space-y-3">
                {chatHistory.map((msg, idx) => (
                  <div key={idx} className={`${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
                    <div className={`inline-block max-w-[80%] p-3 rounded-2xl ${
                      msg.role === 'user' 
                        ? 'bg-blue-500 text-white' 
                        : 'bg-white/10 text-white'
                    }`}>
                      <p className="text-sm">{msg.content}</p>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="p-4 border-t border-white/20">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={chatMessage}
                    onChange={(e) => setChatMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleChatSend()}
                    placeholder="Ask a question..."
                    className="flex-1 px-4 py-2 bg-white/90 rounded-2xl text-sm"
                  />
                  <button 
                    onClick={handleChatSend}
                    className="px-6 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-2xl hover:from-blue-600 hover:to-purple-700 font-semibold"
                  >
                    Send
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {page === 'sos' && (
          <div className="fixed left-6 top-24 w-96 max-h-[calc(100vh-120px)] overflow-y-auto space-y-4">
            <div className="bg-red-500/40 backdrop-blur-2xl rounded-3xl p-6 border-2 border-red-400/50 shadow-2xl">
              <h2 className="text-2xl font-bold mb-4 text-white">🚨 Emergency</h2>
              <div className="space-y-3">
                <input
                  type="text"
                  value={sosLocation}
                  onChange={(e) => setSosLocation(e.target.value)}
                  placeholder="Current Location"
                  className="w-full px-5 py-3 bg-white/90 rounded-2xl text-base"
                />
                <input
                  type="number"
                  value={peopleCount}
                  onChange={(e) => setPeopleCount(e.target.value)}
                  placeholder="People Count"
                  className="w-full px-5 py-3 bg-white/90 rounded-2xl text-base"
                />
                <button onClick={handleSOS} className="w-full py-4 bg-gradient-to-r from-red-600 to-red-700 text-white text-lg font-bold rounded-2xl hover:from-red-700 hover:to-red-800 shadow-2xl">SEND SOS</button>
              </div>
            </div>

            <div className="bg-black/40 backdrop-blur-2xl rounded-3xl p-6 border border-white/20 shadow-2xl">
              <h3 className="text-lg font-bold mb-3 text-white">Recent Alerts</h3>
              <div className="space-y-2">
                {sosAlerts.map((alert) => (
                  <div key={alert.id} className="p-3 bg-red-500/20 rounded-xl border border-red-400/30">
                    <p className="text-white text-sm font-semibold">📍 {alert.location}</p>
                    <p className="text-xs text-white/70">👥 {alert.peopleCount} | {alert.userId?.slice(0, 8)}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
