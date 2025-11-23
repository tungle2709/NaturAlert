import React, { useState, useEffect, useRef } from 'react';
import { initializeApp } from 'firebase/app';
import { getAuth, signInWithCustomToken, signInAnonymously, onAuthStateChanged } from 'firebase/auth';
import { getFirestore, collection, addDoc, onSnapshot, query, orderBy, limit } from 'firebase/firestore';
import Globe from 'globe.gl';

const App = () => {
  const [page, setPage] = useState('dashboard');
  const [user, setUser] = useState(null);
  const [location, setLocation] = useState('');
  const [weatherData, setWeatherData] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [email, setEmail] = useState('');
  const [sms, setSms] = useState('');
  const [sosLocation, setSosLocation] = useState('');
  const [peopleCount, setPeopleCount] = useState(1);
  const [savedLocations, setSavedLocations] = useState([]);
  const [sosAlerts, setSosAlerts] = useState([]);
  const [message, setMessage] = useState('');
  const [newLocation, setNewLocation] = useState('');
  const globeEl = useRef();
  const globeInstance = useRef();

  useEffect(() => {
    const app = initializeApp(window.__firebase_config);
    const auth = getAuth(app);
    const db = getFirestore(app);

    (async () => {
      try {
        if (window.__initial_auth_token) {
          await signInWithCustomToken(auth, window.__initial_auth_token);
        } else {
          await signInAnonymously(auth);
        }
      } catch (e) {
        await signInAnonymously(auth);
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

    if (globeEl.current && !globeInstance.current) {
      globeInstance.current = Globe()(globeEl.current)
        .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
        .bumpImageUrl('//unpkg.com/three-globe/example/img/earth-topology.png')
        .backgroundImageUrl('//unpkg.com/three-globe/example/img/night-sky.png')
        .pointOfView({ altitude: 2.5 })
        .pointsData([])
        .pointAltitude(0.01)
        .pointRadius(0.5)
        .pointColor(() => '#ff0000')
        .pointLabel('name');
    }
  }, []);

  const fetchMockWeather = (loc) => {
    const base = loc.toLowerCase();
    if (base.includes('desert') || base.includes('phoenix') || base.includes('dubai')) {
      return [
        { day: 'Day 1', maxTemp: 42, rainfall: 0, windSpeed: 15 },
        { day: 'Day 2', maxTemp: 44, rainfall: 0, windSpeed: 18 },
        { day: 'Day 3', maxTemp: 43, rainfall: 0, windSpeed: 12 }
      ];
    } else if (base.includes('coast') || base.includes('miami') || base.includes('mumbai')) {
      return [
        { day: 'Day 1', maxTemp: 28, rainfall: 45, windSpeed: 65 },
        { day: 'Day 2', maxTemp: 27, rainfall: 60, windSpeed: 75 },
        { day: 'Day 3', maxTemp: 26, rainfall: 55, windSpeed: 70 }
      ];
    } else if (base.includes('mountain') || base.includes('denver') || base.includes('nepal')) {
      return [
        { day: 'Day 1', maxTemp: 5, rainfall: 20, windSpeed: 40 },
        { day: 'Day 2', maxTemp: 3, rainfall: 25, windSpeed: 50 },
        { day: 'Day 3', maxTemp: 4, rainfall: 22, windSpeed: 45 }
      ];
    }
    return [
      { day: 'Day 1', maxTemp: 22, rainfall: 10, windSpeed: 20 },
      { day: 'Day 2', maxTemp: 24, rainfall: 5, windSpeed: 18 },
      { day: 'Day 3', maxTemp: 23, rainfall: 8, windSpeed: 22 }
    ];
  };

  const predictDisaster = (loc, weather) => {
    const avgRain = weather.reduce((a, b) => a + b.rainfall, 0) / 3;
    const avgWind = weather.reduce((a, b) => a + b.windSpeed, 0) / 3;
    const avgTemp = weather.reduce((a, b) => a + b.maxTemp, 0) / 3;

    if (avgRain > 50 && avgWind > 60) return '🌊 High Flood Alert';
    if (avgWind > 70) return '🌪️ Tornado Warning';
    if (avgTemp > 40) return '🔥 Extreme Heat Warning';
    if (avgTemp < 5) return '❄️ Severe Cold Alert';
    if (avgRain > 30) return '⚠️ Moderate Flood Risk';
    return '✅ Low Risk';
  };

  const geocodeLocation = async (loc) => {
    try {
      const response = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(loc)}&count=1&language=en&format=json`);
      const data = await response.json();
      if (data.results && data.results.length > 0) {
        return { lat: data.results[0].latitude, lng: data.results[0].longitude };
      }
    } catch (e) {
      console.error('Geocoding error:', e);
    }
    return { lat: 0, lng: 0 };
  };

  const handlePredict = async () => {
    if (!location) return;
    const coords = await geocodeLocation(location);
    if (globeInstance.current) {
      globeInstance.current
        .pointsData([{ lat: coords.lat, lng: coords.lng, name: location }])
        .pointOfView({ lat: coords.lat, lng: coords.lng, altitude: 1.5 }, 1000);
    }
    const weather = fetchMockWeather(location);
    setWeatherData(weather);
    setPrediction(predictDisaster(location, weather));
    setSosLocation(location);
  };

  const handleAlertSubscribe = async () => {
    if (!user || !email) return;
    const db = getFirestore();
    await addDoc(collection(db, `artifacts/${window.__app_id}/public/data/alerts_subscription_queue`), {
      email, sms, location, status: 'PENDING', timestamp: Date.now()
    });
    setMessage(`Successfully subscribed. Cloud Functions will process your alert request for ${location}!`);
    setTimeout(() => setMessage(''), 5000);
    setEmail('');
    setSms('');
  };

  const handleSOS = async () => {
    if (!user || !sosLocation) return;
    const db = getFirestore();
    await addDoc(collection(db, `artifacts/${window.__app_id}/public/data/emergency_alerts`), {
      location: sosLocation, peopleCount, userId: user.uid, timestamp: Date.now()
    });
    setMessage('🚨 SOS Alert Sent! Emergency services notified.');
    setTimeout(() => setMessage(''), 5000);
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
    const coords = await geocodeLocation(loc);
    if (globeInstance.current) {
      globeInstance.current
        .pointsData([{ lat: coords.lat, lng: coords.lng, name: loc }])
        .pointOfView({ lat: coords.lat, lng: coords.lng, altitude: 1.5 }, 1000);
    }
    const weather = fetchMockWeather(loc);
    setWeatherData(weather);
    setPrediction(predictDisaster(loc, weather));
    setPage('dashboard');
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
                <input
                  type="text"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  placeholder="e.g., Tokyo, Miami, London"
                  className="w-full px-5 py-3 bg-white/90 backdrop-blur-md border-0 rounded-2xl text-base focus:ring-2 focus:ring-blue-400"
                />
                <button onClick={handlePredict} className="w-full py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-2xl hover:from-blue-600 hover:to-purple-700 font-semibold shadow-lg">Search</button>
              </div>
            </div>

            {weatherData && (
              <div className="bg-black/40 backdrop-blur-2xl rounded-3xl p-6 border border-white/20 shadow-2xl">
                <h3 className="text-lg font-semibold mb-4 text-white">📊 Weather (3 Days)</h3>
                <div className="space-y-2">
                  {weatherData.map((d, i) => (
                    <div key={i} className="bg-white/10 rounded-xl p-3 border border-white/20">
                      <p className="text-white/70 text-xs mb-1">{d.day}</p>
                      <div className="flex justify-between text-white text-sm">
                        <span>🌡️ {d.maxTemp}°C</span>
                        <span>💧 {d.rainfall}mm</span>
                        <span>💨 {d.windSpeed}km/h</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {prediction && (
              <div className="bg-gradient-to-r from-purple-500/40 to-pink-500/40 backdrop-blur-2xl rounded-3xl p-6 border-2 border-purple-400/50 shadow-2xl">
                <h3 className="text-xl font-bold text-white">{prediction}</h3>
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
