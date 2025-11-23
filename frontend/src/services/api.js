/**
 * API Service Layer
 * 
 * Provides functions to interact with the Disaster Early Warning System backend API.
 * Base URL: http://localhost:8000
 */

const API_BASE_URL = 'http://localhost:8000';

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`API Error (${endpoint}):`, error);
    throw error;
  }
}

// ============================================================================
// Risk Assessment API
// ============================================================================

/**
 * Get current disaster risk assessment for a location
 * @param {string} locationId - Location identifier (default: "default")
 * @returns {Promise<Object>} Risk assessment data
 */
export async function getCurrentRisk(locationId = 'default') {
  return apiFetch(`/api/v1/risk/current?location_id=${encodeURIComponent(locationId)}`);
}

/**
 * Analyze weather data for disaster risk using Gemini AI
 * @param {Object} location - Location data with name, latitude, longitude
 * @param {Object} currentWeather - Current weather conditions
 * @param {Array} historicalWeather - Historical weather data (last 3 days)
 * @returns {Promise<Object>} Risk analysis with disaster prediction
 */
export async function analyzeWeatherData(location, currentWeather, historicalWeather) {
  return apiFetch('/api/v1/risk/analyze', {
    method: 'POST',
    body: JSON.stringify({
      location: location,
      current_weather: currentWeather,
      historical_weather: historicalWeather
    }),
  });
}

/**
 * Get weather trend data for comparison
 * @param {string} locationId - Location identifier
 * @param {number} days - Number of days for trend analysis (default: 7)
 * @returns {Promise<Object>} Trend comparison data
 */
export async function getRiskTrends(locationId = 'default', days = 7) {
  return apiFetch(`/api/v1/risk/trends?location_id=${encodeURIComponent(locationId)}&days=${days}`);
}

/**
 * Get hourly predictions for the next N hours
 * @param {string} locationId - Location identifier
 * @param {number} hours - Number of hours to predict (default: 24)
 * @returns {Promise<Object>} Hourly predictions
 */
export async function getHourlyPredictions(locationId = 'default', hours = 24) {
  return apiFetch(`/api/v1/predictions/hourly?location_id=${encodeURIComponent(locationId)}&hours=${hours}`);
}

// ============================================================================
// Gemini AI API
// ============================================================================

/**
 * Get AI-generated explanation for weather data and predictions
 * @param {Object} weatherData - Current weather observations
 * @param {Object} prediction - ML prediction results
 * @param {string} question - Optional specific question
 * @returns {Promise<Object>} AI explanation
 */
export async function explainPrediction(weatherData, prediction, question = null) {
  return apiFetch('/api/v1/gemini/explain', {
    method: 'POST',
    body: JSON.stringify({
      weather_data: weatherData,
      prediction: prediction,
      question: question,
    }),
  });
}

/**
 * Chat with Gemini AI assistant
 * @param {string} message - User's message
 * @param {Object} context - Optional context (location, risk_score, etc.)
 * @param {string} conversationId - Optional conversation ID for continuity
 * @returns {Promise<Object>} AI response
 */
export async function chatWithGemini(message, context = null, conversationId = null) {
  return apiFetch('/api/v1/gemini/chat', {
    method: 'POST',
    body: JSON.stringify({
      message: message,
      context: context,
      conversation_id: conversationId,
    }),
  });
}

// ============================================================================
// Historical Data API
// ============================================================================

/**
 * Query historical disaster events
 * @param {Object} filters - Filter options
 * @param {string} filters.disaster_type - Filter by disaster type
 * @param {string} filters.start_date - Start date (YYYY-MM-DD)
 * @param {string} filters.end_date - End date (YYYY-MM-DD)
 * @param {string} filters.location - Filter by location
 * @param {string} filters.severity - Filter by severity
 * @param {number} filters.limit - Maximum results (default: 50)
 * @param {number} filters.offset - Pagination offset (default: 0)
 * @returns {Promise<Object>} Historical disasters
 */
export async function getDisasterHistory(filters = {}) {
  const params = new URLSearchParams();
  
  if (filters.disaster_type) params.append('disaster_type', filters.disaster_type);
  if (filters.start_date) params.append('start_date', filters.start_date);
  if (filters.end_date) params.append('end_date', filters.end_date);
  if (filters.location) params.append('location', filters.location);
  if (filters.severity) params.append('severity', filters.severity);
  if (filters.limit) params.append('limit', filters.limit);
  if (filters.offset) params.append('offset', filters.offset);
  
  const queryString = params.toString();
  return apiFetch(`/api/v1/history/disasters${queryString ? '?' + queryString : ''}`);
}

/**
 * Get detailed information about a specific disaster
 * @param {string} disasterId - Disaster identifier
 * @returns {Promise<Object>} Disaster details
 */
export async function getDisasterDetail(disasterId) {
  return apiFetch(`/api/v1/history/disasters/${encodeURIComponent(disasterId)}`);
}

// ============================================================================
// Map & Visualization API
// ============================================================================

/**
 * Get risk heatmap data for map visualization
 * @param {string} region - Optional region filter
 * @param {number} minRisk - Minimum risk score to include (default: 0)
 * @returns {Promise<Object>} Heatmap grid points
 */
export async function getHeatmapData(region = null, minRisk = 0) {
  const params = new URLSearchParams();
  if (region) params.append('region', region);
  if (minRisk > 0) params.append('min_risk', minRisk);
  
  const queryString = params.toString();
  return apiFetch(`/api/v1/map/heatmap${queryString ? '?' + queryString : ''}`);
}

/**
 * Get map markers (weather stations, historical disasters)
 * @param {string} markerType - Type: "weather_stations", "historical_disasters", or "all"
 * @param {number} limit - Maximum markers (default: 100)
 * @returns {Promise<Object>} Map markers
 */
export async function getMapMarkers(markerType = 'all', limit = 100) {
  return apiFetch(`/api/v1/map/markers?marker_type=${markerType}&limit=${limit}`);
}

// ============================================================================
// Location Search API (Direct Open-Meteo API calls)
// ============================================================================

/**
 * Search for locations using Open-Meteo Geocoding API (direct call)
 * @param {string} query - Location name to search
 * @param {number} count - Maximum number of results (default: 10)
 * @returns {Promise<Object>} Location search results
 */
export async function searchLocation(query, count = 10) {
  try {
    const response = await fetch(
      `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(query)}&count=${count}&language=en&format=json`
    );
    
    if (!response.ok) {
      throw new Error(`Geocoding API error: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Format results to match our expected structure
    const results = [];
    if (data.results) {
      data.results.forEach(location => {
        results.push({
          id: location.id,
          name: location.name,
          latitude: location.latitude,
          longitude: location.longitude,
          country: location.country,
          country_code: location.country_code,
          admin1: location.admin1,
          admin2: location.admin2,
          timezone: location.timezone,
          population: location.population,
          display_name: `${location.name}${location.admin1 ? ', ' + location.admin1 : ''}${location.country ? ', ' + location.country : ''}`
        });
      });
    }
    
    return {
      results: results,
      count: results.length,
      query: query
    };
  } catch (error) {
    console.error('Location search error:', error);
    throw error;
  }
}

/**
 * Get current weather data for specific coordinates using Open-Meteo Weather API (direct call)
 * @param {number} latitude - Latitude coordinate
 * @param {number} longitude - Longitude coordinate
 * @returns {Promise<Object>} Weather data
 */
export async function getLocationWeather(latitude, longitude) {
  try {
    const response = await fetch(
      `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m&timezone=auto`
    );
    
    if (!response.ok) {
      throw new Error(`Weather API error: ${response.status}`);
    }
    
    const data = await response.json();
    const current = data.current || {};
    
    return {
      weather: {
        latitude: latitude,
        longitude: longitude,
        temperature: current.temperature_2m,
        humidity: current.relative_humidity_2m,
        pressure: current.surface_pressure,
        wind_speed: current.wind_speed_10m,
        wind_direction: current.wind_direction_10m,
        precipitation: current.precipitation,
        timestamp: current.time,
        timezone: data.timezone
      }
    };
  } catch (error) {
    console.error('Weather fetch error:', error);
    throw error;
  }
}

/**
 * Get historical weather data for the last 3 days using Open-Meteo Historical API
 * @param {number} latitude - Latitude coordinate
 * @param {number} longitude - Longitude coordinate
 * @returns {Promise<Object>} Historical weather data for last 3 days
 */
export async function getHistoricalWeather(latitude, longitude) {
  try {
    // Calculate dates for last 3 days
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 3);
    
    const formatDate = (date) => {
      return date.toISOString().split('T')[0];
    };
    
    const response = await fetch(
      `https://archive-api.open-meteo.com/v1/archive?latitude=${latitude}&longitude=${longitude}&start_date=${formatDate(startDate)}&end_date=${formatDate(endDate)}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,rain_sum,wind_speed_10m_max,wind_direction_10m_dominant&timezone=auto`
    );
    
    if (!response.ok) {
      throw new Error(`Historical Weather API error: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Format the historical data
    const daily = data.daily || {};
    const historicalData = [];
    
    if (daily.time) {
      for (let i = 0; i < daily.time.length; i++) {
        historicalData.push({
          date: daily.time[i],
          temperature_max: daily.temperature_2m_max?.[i],
          temperature_min: daily.temperature_2m_min?.[i],
          temperature_mean: daily.temperature_2m_mean?.[i],
          precipitation: daily.precipitation_sum?.[i],
          rain: daily.rain_sum?.[i],
          wind_speed_max: daily.wind_speed_10m_max?.[i],
          wind_direction: daily.wind_direction_10m_dominant?.[i]
        });
      }
    }
    
    return {
      latitude: latitude,
      longitude: longitude,
      timezone: data.timezone,
      historical_data: historicalData,
      days_count: historicalData.length
    };
  } catch (error) {
    console.error('Historical weather fetch error:', error);
    throw error;
  }
}

// ============================================================================
// Health Check
// ============================================================================

/**
 * Check API health status
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
  return apiFetch('/health');
}

/**
 * Get API root information
 * @returns {Promise<Object>} API info
 */
export async function getApiInfo() {
  return apiFetch('/');
}

// ============================================================================
// Export all functions
// ============================================================================

export default {
  // Risk Assessment
  getCurrentRisk,
  getRiskTrends,
  getHourlyPredictions,
  analyzeWeatherData,
  
  // Gemini AI
  explainPrediction,
  chatWithGemini,
  
  // Historical Data
  getDisasterHistory,
  getDisasterDetail,
  
  // Map & Visualization
  getHeatmapData,
  getMapMarkers,
  
  // Location Search
  searchLocation,
  getLocationWeather,
  getHistoricalWeather,
  
  // Health
  checkHealth,
  getApiInfo,
};
