/**
 * API Service Layer
 * 
 * Provides functions to interact with the Disaster Early Warning System backend API.
 * Base URL: http://localhost:5000
 */

const API_BASE_URL = 'http://localhost:5000';

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
  
  // Gemini AI
  explainPrediction,
  chatWithGemini,
  
  // Historical Data
  getDisasterHistory,
  getDisasterDetail,
  
  // Map & Visualization
  getHeatmapData,
  getMapMarkers,
  
  // Health
  checkHealth,
  getApiInfo,
};
