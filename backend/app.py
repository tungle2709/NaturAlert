"""
Disaster Early Warning System - Backend API

FastAPI application providing REST endpoints for disaster prediction,
weather data, AI explanations, and user management.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import sqlite3
import json
import os

# Import backend services
from services.prediction_engine import PredictionEngine
from services.gemini_service import get_gemini_service, GEMINI_AVAILABLE

# Initialize FastAPI app
app = FastAPI(
    title="Disaster Early Warning System API",
    description="REST API for disaster prediction, weather monitoring, and AI-powered insights",
    version="1.0.0"
)

# Configure CORS for frontend (localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite default port
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
try:
    prediction_engine = PredictionEngine(
        db_path='disaster_data.db',
        models_dir='models',
        use_gemini=True
    )
    print("✓ Prediction engine initialized")
except Exception as e:
    print(f"⚠ Warning: Prediction engine initialization failed: {e}")
    prediction_engine = None

try:
    gemini_service = get_gemini_service() if GEMINI_AVAILABLE else None
    if gemini_service:
        print("✓ Gemini AI service initialized")
except Exception as e:
    print(f"⚠ Warning: Gemini service initialization failed: {e}")
    gemini_service = None

# Database path
DB_PATH = 'disaster_data.db'


# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class WeatherSnapshot(BaseModel):
    """Current weather conditions"""
    temperature: float = Field(..., description="Temperature in Celsius")
    pressure: float = Field(..., description="Atmospheric pressure in hPa")
    humidity: float = Field(..., description="Humidity percentage")
    wind_speed: float = Field(..., description="Wind speed in mph")
    rainfall_24h: float = Field(..., description="24-hour rainfall in mm")
    timestamp: str = Field(..., description="Timestamp of observation")


class ConfidenceInterval(BaseModel):
    """Confidence interval for predictions"""
    lower: float = Field(..., description="Lower bound")
    upper: float = Field(..., description="Upper bound")


class RiskResponse(BaseModel):
    """Risk assessment response"""
    location_id: str
    risk_score: float = Field(..., ge=0, le=100, description="Risk score 0-100%")
    disaster_type: str
    confidence: float = Field(..., ge=0, le=100)
    confidence_interval: ConfidenceInterval
    model_version: str
    timestamp: str
    last_updated: str
    weather_snapshot: WeatherSnapshot
    ai_explanation: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat request to Gemini AI"""
    message: str = Field(..., min_length=1, max_length=1000)
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Chat response from Gemini AI"""
    response: str
    conversation_id: str
    timestamp: str


class ExplainRequest(BaseModel):
    """Request for AI explanation"""
    weather_data: Dict[str, Any]
    prediction: Dict[str, Any]
    question: Optional[str] = None


class ExplainResponse(BaseModel):
    """AI explanation response"""
    explanation: str
    timestamp: str


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/", tags=["Root"])
def root():
    """
    API root endpoint with service information.
    """
    return {
        "message": "Disaster Early Warning System API",
        "version": "1.0.0",
        "status": "operational",
        "services": {
            "prediction_engine": prediction_engine is not None,
            "gemini_ai": gemini_service is not None
        },
        "endpoints": {
            "risk": "/api/v1/risk/*",
            "gemini": "/api/v1/gemini/*",
            "history": "/api/v1/history/*",
            "map": "/api/v1/map/*",
            "predictions": "/api/v1/predictions/*"
        },
        "docs": "/docs"
    }


@app.get("/health", tags=["Root"])
def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": os.path.exists(DB_PATH),
            "prediction_engine": prediction_engine is not None,
            "gemini_ai": gemini_service is not None
        }
    }


# ============================================================================
# Risk Assessment Endpoints
# ============================================================================

@app.get("/api/v1/risk/current", tags=["Risk Assessment"])
def get_current_risk(
    location_id: str = Query(default="default", description="Location identifier (coordinates as lat,lng)")
):
    """
    Get current disaster risk assessment for a location.
    
    Accepts coordinates as location_id (e.g., "43.59,-79.64")
    Fetches last 3 days of weather data from Open-Meteo API
    Uses Gemini AI to analyze disaster risk
    
    Returns:
        - Current risk score (0-100%)
        - Predicted disaster type
        - Confidence level
        - Weather snapshot
        - AI-generated explanation
    """
    try:
        # Check if location_id is coordinates (contains comma)
        if ',' in location_id:
            # Parse coordinates
            try:
                lat_str, lng_str = location_id.split(',')
                latitude = float(lat_str)
                longitude = float(lng_str)
            except (ValueError, IndexError):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid coordinates format. Use: latitude,longitude"
                )
            
            # Fetch weather data from Open-Meteo API
            import requests
            
            # Get current weather
            current_response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m",
                    "timezone": "auto"
                },
                timeout=10
            )
            
            if current_response.status_code != 200:
                raise HTTPException(
                    status_code=503,
                    detail="Failed to fetch current weather data"
                )
            
            current_data = current_response.json()
            current_weather = current_data.get("current", {})
            
            # Get historical weather (last 3 days)
            from datetime import datetime, timedelta
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=3)
            
            historical_response = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,wind_speed_10m_max,surface_pressure_mean",
                    "timezone": "auto"
                },
                timeout=10
            )
            
            if historical_response.status_code != 200:
                raise HTTPException(
                    status_code=503,
                    detail="Failed to fetch historical weather data"
                )
            
            historical_data = historical_response.json()
            daily = historical_data.get("daily", {})
            
            # Format historical weather
            historical_weather = []
            if "time" in daily:
                for i in range(len(daily["time"])):
                    historical_weather.append({
                        "date": daily["time"][i],
                        "temperature_max": daily.get("temperature_2m_max", [])[i] if i < len(daily.get("temperature_2m_max", [])) else None,
                        "temperature_min": daily.get("temperature_2m_min", [])[i] if i < len(daily.get("temperature_2m_min", [])) else None,
                        "temperature_mean": daily.get("temperature_2m_mean", [])[i] if i < len(daily.get("temperature_2m_mean", [])) else None,
                        "precipitation": daily.get("precipitation_sum", [])[i] if i < len(daily.get("precipitation_sum", [])) else None,
                        "wind_speed_max": daily.get("wind_speed_10m_max", [])[i] if i < len(daily.get("wind_speed_10m_max", [])) else None,
                        "pressure_mean": daily.get("surface_pressure_mean", [])[i] if i < len(daily.get("surface_pressure_mean", [])) else None
                    })
            
            # Use Gemini AI to analyze the data
            if gemini_service is None:
                raise HTTPException(
                    status_code=503,
                    detail="Gemini AI service not available"
                )
            
            # Calculate accumulated precipitation over 3 days
            total_precipitation = sum(day.get('precipitation', 0) or 0 for day in historical_weather)
            
            # Calculate average pressure
            avg_pressure = sum(day.get('pressure_mean', 0) or 0 for day in historical_weather if day.get('pressure_mean')) / max(len([d for d in historical_weather if d.get('pressure_mean')]), 1)
            
            # Calculate max wind speed
            max_wind = max((day.get('wind_speed_max', 0) or 0 for day in historical_weather), default=0)
            
            # Simple rule-based risk assessment
            risk_score = 0
            disaster_type = "none"
            
            # Check for flood risk
            if total_precipitation > 100 and avg_pressure < 1000:
                risk_score = min(100, 60 + (total_precipitation - 100) / 5)
                disaster_type = "flood"
            # Check for storm risk
            elif max_wind > 40 and avg_pressure < 1005:
                risk_score = min(100, 50 + (max_wind - 40) / 2)
                disaster_type = "storm"
            # Check for hurricane risk
            elif max_wind > 120 and avg_pressure < 980:
                risk_score = min(100, 80 + (120 - avg_pressure) / 10)
                disaster_type = "hurricane"
            # Check for heatwave
            elif any(day.get('temperature_max', 0) > 35 for day in historical_weather):
                hot_days = sum(1 for day in historical_weather if day.get('temperature_max', 0) > 35)
                risk_score = min(100, 40 + hot_days * 15)
                disaster_type = "heatwave"
            else:
                risk_score = 10  # Low baseline risk
                disaster_type = "none"
            
            # Generate explanation using Gemini
            explanation_prompt = f"""
Explain the disaster risk for this location in 2-3 sentences.

Location: {latitude}, {longitude}
Risk Score: {risk_score:.0f}%
Disaster Type: {disaster_type}

Current Weather:
- Temperature: {current_weather.get('temperature_2m')}°C
- Humidity: {current_weather.get('relative_humidity_2m')}%
- Pressure: {current_weather.get('surface_pressure')} hPa
- Wind Speed: {current_weather.get('wind_speed_10m')} km/h

Last 3 Days Summary:
- Total Precipitation: {total_precipitation:.1f} mm
- Average Pressure: {avg_pressure:.1f} hPa
- Max Wind Speed: {max_wind:.1f} km/h

Provide a clear, concise explanation of the risk level and what weather patterns are contributing to it.
"""
            
            try:
                explanation = gemini_service._safe_generate(
                    explanation_prompt,
                    f"Risk level is {risk_score:.0f}% for {disaster_type}. Based on recent weather patterns including precipitation of {total_precipitation:.1f}mm over 3 days and pressure at {avg_pressure:.1f} hPa."
                )
            except Exception as e:
                explanation = f"Risk level is {risk_score:.0f}% for {disaster_type}. Based on recent weather patterns."
            
            analysis = {
                "risk_score": risk_score,
                "disaster_type": disaster_type,
                "confidence": 75,  # Fixed confidence for rule-based system
                "explanation": explanation
            }
            
            # Format response
            return {
                "location_id": location_id,
                "risk_score": analysis.get('risk_score', 0),
                "disaster_type": analysis.get('disaster_type', 'none'),
                "confidence": analysis.get('confidence', 0),
                "confidence_interval": {
                    "lower": max(0, analysis.get('confidence', 0) - 10),
                    "upper": min(100, analysis.get('confidence', 0) + 10)
                },
                "model_version": "gemini-pro-1.0",
                "timestamp": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "weather_snapshot": {
                    "temperature": current_weather.get('temperature_2m', 0),
                    "pressure": current_weather.get('surface_pressure', 0),
                    "humidity": current_weather.get('relative_humidity_2m', 0),
                    "wind_speed": current_weather.get('wind_speed_10m', 0),
                    "rainfall_24h": current_weather.get('precipitation', 0),
                    "timestamp": current_weather.get('time', datetime.now().isoformat())
                },
                "ai_explanation": analysis.get('explanation', 'No explanation available')
            }
        
        else:
            # Use prediction engine for database locations
            if prediction_engine is None:
                raise HTTPException(
                    status_code=503,
                    detail="Prediction engine not available. Please check model files."
                )
            
            prediction = prediction_engine.get_current_prediction(location_id)
            return prediction
            
    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Weather data request timed out"
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Weather service unavailable: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/api/v1/risk/analyze", tags=["Risk Assessment"])
async def analyze_weather_data(request: Dict[str, Any]):
    """
    Analyze weather data and predict disaster risk using Gemini AI.
    Accepts current and historical weather data from frontend.
    
    Request body:
    {
        "location": {"name": "London", "latitude": 51.5, "longitude": -0.1},
        "current_weather": {...},
        "historical_weather": [...]
    }
    """
    if gemini_service is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini AI service not available. Please check API key configuration."
        )
    
    try:
        location = request.get('location', {})
        current_weather = request.get('current_weather', {})
        historical_weather = request.get('historical_weather', [])
        
        # Create analysis prompt for Gemini
        prompt = f"""
Analyze the following weather data and determine if there are signs of any disaster risk.

Location: {location.get('name', 'Unknown')}
Coordinates: {location.get('latitude')}, {location.get('longitude')}

Current Weather:
- Temperature: {current_weather.get('temperature')}°C
- Humidity: {current_weather.get('humidity')}%
- Pressure: {current_weather.get('pressure')} hPa
- Wind Speed: {current_weather.get('wind_speed')} km/h
- Precipitation: {current_weather.get('precipitation')} mm

Historical Weather (Last 3 Days):
"""
        
        for day in historical_weather:
            prompt += f"""
Date: {day.get('date')}
- Temp Range: {day.get('temperature_min')}°C to {day.get('temperature_max')}°C (Avg: {day.get('temperature_mean')}°C)
- Precipitation: {day.get('precipitation')} mm
- Wind Speed Max: {day.get('wind_speed_max')} km/h
"""
        
        prompt += """

Based on this data, provide a JSON response with the following structure:
{
  "has_disaster_risk": true/false,
  "risk_score": 0-100,
  "disaster_type": "flood|storm|hurricane|heatwave|none",
  "confidence": 0-100,
  "explanation": "Brief explanation of the analysis",
  "recommendations": "Safety recommendations if risk exists"
}

Analyze for these disaster types:
- Flood: Heavy accumulated rainfall, low pressure
- Storm: Rapid pressure changes, high winds
- Hurricane: Very low pressure, extreme winds, heavy rain
- Heatwave: Sustained high temperatures
- Extreme Cold: Sustained low temperatures
- None: Normal weather conditions

Respond ONLY with valid JSON, no additional text.
"""
        
        # Call Gemini AI
        import google.generativeai as genai
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Parse JSON response
        import json
        import re
        
        response_text = response.text.strip()
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        
        analysis = json.loads(response_text)
        
        # Format response
        return {
            "location_id": f"{location.get('latitude')},{location.get('longitude')}",
            "location_name": location.get('name', 'Unknown'),
            "risk_score": analysis.get('risk_score', 0),
            "disaster_type": analysis.get('disaster_type', 'none'),
            "confidence": analysis.get('confidence', 0),
            "has_risk": analysis.get('has_disaster_risk', False),
            "explanation": analysis.get('explanation', ''),
            "recommendations": analysis.get('recommendations', ''),
            "timestamp": datetime.now().isoformat(),
            "weather_snapshot": current_weather
        }
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails, return a safe default
        return {
            "location_id": f"{location.get('latitude')},{location.get('longitude')}",
            "location_name": location.get('name', 'Unknown'),
            "risk_score": 0,
            "disaster_type": "none",
            "confidence": 0,
            "has_risk": False,
            "explanation": "Unable to analyze weather data. Please try again.",
            "recommendations": "",
            "timestamp": datetime.now().isoformat(),
            "weather_snapshot": current_weather
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/api/v1/risk/trends", tags=["Risk Assessment"])
def get_risk_trends(
    location_id: str = Query(default="default", description="Location identifier"),
    days: int = Query(default=7, ge=1, le=30, description="Number of days for trend analysis")
):
    """
    Get weather trend data comparing current patterns to historical averages.
    
    Returns 7-day window data for trend comparison visualization.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Fetch current 7-day window
        cursor.execute("""
            SELECT timestamp, temperature, pressure, wind_speed, rainfall_24h
            FROM weather_historical
            WHERE location_id = ?
            AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp ASC
        """.format(days), (location_id,))
        
        current_data = cursor.fetchall()
        
        # Fetch historical average (from older data)
        cursor.execute("""
            SELECT 
                AVG(temperature) as avg_temp,
                AVG(pressure) as avg_pressure,
                AVG(wind_speed) as avg_wind,
                AVG(rainfall_24h) as avg_rain
            FROM weather_historical
            WHERE location_id = ?
            AND timestamp < datetime('now', '-30 days')
        """, (location_id,))
        
        historical_avg = cursor.fetchone()
        conn.close()
        
        # Format current window data
        current_window = [
            {
                "timestamp": row[0],
                "temperature": float(row[1]) if row[1] is not None else 0.0,
                "pressure": float(row[2]) if row[2] is not None else 0.0,
                "wind_speed": float(row[3]) if row[3] is not None else 0.0,
                "rainfall": float(row[4]) if row[4] is not None else 0.0
            }
            for row in current_data
        ]
        
        # Format historical averages
        historical_data = {
            "temperature": float(historical_avg[0]) if historical_avg[0] is not None else 0.0,
            "pressure": float(historical_avg[1]) if historical_avg[1] is not None else 0.0,
            "wind_speed": float(historical_avg[2]) if historical_avg[2] is not None else 0.0,
            "rainfall": float(historical_avg[3]) if historical_avg[3] is not None else 0.0
        }
        
        # Calculate similarity score (simple comparison)
        if len(current_window) > 0:
            current_avg_temp = sum(d['temperature'] for d in current_window) / len(current_window)
            current_avg_pressure = sum(d['pressure'] for d in current_window) / len(current_window)
            
            # Simple similarity metric (inverse of normalized difference)
            temp_diff = abs(current_avg_temp - historical_data['temperature']) / max(historical_data['temperature'], 1)
            pressure_diff = abs(current_avg_pressure - historical_data['pressure']) / max(historical_data['pressure'], 1)
            
            similarity_score = max(0, 100 - (temp_diff + pressure_diff) * 50)
        else:
            similarity_score = 0.0
        
        return {
            "current_window": current_window,
            "historical_avg": historical_data,
            "similarity_score": round(similarity_score, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch trend data: {str(e)}"
        )


@app.get("/api/v1/predictions/hourly", tags=["Risk Assessment"])
def get_hourly_predictions(
    location_id: str = Query(default="default", description="Location identifier"),
    hours: int = Query(default=24, ge=1, le=72, description="Number of hours to predict")
):
    """
    Get hourly disaster risk predictions for the next N hours.
    
    Note: This is a mock implementation for development.
    In production, this would use time-series forecasting.
    """
    try:
        # For now, return mock data based on current prediction
        if prediction_engine is None:
            raise HTTPException(
                status_code=503,
                detail="Prediction engine not available"
            )
        
        current_prediction = prediction_engine.get_current_prediction(location_id)
        base_risk = current_prediction['risk_score']
        
        # Generate mock hourly predictions with slight variations
        import random
        predictions = []
        
        for hour in range(hours):
            # Add some random variation (±5%)
            variation = random.uniform(-5, 5)
            risk_score = max(0, min(100, base_risk + variation))
            
            predictions.append({
                "hour_offset": hour,
                "timestamp": datetime.now().isoformat(),  # Should add hours in production
                "risk_score": round(risk_score, 2),
                "disaster_type": current_prediction['disaster_type']
            })
        
        return {
            "location_id": location_id,
            "predictions": predictions,
            "generated_at": datetime.now().isoformat(),
            "note": "Mock data for development. Production will use time-series forecasting."
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate hourly predictions: {str(e)}"
        )


# ============================================================================
# Gemini AI Endpoints
# ============================================================================

@app.post("/api/v1/gemini/explain", response_model=ExplainResponse, tags=["Gemini AI"])
def explain_prediction(request: ExplainRequest):
    """
    Get AI-generated explanation for weather data and predictions.
    
    Request body should include:
    - weather_data: Current weather observations
    - prediction: ML prediction results
    - question: Optional specific question to answer
    """
    if gemini_service is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini AI service not available. Please check API key configuration."
        )
    
    try:
        # Convert weather_data dict to DataFrame for Gemini service
        import pandas as pd
        weather_df = pd.DataFrame([request.weather_data])
        
        # Generate explanation
        explanation = gemini_service.generate_explanation(
            weather_df,
            request.prediction
        )
        
        return {
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate explanation: {str(e)}"
        )


@app.post("/api/v1/gemini/chat", response_model=ChatResponse, tags=["Gemini AI"])
def chat_with_gemini(request: ChatRequest):
    """
    Chat with Gemini AI assistant about weather and disaster risks.
    
    Supports contextual conversations with optional conversation history.
    """
    if gemini_service is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini AI service not available. Please check API key configuration."
        )
    
    try:
        # Generate response with context
        response_text = gemini_service.chat_response(
            message=request.message,
            context=request.context or {}
        )
        
        # Generate or use existing conversation ID
        conversation_id = request.conversation_id or f"conv_{datetime.now().timestamp()}"
        
        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )


# ============================================================================
# Historical Data Endpoints
# ============================================================================

@app.get("/api/v1/history/disasters", tags=["Historical Data"])
def get_disaster_history(
    disaster_type: Optional[str] = Query(None, description="Filter by disaster type"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    location: Optional[str] = Query(None, description="Filter by location"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    Query historical disaster events with filtering and pagination.
    
    Supports filtering by:
    - disaster_type: flood, storm, hurricane, extreme_rainfall
    - start_date/end_date: Date range filter
    - location: Location identifier or region
    - severity: low, moderate, high, extreme
    
    Returns paginated list of historical disasters.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Build query with filters
        query = "SELECT * FROM disasters_historical WHERE 1=1"
        params = []
        
        if disaster_type:
            query += " AND disaster_type = ?"
            params.append(disaster_type)
        
        if start_date:
            query += " AND event_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND event_date <= ?"
            params.append(end_date)
        
        if location:
            query += " AND (location_id = ? OR region LIKE ? OR country LIKE ?)"
            params.extend([location, f"%{location}%", f"%{location}%"])
        
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        
        # Add ordering and pagination
        query += " ORDER BY event_date DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]
        
        # Fetch results and convert to dictionaries
        disasters = []
        for row in cursor.fetchall():
            disaster = dict(zip(columns, row))
            disasters.append(disaster)
        
        # Get total count for pagination
        count_query = "SELECT COUNT(*) FROM disasters_historical WHERE 1=1"
        count_params = []
        
        if disaster_type:
            count_query += " AND disaster_type = ?"
            count_params.append(disaster_type)
        
        if start_date:
            count_query += " AND event_date >= ?"
            count_params.append(start_date)
        
        if end_date:
            count_query += " AND event_date <= ?"
            count_params.append(end_date)
        
        if location:
            count_query += " AND (location_id = ? OR region LIKE ? OR country LIKE ?)"
            count_params.extend([location, f"%{location}%", f"%{location}%"])
        
        if severity:
            count_query += " AND severity = ?"
            count_params.append(severity)
        
        cursor.execute(count_query, count_params)
        total_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "disasters": disasters,
            "total_count": total_count,
            "returned_count": len(disasters),
            "limit": limit,
            "offset": offset,
            "has_more": (offset + len(disasters)) < total_count
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch disaster history: {str(e)}"
        )


@app.get("/api/v1/history/disasters/{disaster_id}", tags=["Historical Data"])
def get_disaster_detail(disaster_id: str):
    """
    Get detailed information about a specific historical disaster.
    
    Includes:
    - Disaster event details
    - Weather conditions from 7-day pre-disaster window
    - AI-generated comparison to current conditions (if available)
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Fetch disaster details
        cursor.execute(
            "SELECT * FROM disasters_historical WHERE disaster_id = ?",
            (disaster_id,)
        )
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise HTTPException(
                status_code=404,
                detail=f"Disaster {disaster_id} not found"
            )
        
        columns = [desc[0] for desc in cursor.description]
        disaster = dict(zip(columns, row))
        
        # Fetch weather data from 7-day pre-disaster window (if available)
        # This would require linking disasters to weather data
        # For now, return empty list
        weather_window = []
        
        conn.close()
        
        return {
            "disaster": disaster,
            "weather_window": weather_window,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch disaster details: {str(e)}"
        )


# ============================================================================
# Location Search Endpoints
# ============================================================================

@app.get("/api/v1/location/search", tags=["Location"])
def search_location(
    query: str = Query(..., min_length=1, description="Location name to search"),
    count: int = Query(10, ge=1, le=100, description="Maximum number of results")
):
    """
    Search for locations using Open-Meteo Geocoding API.
    
    Returns location suggestions with coordinates, country, and admin areas.
    """
    try:
        import requests
        
        # Call Open-Meteo Geocoding API
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": query,
                "count": count,
                "language": "en",
                "format": "json"
            },
            timeout=10
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Geocoding API error: {response.status_code}"
            )
        
        data = response.json()
        
        # Format results
        results = []
        if "results" in data:
            for location in data["results"]:
                results.append({
                    "id": location.get("id"),
                    "name": location.get("name"),
                    "latitude": location.get("latitude"),
                    "longitude": location.get("longitude"),
                    "country": location.get("country"),
                    "country_code": location.get("country_code"),
                    "admin1": location.get("admin1"),  # State/Province
                    "admin2": location.get("admin2"),  # County/District
                    "timezone": location.get("timezone"),
                    "population": location.get("population"),
                    "display_name": f"{location.get('name')}, {location.get('admin1', '')}, {location.get('country', '')}".replace(", ,", ",").strip(", ")
                })
        
        return {
            "results": results,
            "count": len(results),
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Location search timed out. Please try again."
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Location search service unavailable: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Location search failed: {str(e)}"
        )


@app.get("/api/v1/location/weather", tags=["Location"])
def get_location_weather(
    latitude: float = Query(..., ge=-90, le=90, description="Latitude"),
    longitude: float = Query(..., ge=-180, le=180, description="Longitude")
):
    """
    Get current weather data for a specific location using Open-Meteo Weather API.
    
    Returns real-time weather observations for the given coordinates.
    """
    try:
        import requests
        
        # Call Open-Meteo Weather API
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m",
                "timezone": "auto"
            },
            timeout=10
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Weather API error: {response.status_code}"
            )
        
        data = response.json()
        current = data.get("current", {})
        
        # Format weather data
        weather_data = {
            "latitude": latitude,
            "longitude": longitude,
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "pressure": current.get("surface_pressure"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_direction": current.get("wind_direction_10m"),
            "precipitation": current.get("precipitation"),
            "timestamp": current.get("time"),
            "timezone": data.get("timezone")
        }
        
        return {
            "weather": weather_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Weather data request timed out. Please try again."
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Weather service unavailable: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch weather data: {str(e)}"
        )


# ============================================================================
# Map and Visualization Endpoints
# ============================================================================

@app.get("/api/v1/map/heatmap", tags=["Map & Visualization"])
def get_heatmap_data(
    region: Optional[str] = Query(None, description="Filter by region"),
    min_risk: float = Query(0, ge=0, le=100, description="Minimum risk score to include")
):
    """
    Get risk heatmap data for map visualization.
    
    Returns grid points with risk scores for rendering heatmap overlay.
    
    Note: This is sample data for development. In production, this would
    query multiple locations and interpolate risk scores across a grid.
    """
    try:
        # For localhost development, return sample heatmap data
        # In production, this would query predictions for multiple locations
        
        # Sample data representing different risk zones
        grid_points = [
            {"lat": 16.0, "lng": 108.0, "risk_score": 75.5, "location_id": "loc_001"},
            {"lat": 16.05, "lng": 108.05, "risk_score": 68.2, "location_id": "loc_002"},
            {"lat": 16.1, "lng": 108.1, "risk_score": 82.3, "location_id": "loc_003"},
            {"lat": 16.15, "lng": 108.15, "risk_score": 71.8, "location_id": "loc_004"},
            {"lat": 16.2, "lng": 108.2, "risk_score": 65.4, "location_id": "loc_005"},
            {"lat": 15.95, "lng": 107.95, "risk_score": 58.9, "location_id": "loc_006"},
            {"lat": 16.25, "lng": 108.25, "risk_score": 79.1, "location_id": "loc_007"},
            {"lat": 16.3, "lng": 108.3, "risk_score": 72.6, "location_id": "loc_008"},
        ]
        
        # Filter by minimum risk if specified
        if min_risk > 0:
            grid_points = [p for p in grid_points if p['risk_score'] >= min_risk]
        
        return {
            "grid_points": grid_points,
            "timestamp": datetime.now().isoformat(),
            "region": region or "default",
            "note": "Sample data for development. Production will query real location data."
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate heatmap data: {str(e)}"
        )


@app.get("/api/v1/map/markers", tags=["Map & Visualization"])
def get_map_markers(
    marker_type: str = Query(..., description="Type of markers: weather_stations, historical_disasters, or all"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of markers")
):
    """
    Get markers for map visualization.
    
    Marker types:
    - weather_stations: Active weather monitoring stations
    - historical_disasters: Past disaster event locations
    - all: Both types
    """
    try:
        markers = []
        
        if marker_type in ["weather_stations", "all"]:
            # Sample weather station markers
            weather_stations = [
                {
                    "type": "weather_station",
                    "id": "ws_001",
                    "lat": 16.0,
                    "lng": 108.0,
                    "name": "Central Station",
                    "status": "active"
                },
                {
                    "type": "weather_station",
                    "id": "ws_002",
                    "lat": 16.2,
                    "lng": 108.2,
                    "name": "North Station",
                    "status": "active"
                }
            ]
            markers.extend(weather_stations)
        
        if marker_type in ["historical_disasters", "all"]:
            # Query historical disasters from database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT disaster_id, disaster_type, event_date, 
                       latitude, longitude, severity, region
                FROM disasters_historical
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                ORDER BY event_date DESC
                LIMIT ?
            """, (limit,))
            
            for row in cursor.fetchall():
                markers.append({
                    "type": "historical_disaster",
                    "id": row[0],
                    "disaster_type": row[1],
                    "date": row[2],
                    "lat": float(row[3]),
                    "lng": float(row[4]),
                    "severity": row[5],
                    "region": row[6]
                })
            
            conn.close()
        
        return {
            "markers": markers[:limit],
            "total_count": len(markers),
            "marker_type": marker_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch map markers: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Try port 5000, fallback to 8000 if occupied
    port = 5000
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        if result == 0:
            port = 8000
            print(f"⚠️  Port 5000 is in use, using port {port} instead")
    except:
        pass
    
    print("\n" + "="*60)
    print("🌊 Disaster Early Warning System API")
    print("="*60)
    print(f"📍 Server: http://localhost:{port}")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print(f"🔍 Health: http://localhost:{port}/health")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
