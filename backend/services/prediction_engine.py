"""
Prediction Engine Module

This module handles ML predictions for disaster risk assessment.
It loads trained models, calculates features from weather data,
runs predictions, and stores results in the database.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import joblib
import json
import uuid
import os

# Import Gemini service for AI explanations
try:
    from backend.services.gemini_service import get_gemini_service
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Gemini service not available. AI explanations will be disabled.")


class PredictionEngine:
    """
    Core prediction engine for disaster risk assessment.
    
    Responsibilities:
    - Load trained ML models
    - Fetch weather data from database
    - Calculate engineered features
    - Run ML predictions
    - Store prediction logs
    """
    
    def __init__(self, db_path: str = 'disaster_data.db', models_dir: str = 'models', use_gemini: bool = True):
        """
        Initialize the prediction engine.
        
        Args:
            db_path: Path to SQLite database
            models_dir: Directory containing trained models
            use_gemini: Whether to use Gemini for AI explanations (default: True)
        """
        self.db_path = db_path
        self.models_dir = models_dir
        self.use_gemini = use_gemini and GEMINI_AVAILABLE
        
        # Load ML models
        binary_model_path = os.path.join(models_dir, 'disaster_prediction_model.pkl')
        type_model_path = os.path.join(models_dir, 'disaster_type_model.pkl')
        
        if not os.path.exists(binary_model_path):
            raise FileNotFoundError(f"Binary model not found at {binary_model_path}")
        if not os.path.exists(type_model_path):
            raise FileNotFoundError(f"Type model not found at {type_model_path}")
        
        self.model_binary = joblib.load(binary_model_path)
        self.model_type = joblib.load(type_model_path)
        
        # Initialize Gemini service if available
        self.gemini_service = None
        if self.use_gemini:
            try:
                self.gemini_service = get_gemini_service()
                print(f"✓ Loaded disaster prediction models from {models_dir}")
                print(f"✓ Gemini AI service initialized")
            except Exception as e:
                print(f"⚠ Gemini service initialization failed: {e}")
                print(f"✓ Loaded disaster prediction models from {models_dir} (AI explanations disabled)")
                self.gemini_service = None
        else:
            print(f"✓ Loaded disaster prediction models from {models_dir} (AI explanations disabled)")
    
    def get_current_prediction(self, location_id: str = 'default') -> Dict:
        """
        Get current disaster prediction for a location.
        
        This is the main entry point for generating predictions.
        
        Args:
            location_id: Location identifier
            
        Returns:
            Dictionary containing prediction results and weather snapshot
        """
        # 1. Fetch latest 7 days of weather data
        weather_data = self.get_latest_weather(location_id, days=7)
        
        # 2. Calculate features
        features = self.calculate_features(weather_data)
        
        # 3. Run ML prediction
        prediction = self.run_ml_prediction(features)
        
        # 4. Generate AI explanation if Gemini is available
        ai_explanation = None
        if self.gemini_service:
            try:
                ai_explanation = self.gemini_service.generate_explanation(
                    weather_data, 
                    prediction
                )
            except Exception as e:
                print(f"Warning: Failed to generate AI explanation: {e}")
                ai_explanation = "AI explanation unavailable."
        
        # 5. Get weather snapshot
        weather_snapshot = self.get_weather_snapshot(weather_data)
        
        # 6. Store prediction log with AI explanation
        self.store_prediction(location_id, prediction, features, ai_explanation)
        
        return {
            **prediction,
            'location_id': location_id,
            'weather_snapshot': weather_snapshot,
            'ai_explanation': ai_explanation,
            'last_updated': prediction['timestamp']
        }
    
    def get_latest_weather(self, location_id: str, days: int = 7) -> pd.DataFrame:
        """
        Fetch latest weather data from database.
        
        Args:
            location_id: Location identifier
            days: Number of days to fetch
            
        Returns:
            DataFrame with weather data
            
        Raises:
            ValueError: If insufficient data is available
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM weather_historical
            WHERE location_id = ?
            AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp ASC
        """.format(days)
        
        df = pd.read_sql_query(query, conn, params=(location_id,))
        conn.close()
        
        if len(df) < 2:
            raise ValueError(
                f"Insufficient weather data for location {location_id}. "
                f"Found {len(df)} records, need at least 2."
            )
        
        return df
    
    def get_weather_snapshot(self, weather_df: pd.DataFrame) -> Dict:
        """
        Get current weather conditions snapshot.
        
        Args:
            weather_df: DataFrame with weather data
            
        Returns:
            Dictionary with current weather metrics
        """
        if len(weather_df) == 0:
            return {}
        
        # Get the most recent weather record
        latest = weather_df.iloc[-1]
        
        return {
            'temperature': float(latest['temperature']),
            'pressure': float(latest['pressure']),
            'humidity': float(latest['humidity']),
            'wind_speed': float(latest['wind_speed']),
            'rainfall_24h': float(latest['rainfall_24h']),
            'timestamp': str(latest['timestamp'])
        }
    
    def calculate_features(self, weather_df: pd.DataFrame) -> Dict:
        """
        Calculate engineered features from weather data.
        
        Features calculated (matching trained model):
        - temperature: Current temperature
        - pressure: Current pressure
        - wind_speed: Current wind speed
        - humidity: Current humidity
        - pressure_drop_7d: Maximum pressure drop over the period
        - wind_spike_max: Maximum wind speed
        - humidity_trend: Linear regression slope of humidity
        - temp_deviation: Temperature standard deviation
        - pressure_velocity: Rate of pressure change
        - wind_gust_ratio: Max wind / mean wind
        
        Args:
            weather_df: DataFrame with weather data
            
        Returns:
            Dictionary with calculated features
            
        Raises:
            ValueError: If insufficient data for feature calculation
        """
        if len(weather_df) < 2:
            raise ValueError("Insufficient weather data for feature calculation")
        
        # Sort by timestamp to ensure correct order
        weather_df = weather_df.sort_values('timestamp').copy()
        
        # Get current weather values (most recent)
        latest = weather_df.iloc[-1]
        
        # Calculate features (matching the order in training_data table)
        features = {
            'temperature': float(latest['temperature']),
            'pressure': float(latest['pressure']),
            'wind_speed': float(latest['wind_speed']),
            'humidity': float(latest['humidity']),
            'pressure_drop_7d': self._calc_pressure_drop(weather_df),
            'wind_spike_max': self._calc_wind_spike(weather_df),
            'humidity_trend': self._calc_humidity_trend(weather_df),
            'temp_deviation': self._calc_temp_deviation(weather_df),
            'pressure_velocity': self._calc_pressure_velocity(weather_df),
            'wind_gust_ratio': self._calc_wind_gust_ratio(weather_df)
        }
        
        return features
    
    def _calc_pressure_drop(self, df: pd.DataFrame) -> float:
        """Calculate maximum pressure drop over the period."""
        if len(df) < 2:
            return 0.0
        max_pressure = df['pressure'].max()
        min_pressure = df['pressure'].min()
        return float(max_pressure - min_pressure)
    
    def _calc_wind_spike(self, df: pd.DataFrame) -> float:
        """Calculate maximum wind speed spike."""
        return float(df['wind_speed'].max())
    
    def _calc_humidity_trend(self, df: pd.DataFrame) -> float:
        """Calculate humidity trend (linear regression slope)."""
        if len(df) < 2:
            return 0.0
        x = np.arange(len(df))
        y = df['humidity'].values
        
        # Handle NaN values
        if np.any(np.isnan(y)):
            y = np.nan_to_num(y, nan=np.nanmean(y))
        
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    def _calc_temp_deviation(self, df: pd.DataFrame) -> float:
        """Calculate temperature standard deviation."""
        return float(df['temperature'].std())
    
    def _calc_pressure_velocity(self, df: pd.DataFrame) -> float:
        """Calculate rate of pressure change."""
        if len(df) < 2:
            return 0.0
        pressure_diff = df['pressure'].diff().abs()
        return float(pressure_diff.mean())
    
    def _calc_wind_gust_ratio(self, df: pd.DataFrame) -> float:
        """Calculate wind gust ratio (max/mean)."""
        mean_wind = df['wind_speed'].mean()
        if mean_wind == 0 or np.isnan(mean_wind):
            return 1.0
        max_wind = df['wind_speed'].max()
        return float(max_wind / mean_wind)
    
    def run_ml_prediction(self, features: Dict) -> Dict:
        """
        Run ML model prediction.
        
        Args:
            features: Dictionary with calculated features
            
        Returns:
            Dictionary with prediction results
        """
        # Prepare feature array in correct order (matching training_data table)
        feature_array = np.array([[
            features['temperature'],
            features['pressure'],
            features['wind_speed'],
            features['humidity'],
            features['pressure_drop_7d'],
            features['wind_spike_max'],
            features['humidity_trend'],
            features['temp_deviation'],
            features['pressure_velocity'],
            features['wind_gust_ratio']
        ]])
        
        # Get probability of disaster from binary model
        risk_proba = self.model_binary.predict_proba(feature_array)[0, 1]
        risk_score = float(risk_proba * 100)
        
        # If risk is high, predict disaster type
        if risk_score > 50:
            disaster_type = self.model_type.predict(feature_array)[0]
            type_proba = self.model_type.predict_proba(feature_array)[0]
            confidence = float(np.max(type_proba) * 100)
        else:
            disaster_type = 'none'
            confidence = float((1 - risk_proba) * 100)
        
        # Calculate confidence interval (simple ±10% for now)
        confidence_lower = max(0.0, risk_score - 10.0)
        confidence_upper = min(100.0, risk_score + 10.0)
        
        return {
            'risk_score': risk_score,
            'disaster_type': disaster_type,
            'confidence': confidence,
            'confidence_interval': {
                'lower': confidence_lower,
                'upper': confidence_upper
            },
            'model_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
    
    def store_prediction(
        self, 
        location_id: str, 
        prediction: Dict, 
        features: Dict,
        ai_explanation: Optional[str] = None
    ):
        """
        Store prediction in database.
        
        Args:
            location_id: Location identifier
            prediction: Prediction results
            features: Feature values used for prediction
            ai_explanation: Optional AI-generated explanation
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        prediction_id = str(uuid.uuid4())
        
        try:
            cursor.execute("""
                INSERT INTO predictions_log (
                    prediction_id, timestamp, location_id, risk_score,
                    confidence_interval_lower, confidence_interval_upper,
                    predicted_disaster_type, feature_values, model_version,
                    ai_explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_id,
                prediction['timestamp'],
                location_id,
                prediction['risk_score'],
                prediction['confidence_interval']['lower'],
                prediction['confidence_interval']['upper'],
                prediction['disaster_type'],
                json.dumps(features),
                prediction['model_version'],
                ai_explanation
            ))
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Error storing prediction: {e}")
            raise
        finally:
            conn.close()
