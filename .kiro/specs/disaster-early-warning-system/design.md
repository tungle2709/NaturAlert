# Design Document

## Overview

The Disaster Early Warning System is a full-stack cloud-native application built on Google Cloud Platform (GCP) that leverages machine learning and AI to predict natural disasters. The system follows a microservices architecture with clear separation between data ingestion, ML training/prediction, backend services, and frontend presentation layers.

### Key Design Principles

1. **Development-First**: Start with localhost development using Python for all backend/ML work
2. **Dataset-Driven**: Use existing CSV weather datasets in `dataset/` folder as primary data source
3. **Incremental Deployment**: Build locally first, then migrate to cloud when ready
4. **Real-time Processing**: 30-minute refresh cycles for weather data and predictions (production)
5. **AI-First**: Gemini API integration for natural language explanations and user interaction
6. **User-Centric**: Modern, responsive web interface with multiple user personas
7. **Reliability**: 99.5% data processing success rate, sub-5-second response times

### Available Datasets

The project includes several weather datasets in the `dataset/` folder:

1. **GlobalWeatherRepository.csv** (107,964 rows)
   - Global weather data for 195+ countries
   - Columns: country, location, lat/lng, temperature, pressure, humidity, wind speed, precipitation, cloud cover, visibility, air quality metrics
   - Real-time snapshot data from May 2024
   - Best for: Location-based weather patterns and global coverage

2. **weather_data.csv** (1,000,000 rows)
   - Large-scale weather observations
   - Best for: ML model training with extensive historical patterns

3. **weather_classification_data.csv** (13,201 rows)
   - Labeled weather types: Rainy, Cloudy, Sunny, Snowy
   - Columns: Temperature, Humidity, Wind Speed, Precipitation %, Cloud Cover, Atmospheric Pressure, UV Index, Season, Visibility, Location, Weather Type
   - Best for: Weather classification and pattern recognition

4. **rain_prediction_2500observations.csv** (2,501 rows)
   - Binary rain prediction dataset
   - Columns: Temperature, Humidity, Wind_Speed, Cloud_Cover, Pressure, Rain (rain/no rain)
   - Best for: Rain prediction model training

5. **top100cities_weather_data.csv** (101 rows)
   - Weather data for top 100 global cities
   - Best for: City-specific monitoring

6. **seattle-weather.csv** (1,462 rows)
   - Historical Seattle weather data
   - Best for: Time-series analysis example

**Data Strategy**:
- Use `weather_classification_data.csv` and `rain_prediction_2500observations.csv` for initial ML model training
- Use `GlobalWeatherRepository.csv` for location-based current weather display
- Simulate disaster events based on extreme weather patterns (high precipitation + low pressure + high wind speed)
- Create synthetic disaster labels by identifying extreme weather conditions in the datasets

### Development Approach

**Phase 1: Local Development (Current Focus)**
- Python backend (Flask/FastAPI) running on localhost:5000
- React frontend running on localhost:3000
- SQLite database for development
- Local ML model training with scikit-learn/XGBoost
- Use existing Excel dataset for historical disaster data
- OpenWeather API (free tier) for weather data
- Gemini API for AI explanations

**Phase 2: Cloud Migration (Future)**
- Deploy backend to Cloud Run
- Migrate to BigQuery for data warehouse
- Use BigQuery ML or Vertex AI for production ML
- Firebase for authentication and notifications
- Cloud Scheduler for automated tasks

### Technology Stack

**Development (Localhost)**:
- **Frontend**: React.js with Material-UI, Leaflet/OpenStreetMap (free alternative to Google Maps)
- **Backend**: Python (Flask/FastAPI)
- **Database**: SQLite (development), PostgreSQL (production-ready)
- **ML Platform**: scikit-learn, XGBoost (local training)
- **AI Service**: Gemini API (gemini-pro model)
- **Authentication**: Simple JWT tokens (development), Firebase Auth (production)
- **Data Processing**: Pandas, NumPy (Jupyter Notebook)
- **APIs**: OpenWeather API (free tier), local CSV datasets

**Production (Cloud - Future)**:
- **Backend**: Python on Cloud Run
- **Database**: BigQuery (data warehouse), Firestore (user data)
- **ML Platform**: BigQuery ML or Vertex AI
- **Authentication**: Firebase Authentication
- **Notifications**: Firebase Cloud Messaging, SendGrid (email)
- **Orchestration**: Cloud Scheduler, Cloud Functions

## Architecture

### Localhost Development Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Sources (Local)                         │
│  Excel Dataset (EM-DAT) | OpenWeather API (Free Tier)           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Jupyter Notebook (Data Processing)                  │
│  - Load Excel dataset                                            │
│  - Fetch historical weather from OpenWeather API                │
│  - Feature engineering                                           │
│  - Export to SQLite                                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SQLite Database (Local)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   disasters  │  │    weather   │  │   features   │          │
│  │   _history   │  │   _history   │  │   _training  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │  predictions │  │     users    │                            │
│  │     _log     │  │              │                            │
│  └──────────────┘  └──────────────┘                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           Python ML Training (scikit-learn/XGBoost)              │
│  - Train models on processed features                            │
│  - Save models as .pkl files                                     │
│  - Evaluate model performance                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         Python Backend (Flask/FastAPI) - localhost:5000          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │  Prediction    │  │   Gemini AI    │  │     Alert      │   │
│  │    Engine      │  │   Integration  │  │    Manager     │   │
│  │ (Load .pkl)    │  │                │  │                │   │
│  └────────────────┘  └────────────────┘  └────────────────┘   │
│                                                                   │
│  ┌────────────────┐  ┌────────────────┐                        │
│  │   REST API     │  │   SQLAlchemy   │                        │
│  │   Endpoints    │  │      ORM       │                        │
│  └────────────────┘  └────────────────┘                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         React Frontend - localhost:3000                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Dashboard   │  │  Monitoring  │  │   History    │         │
│  │     Page     │  │     Page     │  │     Page     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Explainability│ │    Alerts    │  │    Profile   │         │
│  │     Page     │  │    Center    │  │     Page     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                   │
│  ┌──────────────────────────────────────────────────┐           │
│  │    Gemini Chat Assistant + Leaflet Maps          │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### System Architecture Diagram (Production - Future)

```
┌─────────────────────────────────────────────────────────────────┐
│                     External Data Sources                        │
│  NOAA | NASA | Google Weather API | OpenWeather | Gov Datasets  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Data Ingestion Layer (Cloud Functions)         │
│  - Historical data loader                                        │
│  - Real-time weather fetcher (triggered every 30 min)           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         BigQuery                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Raw Weather  │  │  Disasters   │  │  Processed   │          │
│  │    Data      │  │   History    │  │   Features   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
│  ┌──────────────────────────────────────────────────┐           │
│  │         BigQuery ML Models                        │           │
│  │  - disaster_prediction_model                      │           │
│  │  - risk_classification_model                      │           │
│  └──────────────────────────────────────────────────┘           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Backend Service (Cloud Run)                         │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │  Prediction    │  │   Gemini AI    │  │     Alert      │   │
│  │    Engine      │  │   Integration  │  │    Manager     │   │
│  └────────────────┘  └────────────────┘  └────────────────┘   │
│                                                                   │
│  ┌────────────────┐  ┌────────────────┐                        │
│  │   REST API     │  │   WebSocket    │                        │
│  │   Endpoints    │  │   (Real-time)  │                        │
│  └────────────────┘  └────────────────┘                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Firebase Services                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │     Auth     │  │   Firestore  │  │     FCM      │         │
│  │              │  │  (User Data) │  │ (Push Notif) │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Web Frontend (React)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Dashboard   │  │  Monitoring  │  │   History    │         │
│  │     Page     │  │     Page     │  │     Page     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Explainability│ │    Alerts    │  │    Profile   │         │
│  │     Page     │  │    Center    │  │     Page     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                   │
│  ┌──────────────────────────────────────────────────┐           │
│  │         Gemini Chat Assistant                     │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Historical Data Pipeline** (One-time/Periodic):
   - Jupyter Notebook loads historical disaster and weather data
   - Feature engineering creates pre-disaster window features
   - Cleaned data exported to BigQuery tables
   - BigQuery ML trains prediction models

2. **Real-Time Prediction Pipeline** (Every 30 minutes):
   - Cloud Scheduler triggers Cloud Function
   - Cloud Function fetches current weather from APIs
   - Weather data inserted into BigQuery
   - Backend service queries BigQuery ML for prediction
   - If risk > threshold, trigger alert workflow
   - Send weather + prediction to Gemini for explanation
   - Store results in Firestore and send to frontend

3. **User Interaction Flow**:
   - User accesses web dashboard
   - Frontend fetches latest predictions from backend API
   - Dashboard displays risk, weather, trends, and AI insights
   - User can interact with chat assistant (Gemini)
   - User receives push notifications for high-risk events

## Components and Interfaces

### 1. Data Ingestion Layer

#### Historical Data Loader (Jupyter Notebook)

**Purpose**: Load and process historical weather data for ML training

**Components**:
- `data_loader.py`: Loads CSV datasets from `dataset/` folder
- `feature_engineer.py`: Creates disaster labels from extreme weather patterns and calculates features
- `database_exporter.py`: Exports processed data to SQLite (development) or PostgreSQL (production)

**Key Functions**:
```python
def load_weather_datasets() -> dict:
    """Load all CSV datasets from dataset folder"""
    datasets = {
        'global': pd.read_csv('dataset/GlobalWeatherRepository.csv'),
        'classification': pd.read_csv('dataset/weather_classification_data.csv'),
        'rain_prediction': pd.read_csv('dataset/rain_prediction_2500observations.csv'),
        'large_scale': pd.read_csv('dataset/weather_data.csv'),
        'top_cities': pd.read_csv('dataset/top100cities_weather_data.csv')
    }
    return datasets

def create_disaster_labels(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic disaster labels based on extreme weather conditions
    
    Disaster criteria:
    - Flood: High precipitation (>100mm) + Low pressure (<1000 hPa)
    - Storm: High wind speed (>40 mph) + Low pressure (<1005 hPa)
    - Extreme Rainfall: Very high precipitation (>150mm)
    - Hurricane: Very high wind (>74 mph) + Very low pressure (<980 hPa)
    """
    df = weather_df.copy()
    
    # Initialize disaster columns
    df['disaster_occurred'] = 0
    df['disaster_type'] = 'none'
    
    # Define disaster conditions
    flood_mask = (df['precipitation'] > 100) & (df['pressure'] < 1000)
    storm_mask = (df['wind_speed'] > 40) & (df['pressure'] < 1005)
    extreme_rain_mask = df['precipitation'] > 150
    hurricane_mask = (df['wind_speed'] > 74) & (df['pressure'] < 980)
    
    # Apply labels
    df.loc[flood_mask, ['disaster_occurred', 'disaster_type']] = [1, 'flood']
    df.loc[storm_mask, ['disaster_occurred', 'disaster_type']] = [1, 'storm']
    df.loc[extreme_rain_mask, ['disaster_occurred', 'disaster_type']] = [1, 'extreme_rainfall']
    df.loc[hurricane_mask, ['disaster_occurred', 'disaster_type']] = [1, 'hurricane']
    
    return df

def engineer_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate engineered features from weather data"""
    df = weather_df.copy()
    
    # Calculate rolling statistics (simulating 7-day windows)
    df['pressure_drop_7d'] = df['pressure'].rolling(7).max() - df['pressure'].rolling(7).min()
    df['wind_spike_max'] = df['wind_speed'].rolling(7).max()
    df['rain_accumulation_7d'] = df['precipitation'].rolling(7).sum()
    df['humidity_trend'] = df['humidity'].rolling(7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    df['temp_deviation'] = df['temperature'].rolling(7).std()
    df['pressure_velocity'] = df['pressure'].diff().abs().rolling(7).mean()
    df['wind_gust_ratio'] = df['wind_speed'].rolling(7).max() / df['wind_speed'].rolling(7).mean()
    
    return df

def export_to_sqlite(data: pd.DataFrame, table_name: str, db_path: str = 'disaster_data.db') -> bool:
    """Export processed data to SQLite database"""
    import sqlite3
    conn = sqlite3.connect(db_path)
    data.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    return True
```

**Engineered Features**:
- `pressure_drop_7d`: Maximum pressure drop over 7 days (hPa)
- `wind_spike_max`: Maximum wind speed spike (mph)
- `rain_accumulation_7d`: Total rainfall over 7 days (mm)
- `humidity_trend`: Linear regression slope of humidity
- `temp_deviation`: Standard deviation from historical average (°C)
- `pressure_velocity`: Rate of pressure change (hPa/hour)
- `wind_gust_ratio`: Max gust / average wind speed

**Data Preprocessing Pipeline**:

1. **Load Raw Data**: Load CSV files from `dataset/` folder
2. **Data Cleaning**:
   - Handle missing values (forward fill for time series)
   - Remove duplicates
   - Standardize column names
   - Convert units if needed (ensure consistent units)
3. **Feature Engineering**:
   - Calculate rolling window statistics (7-day windows)
   - Create disaster labels from extreme weather patterns
   - Normalize/scale features for ML
4. **Train/Test Split**:
   - 80% training, 20% testing
   - Stratified split to maintain disaster/non-disaster ratio
5. **Export to SQLite**:
   - Store processed data in `disaster_data.db`
   - Create indexes for efficient querying

**Disaster Labeling Strategy** (Synthetic Labels):

Since we don't have explicit disaster event data, we create labels based on extreme weather thresholds:

| Disaster Type | Criteria |
|---------------|----------|
| **Flood** | Precipitation > 100mm AND Pressure < 1000 hPa |
| **Storm** | Wind Speed > 40 mph AND Pressure < 1005 hPa |
| **Extreme Rainfall** | Precipitation > 150mm |
| **Hurricane** | Wind Speed > 74 mph AND Pressure < 980 hPa |

These thresholds are based on meteorological standards and can be adjusted based on model performance.

#### Real-Time Weather Fetcher (Cloud Function)

**Purpose**: Fetch current weather data every 30 minutes

**Trigger**: Cloud Scheduler (cron: `*/30 * * * *`)

**API Integration**:
- Primary: Google Weather API
- Fallback: OpenWeather API (if primary fails)

**Function Signature**:
```python
def fetch_weather(request):
    """
    Triggered by Cloud Scheduler every 30 minutes
    Returns: HTTP 200 on success, 500 on failure
    """
    locations = get_monitored_locations()
    for location in locations:
        weather_data = call_weather_api(location)
        insert_to_bigquery(weather_data)
    return {"status": "success", "locations_updated": len(locations)}
```

**Output Schema** (BigQuery `weather_realtime` table):
```sql
CREATE TABLE weather_realtime (
  timestamp TIMESTAMP,
  location_id STRING,
  latitude FLOAT64,
  longitude FLOAT64,
  temperature FLOAT64,  -- Celsius
  pressure FLOAT64,     -- hPa
  humidity FLOAT64,     -- percentage
  wind_speed FLOAT64,   -- mph
  wind_direction FLOAT64, -- degrees
  rainfall_1h FLOAT64,  -- mm
  rainfall_24h FLOAT64, -- mm
  cloud_cover FLOAT64,  -- percentage
  visibility FLOAT64    -- km
)
PARTITION BY DATE(timestamp)
CLUSTER BY location_id;
```

### 2. Database Layer (SQLite for Localhost)

#### SQLite Database Schema

**Database File**: `disaster_data.db` (local file)

**disasters_historical**:
```sql
CREATE TABLE disasters_historical (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  disaster_id TEXT UNIQUE NOT NULL,
  disaster_type TEXT NOT NULL,  -- flood, storm, hurricane, rainfall
  event_date DATE NOT NULL,
  location_id TEXT,
  country TEXT,
  region TEXT,
  latitude REAL,
  longitude REAL,
  severity TEXT,                -- low, moderate, high, extreme
  impact_description TEXT,
  casualties INTEGER,
  economic_damage REAL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_disasters_date ON disasters_historical(event_date);
CREATE INDEX idx_disasters_type ON disasters_historical(disaster_type);
CREATE INDEX idx_disasters_location ON disasters_historical(location_id);
```

**weather_historical**:
```sql
CREATE TABLE weather_historical (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp DATETIME NOT NULL,
  location_id TEXT NOT NULL,
  latitude REAL,
  longitude REAL,
  temperature REAL,      -- Celsius
  pressure REAL,         -- hPa
  humidity REAL,         -- percentage
  wind_speed REAL,       -- mph
  wind_direction REAL,   -- degrees
  rainfall_1h REAL,      -- mm
  rainfall_24h REAL,     -- mm
  cloud_cover REAL,      -- percentage
  visibility REAL,       -- km
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_weather_hist_location ON weather_historical(location_id);
CREATE INDEX idx_weather_hist_timestamp ON weather_historical(timestamp);
```

**features_training**:
```sql
CREATE TABLE features_training (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  feature_id TEXT UNIQUE NOT NULL,
  disaster_id TEXT,
  location_id TEXT NOT NULL,
  disaster_date DATE,
  window_start_date DATE,      -- 7 days before disaster
  pressure_drop_7d REAL,
  wind_spike_max REAL,
  rain_accumulation_7d REAL,
  humidity_trend REAL,
  temp_deviation REAL,
  pressure_velocity REAL,
  wind_gust_ratio REAL,
  disaster_occurred INTEGER,   -- 0 or 1 (boolean)
  disaster_type TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (disaster_id) REFERENCES disasters_historical(disaster_id)
);

CREATE INDEX idx_features_disaster ON features_training(disaster_id);
CREATE INDEX idx_features_location ON features_training(location_id);
```

**predictions_log**:
```sql
CREATE TABLE predictions_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  prediction_id TEXT UNIQUE NOT NULL,
  timestamp DATETIME NOT NULL,
  location_id TEXT NOT NULL,
  risk_score REAL,                    -- 0-100%
  confidence_interval_lower REAL,
  confidence_interval_upper REAL,
  predicted_disaster_type TEXT,
  feature_values TEXT,                -- JSON string of input features
  model_version TEXT,
  ai_explanation TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_location ON predictions_log(location_id);
CREATE INDEX idx_predictions_timestamp ON predictions_log(timestamp);
```

**users** (for authentication):
```sql
CREATE TABLE users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT UNIQUE NOT NULL,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT,
  display_name TEXT,
  home_location_id TEXT,
  home_latitude REAL,
  home_longitude REAL,
  home_address TEXT,
  alert_threshold REAL DEFAULT 70.0,
  notification_channels TEXT,         -- JSON array: ["email", "push"]
  theme TEXT DEFAULT 'light',
  language TEXT DEFAULT 'en',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  last_login DATETIME
);

CREATE INDEX idx_users_email ON users(email);
```

### 3. Machine Learning Models (Python - Local Training)

#### Model Training with scikit-learn

**Training Script**: `train_models.py`

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import sqlite3

# Load training data from SQLite
conn = sqlite3.connect('disaster_data.db')
df = pd.read_sql_query("SELECT * FROM features_training", conn)

# Prepare features and labels
feature_cols = [
    'pressure_drop_7d',
    'wind_spike_max',
    'rain_accumulation_7d',
    'humidity_trend',
    'temp_deviation',
    'pressure_velocity',
    'wind_gust_ratio'
]

X = df[feature_cols]
y = df['disaster_occurred']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train primary model: Random Forest for binary classification
model_binary = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
model_binary.fit(X_train, y_train)

# Evaluate
y_pred = model_binary.predict(X_test)
y_pred_proba = model_binary.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# Save model
joblib.dump(model_binary, 'models/disaster_prediction_model.pkl')

# Train secondary model: Multi-class for disaster type
df_disasters = df[df['disaster_occurred'] == 1]
X_type = df_disasters[feature_cols]
y_type = df_disasters['disaster_type']

model_type = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model_type.fit(X_type, y_type)

# Save model
joblib.dump(model_type, 'models/disaster_type_model.pkl')

print("Models saved successfully!")
```

#### Model Prediction (in Backend)

```python
import joblib
import numpy as np

# Load models at startup
model_binary = joblib.load('models/disaster_prediction_model.pkl')
model_type = joblib.load('models/disaster_type_model.pkl')

def predict_disaster_risk(features: dict) -> dict:
    """
    Predict disaster risk from weather features
    
    Args:
        features: Dictionary with feature values
        
    Returns:
        Dictionary with risk_score, disaster_type, confidence
    """
    # Prepare feature array
    feature_array = np.array([[
        features['pressure_drop_7d'],
        features['wind_spike_max'],
        features['rain_accumulation_7d'],
        features['humidity_trend'],
        features['temp_deviation'],
        features['pressure_velocity'],
        features['wind_gust_ratio']
    ]])
    
    # Get probability of disaster
    risk_score = model_binary.predict_proba(feature_array)[0, 1] * 100
    
    # If risk is high, predict disaster type
    if risk_score > 50:
        disaster_type = model_type.predict(feature_array)[0]
        type_proba = model_type.predict_proba(feature_array)[0]
        confidence = np.max(type_proba) * 100
    else:
        disaster_type = 'none'
        confidence = (1 - risk_score / 100) * 100
    
    return {
        'risk_score': float(risk_score),
        'disaster_type': disaster_type,
        'confidence': float(confidence),
        'model_version': '1.0.0'
    }
```

#### Model Evaluation Metrics

- **Accuracy**: Target 75%+
- **Precision**: Minimize false positives (unnecessary alerts)
- **Recall**: Maximize true positives (catch real disasters)
- **F1 Score**: Balance precision and recall
- **ROC-AUC**: Measure classification performance

### 4. Backend Service (Python Flask/FastAPI)

#### Technology Choice (Localhost Development)
- **Language**: Python 3.10+
- **Framework**: Flask or FastAPI (FastAPI recommended for async support)
- **Database**: SQLite (development) with SQLAlchemy ORM
- **ML Integration**: scikit-learn, joblib for model loading
- **Deployment**: Local development server (localhost:5000)
- **Future**: Docker container on Cloud Run for production

#### Flask/FastAPI Backend Example

```python
# app.py (FastAPI example)
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
from datetime import datetime

from prediction_engine import PredictionEngine
from gemini_service import GeminiService

app = FastAPI(title="Disaster Early Warning API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
prediction_engine = PredictionEngine()
gemini_service = GeminiService()

# Pydantic models
class RiskResponse(BaseModel):
    location_id: str
    risk_score: float
    disaster_type: str
    confidence: float
    last_updated: str
    weather_snapshot: dict
    ai_explanation: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    context: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

# API Endpoints
@app.get("/")
def root():
    return {"message": "Disaster Early Warning System API", "version": "1.0.0"}

@app.get("/api/v1/risk/current", response_model=RiskResponse)
def get_current_risk(location_id: str = "default"):
    """Get current risk assessment for a location"""
    try:
        prediction = prediction_engine.get_current_prediction(location_id)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/risk/trends")
def get_risk_trends(location_id: str = "default"):
    """Get 7-day trend data for comparison"""
    try:
        # Fetch current and historical data
        conn = sqlite3.connect('disaster_data.db')
        cursor = conn.cursor()
        
        # Current 7-day window
        cursor.execute("""
            SELECT timestamp, temperature, pressure, wind_speed, rainfall_24h
            FROM weather_historical
            WHERE location_id = ?
            AND timestamp >= datetime('now', '-7 days')
            ORDER BY timestamp
        """, (location_id,))
        current_data = cursor.fetchall()
        
        # Historical average (from training data)
        cursor.execute("""
            SELECT AVG(temperature), AVG(pressure), AVG(wind_speed), AVG(rainfall_24h)
            FROM weather_historical
            WHERE location_id = ?
            AND timestamp < datetime('now', '-30 days')
        """, (location_id,))
        historical_avg = cursor.fetchone()
        
        conn.close()
        
        return {
            "current_window": [
                {
                    "timestamp": row[0],
                    "temperature": row[1],
                    "pressure": row[2],
                    "wind_speed": row[3],
                    "rainfall": row[4]
                }
                for row in current_data
            ],
            "historical_avg": {
                "temperature": historical_avg[0],
                "pressure": historical_avg[1],
                "wind_speed": historical_avg[2],
                "rainfall": historical_avg[3]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/gemini/chat", response_model=ChatResponse)
def chat_with_gemini(request: ChatRequest):
    """Chat with Gemini AI assistant"""
    try:
        response_text = gemini_service.chat_response(
            request.message,
            request.context or {}
        )
        
        conversation_id = request.conversation_id or str(datetime.now().timestamp())
        
        return {
            "response": response_text,
            "conversation_id": conversation_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/history/disasters")
def get_disaster_history(
    disaster_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 50
):
    """Query historical disasters"""
    try:
        conn = sqlite3.connect('disaster_data.db')
        cursor = conn.cursor()
        
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
        
        query += " ORDER BY event_date DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        disasters = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "disasters": disasters,
            "total_count": len(disasters)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/map/heatmap")
def get_heatmap_data():
    """Get risk heatmap data for map visualization"""
    try:
        # For localhost, return sample data
        # In production, this would query multiple locations
        return {
            "grid_points": [
                {"lat": 16.0, "lng": 108.0, "risk_score": 75},
                {"lat": 16.1, "lng": 108.1, "risk_score": 65},
                {"lat": 16.2, "lng": 108.2, "risk_score": 80},
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
```

#### API Endpoints

**GET /api/v1/risk/current**

```typescript
// Returns current risk assessment for a location
Response: {
  location_id: string;
  risk_score: number;        // 0-100
  disaster_type: string;
  confidence: number;
  last_updated: string;      // ISO timestamp
  weather_snapshot: {
    temperature: number;
    pressure: number;
    humidity: number;
    wind_speed: number;
    rainfall_24h: number;
  };
  ai_explanation: string;    // from Gemini
}
```

**GET /api/v1/risk/trends**
```typescript
// Returns 7-day trend data for comparison
Response: {
  current_window: WeatherDataPoint[];
  historical_avg: WeatherDataPoint[];
  similarity_score: number;  // 0-100%
}
```

**GET /api/v1/predictions/hourly**
```typescript
// Returns hourly predictions for next 24 hours
Response: {
  predictions: Array<{
    timestamp: string;
    risk_score: number;
    disaster_type: string;
  }>;
}
```

**POST /api/v1/gemini/explain**
```typescript
// Request AI explanation for specific data
Request: {
  weather_data: object;
  prediction: object;
  question?: string;
}
Response: {
  explanation: string;
  risk_summary: string;
  recommendations: string[];
}
```

**POST /api/v1/gemini/chat**
```typescript
// Chat with Gemini assistant
Request: {
  message: string;
  conversation_id?: string;
  context?: object;
}
Response: {
  response: string;
  conversation_id: string;
  suggested_followups: string[];
}
```

**GET /api/v1/history/disasters**
```typescript
// Query historical disasters
Query params: type, start_date, end_date, location, severity
Response: {
  disasters: Array<DisasterEvent>;
  total_count: number;
}
```

**GET /api/v1/map/heatmap**
```typescript
// Returns risk heatmap data for map visualization
Response: {
  grid_points: Array<{
    lat: number;
    lng: number;
    risk_score: number;
  }>;
  timestamp: string;
}
```

**POST /api/v1/alerts/subscribe**
```typescript
// Subscribe user to alerts
Request: {
  user_id: string;
  location_id: string;
  threshold: number;        // 50-90
  channels: string[];       // ["push", "email", "sms"]
}
Response: {
  subscription_id: string;
  status: string;
}
```

#### Prediction Engine Module (Python)

**Purpose**: Execute ML predictions and manage prediction lifecycle

```python
# prediction_engine.py
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import joblib

class PredictionEngine:
    def __init__(self, db_path='disaster_data.db'):
        self.db_path = db_path
        self.model_binary = joblib.load('models/disaster_prediction_model.pkl')
        self.model_type = joblib.load('models/disaster_type_model.pkl')
    
    def get_current_prediction(self, location_id: str) -> Dict:
        """Get current disaster prediction for a location"""
        # 1. Fetch latest 7 days of weather data
        weather_data = self.get_latest_weather(location_id, days=7)
        
        # 2. Calculate features
        features = self.calculate_features(weather_data)
        
        # 3. Run ML prediction
        prediction = self.run_ml_prediction(features)
        
        # 4. Get Gemini explanation
        explanation = self.get_gemini_explanation(weather_data, prediction)
        
        # 5. Store prediction log
        self.store_prediction(location_id, prediction, explanation, features)
        
        # 6. Check alert threshold
        if prediction['risk_score'] > 70:  # Default threshold
            self.trigger_alert(location_id, prediction)
        
        return {
            **prediction,
            'ai_explanation': explanation,
            'weather_snapshot': self.get_weather_snapshot(weather_data)
        }
    
    def get_latest_weather(self, location_id: str, days: int = 7) -> pd.DataFrame:
        """Fetch latest weather data from database"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT * FROM weather_historical
            WHERE location_id = ?
            AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        """.format(days)
        df = pd.read_sql_query(query, conn, params=(location_id,))
        conn.close()
        return df
    
    def calculate_features(self, weather_df: pd.DataFrame) -> Dict:
        """Calculate engineered features from weather data"""
        if len(weather_df) < 2:
            raise ValueError("Insufficient weather data for feature calculation")
        
        # Sort by timestamp
        weather_df = weather_df.sort_values('timestamp')
        
        # Calculate features
        features = {
            'pressure_drop_7d': self.calc_pressure_drop(weather_df),
            'wind_spike_max': self.calc_wind_spike(weather_df),
            'rain_accumulation_7d': self.calc_rain_accumulation(weather_df),
            'humidity_trend': self.calc_humidity_trend(weather_df),
            'temp_deviation': self.calc_temp_deviation(weather_df),
            'pressure_velocity': self.calc_pressure_velocity(weather_df),
            'wind_gust_ratio': self.calc_wind_gust_ratio(weather_df)
        }
        
        return features
    
    def calc_pressure_drop(self, df: pd.DataFrame) -> float:
        """Calculate maximum pressure drop over the period"""
        if len(df) < 2:
            return 0.0
        max_pressure = df['pressure'].max()
        min_pressure = df['pressure'].min()
        return float(max_pressure - min_pressure)
    
    def calc_wind_spike(self, df: pd.DataFrame) -> float:
        """Calculate maximum wind speed spike"""
        return float(df['wind_speed'].max())
    
    def calc_rain_accumulation(self, df: pd.DataFrame) -> float:
        """Calculate total rainfall accumulation"""
        return float(df['rainfall_24h'].sum())
    
    def calc_humidity_trend(self, df: pd.DataFrame) -> float:
        """Calculate humidity trend (linear regression slope)"""
        if len(df) < 2:
            return 0.0
        x = np.arange(len(df))
        y = df['humidity'].values
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    def calc_temp_deviation(self, df: pd.DataFrame) -> float:
        """Calculate temperature standard deviation"""
        return float(df['temperature'].std())
    
    def calc_pressure_velocity(self, df: pd.DataFrame) -> float:
        """Calculate rate of pressure change"""
        if len(df) < 2:
            return 0.0
        pressure_diff = df['pressure'].diff().abs()
        return float(pressure_diff.mean())
    
    def calc_wind_gust_ratio(self, df: pd.DataFrame) -> float:
        """Calculate wind gust ratio (max/mean)"""
        mean_wind = df['wind_speed'].mean()
        if mean_wind == 0:
            return 1.0
        max_wind = df['wind_speed'].max()
        return float(max_wind / mean_wind)
    
    def run_ml_prediction(self, features: Dict) -> Dict:
        """Run ML model prediction"""
        # Prepare feature array
        feature_array = np.array([[
            features['pressure_drop_7d'],
            features['wind_spike_max'],
            features['rain_accumulation_7d'],
            features['humidity_trend'],
            features['temp_deviation'],
            features['pressure_velocity'],
            features['wind_gust_ratio']
        ]])
        
        # Get probability of disaster
        risk_score = self.model_binary.predict_proba(feature_array)[0, 1] * 100
        
        # If risk is high, predict disaster type
        if risk_score > 50:
            disaster_type = self.model_type.predict(feature_array)[0]
            type_proba = self.model_type.predict_proba(feature_array)[0]
            confidence = np.max(type_proba) * 100
        else:
            disaster_type = 'none'
            confidence = (1 - risk_score / 100) * 100
        
        return {
            'risk_score': float(risk_score),
            'disaster_type': disaster_type,
            'confidence': float(confidence),
            'confidence_interval': {
                'lower': max(0, risk_score - 10),
                'upper': min(100, risk_score + 10)
            },
            'model_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
    
    def store_prediction(self, location_id: str, prediction: Dict, 
                        explanation: str, features: Dict):
        """Store prediction in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        import json
        import uuid
        
        cursor.execute("""
            INSERT INTO predictions_log (
                prediction_id, timestamp, location_id, risk_score,
                confidence_interval_lower, confidence_interval_upper,
                predicted_disaster_type, feature_values, model_version,
                ai_explanation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            prediction['timestamp'],
            location_id,
            prediction['risk_score'],
            prediction['confidence_interval']['lower'],
            prediction['confidence_interval']['upper'],
            prediction['disaster_type'],
            json.dumps(features),
            prediction['model_version'],
            explanation
        ))
        
        conn.commit()
        conn.close()
```

#### Gemini Integration Module (Python)

**Purpose**: Generate natural language explanations and handle chat

```python
# gemini_service.py
import google.generativeai as genai
import os
from typing import Dict, List
import pandas as pd

class GeminiService:
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-pro')
    
    def generate_explanation(self, weather_data: pd.DataFrame, 
                           prediction: Dict) -> str:
        """Generate natural language explanation for prediction"""
        # Get latest weather snapshot
        latest = weather_data.iloc[-1] if len(weather_data) > 0 else None
        
        if latest is None:
            return "Insufficient weather data for explanation."
        
        prompt = f"""
You are a disaster early warning system AI assistant.

Current weather conditions:
- Temperature: {latest['temperature']:.1f}°C
- Pressure: {latest['pressure']:.1f} hPa
- Humidity: {latest['humidity']:.1f}%
- Wind Speed: {latest['wind_speed']:.1f} mph
- 24h Rainfall: {latest['rainfall_24h']:.1f} mm

ML Prediction:
- Risk Score: {prediction['risk_score']:.1f}%
- Disaster Type: {prediction['disaster_type']}
- Confidence: {prediction['confidence']:.1f}%

Explain in 2-3 sentences why the risk is at this level, 
referencing specific weather patterns that match historical 
pre-disaster conditions. Be clear and actionable.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"AI explanation unavailable: {str(e)}"
    
    def chat_response(self, message: str, context: Dict) -> str:
        """Generate chat response with context"""
        prompt = f"""
You are a disaster early warning assistant.

Current context:
- Location: {context.get('location', 'Unknown')}
- Current Risk: {context.get('risk_score', 0):.1f}%
- Recent Weather: Temperature {context.get('temperature', 'N/A')}°C, 
  Pressure {context.get('pressure', 'N/A')} hPa

User question: {message}

Provide a helpful, accurate response. Include relevant data 
and historical comparisons when appropriate. Keep responses 
concise (3-4 sentences max).
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Sorry, I'm unable to respond right now: {str(e)}"
    
    def generate_feature_importance_explanation(self, 
                                               feature_importance: Dict[str, float]) -> str:
        """Explain feature importance in simple terms"""
        features_text = '\n'.join([
            f"- {feature}: {importance:.1f}%"
            for feature, importance in feature_importance.items()
        ])
        
        prompt = f"""
Explain why these weather features are important for 
disaster prediction:

{features_text}

Provide a clear explanation for each feature in simple terms 
that a non-technical person can understand.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Explanation unavailable: {str(e)}"
    
    def generate_alert_message(self, prediction: Dict) -> str:
        """Generate user-friendly alert message"""
        prompt = f"""
Generate a concise, urgent alert message for users about an 
impending disaster.

Risk Score: {prediction['risk_score']:.1f}%
Disaster Type: {prediction['disaster_type']}

The message should:
1. Be 1-2 sentences
2. Be clear and actionable
3. Not cause panic but convey urgency
4. Suggest basic preparedness actions

Example: "High flood risk detected (85%). Secure valuables 
and monitor local emergency updates."
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ {prediction['disaster_type'].title()} risk: {prediction['risk_score']:.0f}%. Stay alert and monitor conditions."
```

#### Alert Manager Module

**Purpose**: Handle alert triggering and notification delivery

```typescript
class AlertManager {
  async triggerAlert(
    locationId: string,
    prediction: Prediction
  ): Promise<void> {
    // 1. Get subscribed users for this location
    const subscriptions = await this.getSubscriptions(locationId);
    
    // 2. Filter by threshold
    const eligibleUsers = subscriptions.filter(
      sub => prediction.risk_score >= sub.threshold
    );
    
    // 3. Check for duplicate alerts (6-hour window)
    const recentAlerts = await this.getRecentAlerts(locationId, 6);
    if (recentAlerts.length > 0) {
      return; // Skip duplicate
    }
    
    // 4. Generate alert message with Gemini
    const alertMessage = await this.geminiService.generateAlertMessage(
      prediction
    );
    
    // 5. Send notifications
    for (const user of eligibleUsers) {
      if (user.channels.includes('push')) {
        await this.sendPushNotification(user.fcm_token, alertMessage);
      }
      if (user.channels.includes('email')) {
        await this.sendEmailNotification(user.email, alertMessage);
      }
      if (user.channels.includes('sms')) {
        await this.sendSMSNotification(user.phone, alertMessage);
      }
    }
    
    // 6. Log alert
    await this.logAlert(locationId, prediction, eligibleUsers.length);
  }
  
  async sendPushNotification(
    fcmToken: string,
    message: AlertMessage
  ): Promise<void> {
    await this.fcmClient.send({
      token: fcmToken,
      notification: {
        title: `🚨 ${message.disaster_type} Risk: ${message.risk_score}%`,
        body: message.summary
      },
      data: {
        risk_score: message.risk_score.toString(),
        disaster_type: message.disaster_type,
        timestamp: message.timestamp
      }
    });
  }
}
```

### 5. Firebase Services

#### Authentication
- **Providers**: Google OAuth, Email/Password
- **User Profile Storage**: Firestore `users` collection
- **Session Management**: Firebase Auth tokens (1-hour expiry)

#### Firestore Schema

**users** collection:
```typescript
{
  uid: string;
  email: string;
  display_name: string;
  home_location: {
    location_id: string;
    latitude: number;
    longitude: number;
    address: string;
  };
  preferences: {
    alert_threshold: number;     // 50-90
    notification_channels: string[];
    theme: 'light' | 'dark';
    language: string;
  };
  created_at: Timestamp;
  last_login: Timestamp;
}
```

**alert_subscriptions** collection:
```typescript
{
  subscription_id: string;
  user_id: string;
  location_id: string;
  threshold: number;
  channels: string[];
  active: boolean;
  created_at: Timestamp;
}
```

**alert_history** collection:
```typescript
{
  alert_id: string;
  user_id: string;
  location_id: string;
  risk_score: number;
  disaster_type: string;
  message: string;
  sent_at: Timestamp;
  channels_used: string[];
  read: boolean;
}
```

#### Firebase Cloud Messaging (FCM)
- **Push Notifications**: Delivered to web and mobile clients
- **Token Management**: Stored in Firestore user profiles
- **Payload**: Risk score, disaster type, AI summary

### 6. Frontend Web Application

#### Technology Stack
- **Framework**: React 18 with TypeScript
- **UI Library**: Material-UI (MUI) v5
- **State Management**: Redux Toolkit + RTK Query
- **Routing**: React Router v6
- **Charts**: Recharts
- **Maps**: Google Maps JavaScript API + @react-google-maps/api
- **Real-time**: Socket.io client (WebSocket)
- **Build Tool**: Vite
- **Deployment**: Firebase Hosting

#### Component Architecture

```
src/
├── components/
│   ├── dashboard/
│   │   ├── RiskOverviewCard.tsx
│   │   ├── WeatherSnapshot.tsx
│   │   ├── TrendComparisonChart.tsx
│   │   ├── GeminiInsightBox.tsx
│   │   └── InteractiveMap.tsx
│   ├── monitoring/
│   │   ├── LiveWeatherFeed.tsx
│   │   ├── HourlyPredictionCurve.tsx
│   │   ├── StormTracker.tsx
│   │   └── PressureTimeline.tsx
│   ├── history/
│   │   ├── DisasterArchive.tsx
│   │   ├── DisasterDetailPanel.tsx
│   │   └── HistoricalComparison.tsx
│   ├── explainability/
│   │   ├── FeatureImportanceChart.tsx
│   │   ├── ModelMetrics.tsx
│   │   └── AIExplanation.tsx
│   ├── alerts/
│   │   ├── AlertsList.tsx
│   │   ├── AlertSettings.tsx
│   │   └── NotificationPreferences.tsx
│   ├── chat/
│   │   ├── GeminiChatInterface.tsx
│   │   ├── ChatMessage.tsx
│   │   └── SuggestedQuestions.tsx
│   └── common/
│       ├── Layout.tsx
│       ├── Navigation.tsx
│       ├── ThemeToggle.tsx
│       └── LoadingSpinner.tsx
├── pages/
│   ├── DashboardPage.tsx
│   ├── MonitoringPage.tsx
│   ├── HistoryPage.tsx
│   ├── ExplainabilityPage.tsx
│   ├── AlertsCenterPage.tsx
│   ├── ProfilePage.tsx
│   └── LoginPage.tsx
├── services/
│   ├── api.ts
│   ├── firebase.ts
│   ├── websocket.ts
│   └── maps.ts
├── store/
│   ├── slices/
│   │   ├── riskSlice.ts
│   │   ├── weatherSlice.ts
│   │   ├── alertsSlice.ts
│   │   └── userSlice.ts
│   └── store.ts
└── types/
    ├── risk.ts
    ├── weather.ts
    └── user.ts
```


#### Page Designs

**Dashboard Page (Home)**

Components:
1. **RiskOverviewCard**: Large card showing current risk score with color-coded bar
2. **WeatherSnapshot**: Grid of current weather metrics with icons
3. **TrendComparisonChart**: Line charts comparing current vs historical patterns
4. **GeminiInsightBox**: AI-generated explanation in chat bubble style
5. **InteractiveMap**: Google Maps with risk heatmap overlay

Layout:
```
┌─────────────────────────────────────────────────────┐
│  Navigation Bar                                      │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │  🌊 Flood Risk Today: 78% (HIGH)            │   │
│  │  [████████████████░░░░░░░░]                 │   │
│  │  Last updated: 2 minutes ago                │   │
│  │  [Explain this (AI)]                        │   │
│  └─────────────────────────────────────────────┘   │
│                                                       │
│  ┌──────────────┐  ┌──────────────────────────┐    │
│  │  Weather     │  │  Trend Comparison         │    │
│  │  Snapshot    │  │  [Line Chart]             │    │
│  │  🌡 29°C     │  │                           │    │
│  │  💨 42 mph   │  │                           │    │
│  │  💧 81%      │  │                           │    │
│  └──────────────┘  └──────────────────────────┘    │
│                                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │  💬 Gemini AI Insight                       │   │
│  │  "Heavy rainfall (120mm) and sharp          │   │
│  │  pressure drop match 81% of historical      │   │
│  │  pre-flood patterns..."                     │   │
│  │  [Ask follow-up] [Generate report]          │   │
│  └─────────────────────────────────────────────┘   │
│                                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │  Interactive Map                             │   │
│  │  [Google Maps with heatmap]                 │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**Real-Time Monitoring Page**

Features:
- Auto-refresh every 30 seconds
- Live weather feed with streaming updates
- Hourly prediction curve (next 24 hours)
- Wind vector visualization
- Pressure drop timeline
- Export data button

**History Page**

Features:
- Searchable table of past disasters
- Filters: type, date range, location, severity
- Detail panel with weather conditions
- Gemini-generated comparison to current conditions
- Timeline visualization

**Explainability Page**

Features:
- Feature importance bar chart
- Model performance metrics (accuracy, precision, recall)
- Gemini explanation of feature importance
- Confusion matrix visualization
- Model version history

**Alerts Center Page**

Features:
- Alert history list with timestamps
- Alert threshold slider (50-90%)
- Notification channel toggles (push, email, SMS)
- Location selector
- Test alert button

**Profile Page**

Features:
- User info display
- Home location selector (map + search)
- Theme toggle (light/dark)
- Language selector
- Data usage settings
- Sign out button

#### State Management

**Redux Store Structure**:
```typescript
{
  risk: {
    current: Prediction | null;
    trends: TrendData | null;
    hourlyPredictions: Prediction[];
    loading: boolean;
    error: string | null;
  },
  weather: {
    current: WeatherData | null;
    historical: WeatherData[];
    loading: boolean;
  },
  alerts: {
    history: Alert[];
    settings: AlertSettings;
    unreadCount: number;
  },
  user: {
    profile: UserProfile | null;
    authenticated: boolean;
    preferences: UserPreferences;
  },
  chat: {
    conversations: Record<string, ChatMessage[]>;
    activeConversationId: string | null;
  }
}
```

**RTK Query API Endpoints**:
```typescript
const api = createApi({
  baseQuery: fetchBaseQuery({ baseUrl: '/api/v1' }),
  endpoints: (builder) => ({
    getCurrentRisk: builder.query<Prediction, string>({
      query: (locationId) => `/risk/current?location=${locationId}`,
      pollingInterval: 30000, // Poll every 30 seconds
    }),
    getTrends: builder.query<TrendData, string>({
      query: (locationId) => `/risk/trends?location=${locationId}`,
    }),
    getHourlyPredictions: builder.query<Prediction[], string>({
      query: (locationId) => `/predictions/hourly?location=${locationId}`,
    }),
    chatWithGemini: builder.mutation<ChatResponse, ChatRequest>({
      query: (body) => ({
        url: '/gemini/chat',
        method: 'POST',
        body,
      }),
    }),
    // ... more endpoints
  }),
});
```

#### Real-Time Updates

**WebSocket Connection**:
```typescript
// Establish WebSocket for real-time updates
const socket = io(BACKEND_URL, {
  auth: { token: firebaseAuthToken }
});

socket.on('risk_update', (data: Prediction) => {
  dispatch(updateCurrentRisk(data));
  showNotification('Risk level updated');
});

socket.on('alert', (data: Alert) => {
  dispatch(addAlert(data));
  showPushNotification(data);
});

socket.on('weather_update', (data: WeatherData) => {
  dispatch(updateWeather(data));
});
```

#### Map Integration

**Google Maps Configuration**:
```typescript
const mapOptions = {
  center: { lat: userLocation.lat, lng: userLocation.lng },
  zoom: 8,
  styles: darkModeStyles, // Custom styling
  mapTypeControl: true,
  streetViewControl: false,
};

// Heatmap layer for risk visualization
const heatmapLayer = new google.maps.visualization.HeatmapLayer({
  data: riskDataPoints.map(point => ({
    location: new google.maps.LatLng(point.lat, point.lng),
    weight: point.risk_score / 100,
  })),
  radius: 50,
  gradient: [
    'rgba(0, 255, 0, 0)',
    'rgba(0, 255, 0, 1)',
    'rgba(255, 255, 0, 1)',
    'rgba(255, 165, 0, 1)',
    'rgba(255, 0, 0, 1)',
  ],
});

// Markers for weather stations
weatherStations.forEach(station => {
  new google.maps.Marker({
    position: { lat: station.lat, lng: station.lng },
    map: map,
    icon: weatherStationIcon,
    title: station.name,
  });
});

// Markers for historical disasters
historicalDisasters.forEach(disaster => {
  new google.maps.Marker({
    position: { lat: disaster.lat, lng: disaster.lng },
    map: map,
    icon: disasterIcon,
    title: `${disaster.type} - ${disaster.date}`,
  });
});
```

#### Chart Visualizations

**Trend Comparison Chart** (using Recharts):
```typescript
<ResponsiveContainer width="100%" height={300}>
  <LineChart data={trendData}>
    <CartesianGrid strokeDasharray="3 3" />
    <XAxis dataKey="date" />
    <YAxis />
    <Tooltip />
    <Legend />
    <Line 
      type="monotone" 
      dataKey="current_rainfall" 
      stroke="#2196F3" 
      name="Current Rainfall"
    />
    <Line 
      type="monotone" 
      dataKey="historical_avg_rainfall" 
      stroke="#FF9800" 
      name="Pre-Disaster Average"
      strokeDasharray="5 5"
    />
  </LineChart>
</ResponsiveContainer>
```

**Feature Importance Chart**:
```typescript
<ResponsiveContainer width="100%" height={400}>
  <BarChart data={featureImportance} layout="vertical">
    <CartesianGrid strokeDasharray="3 3" />
    <XAxis type="number" domain={[0, 100]} />
    <YAxis type="category" dataKey="feature" width={150} />
    <Tooltip />
    <Bar dataKey="importance" fill="#2196F3">
      {featureImportance.map((entry, index) => (
        <Cell key={`cell-${index}`} fill={getColorByImportance(entry.importance)} />
      ))}
    </Bar>
  </BarChart>
</ResponsiveContainer>
```

### 7. Cloud Scheduler Configuration

**Weather Fetch Job**:
```yaml
name: fetch-weather-data
schedule: "*/30 * * * *"  # Every 30 minutes
time_zone: UTC
http_target:
  uri: https://[CLOUD_FUNCTION_URL]/fetch-weather
  http_method: POST
  headers:
    Content-Type: application/json
  body: |
    {
      "locations": ["all"]
    }
retry_config:
  retry_count: 3
  max_retry_duration: 300s
  min_backoff_duration: 5s
  max_backoff_duration: 60s
```

**Prediction Job**:
```yaml
name: run-predictions
schedule: "*/30 * * * *"  # Every 30 minutes (offset by 5 min)
time_zone: UTC
http_target:
  uri: https://[CLOUD_RUN_URL]/api/v1/predictions/run
  http_method: POST
  oidc_token:
    service_account_email: [SERVICE_ACCOUNT]
retry_config:
  retry_count: 2
```

**Model Retraining Job**:
```yaml
name: retrain-models
schedule: "0 2 * * 0"  # Weekly on Sunday at 2 AM
time_zone: UTC
http_target:
  uri: https://[CLOUD_RUN_URL]/api/v1/ml/retrain
  http_method: POST
```

## Data Models

### TypeScript Interfaces

```typescript
interface WeatherData {
  timestamp: string;
  location_id: string;
  latitude: number;
  longitude: number;
  temperature: number;      // Celsius
  pressure: number;         // hPa
  humidity: number;         // percentage
  wind_speed: number;       // mph
  wind_direction: number;   // degrees
  rainfall_1h: number;      // mm
  rainfall_24h: number;     // mm
  cloud_cover: number;      // percentage
  visibility: number;       // km
}

interface Prediction {
  prediction_id: string;
  timestamp: string;
  location_id: string;
  risk_score: number;       // 0-100
  confidence_interval: {
    lower: number;
    upper: number;
  };
  disaster_type: 'flood' | 'storm' | 'hurricane' | 'extreme_rainfall';
  feature_values: Features;
  model_version: string;
  ai_explanation?: string;
}

interface Features {
  pressure_drop_7d: number;
  wind_spike_max: number;
  rain_accumulation_7d: number;
  humidity_trend: number;
  temp_deviation: number;
  pressure_velocity: number;
  wind_gust_ratio: number;
}

interface TrendData {
  current_window: WeatherDataPoint[];
  historical_avg: WeatherDataPoint[];
  similarity_score: number;
}

interface WeatherDataPoint {
  date: string;
  rainfall: number;
  pressure: number;
  wind_speed: number;
  temperature: number;
}

interface Alert {
  alert_id: string;
  user_id: string;
  location_id: string;
  risk_score: number;
  disaster_type: string;
  message: string;
  sent_at: string;
  channels_used: ('push' | 'email' | 'sms')[];
  read: boolean;
}

interface UserProfile {
  uid: string;
  email: string;
  display_name: string;
  home_location: {
    location_id: string;
    latitude: number;
    longitude: number;
    address: string;
  };
  preferences: UserPreferences;
  created_at: string;
  last_login: string;
}

interface UserPreferences {
  alert_threshold: number;
  notification_channels: ('push' | 'email' | 'sms')[];
  theme: 'light' | 'dark';
  language: string;
}

interface DisasterEvent {
  disaster_id: string;
  disaster_type: string;
  event_date: string;
  location_id: string;
  latitude: number;
  longitude: number;
  severity: 'low' | 'moderate' | 'high' | 'extreme';
  impact_description: string;
  casualties: number;
  economic_damage: number;
}

interface ChatMessage {
  message_id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

interface ChatContext {
  location: string;
  risk_score: number;
  weather: WeatherData;
  recent_predictions: Prediction[];
}
```

## Error Handling

### Backend Error Handling Strategy

**Error Categories**:
1. **External API Errors**: Weather API failures, Gemini API errors
2. **Database Errors**: BigQuery query failures, Firestore write errors
3. **ML Model Errors**: Prediction failures, model not found
4. **Authentication Errors**: Invalid tokens, expired sessions
5. **Validation Errors**: Invalid input data, missing required fields

**Error Response Format**:
```typescript
interface ErrorResponse {
  error: {
    code: string;           // e.g., "WEATHER_API_UNAVAILABLE"
    message: string;        // User-friendly message
    details?: any;          // Additional error context
    timestamp: string;
  };
  status: number;           // HTTP status code
}
```

**Retry Logic**:
```typescript
async function fetchWithRetry<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  backoff: number = 1000
): Promise<T> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await sleep(backoff * Math.pow(2, i));
    }
  }
  throw new Error('Max retries exceeded');
}
```

**Fallback Mechanisms**:
- Weather API: Primary (Google) → Fallback (OpenWeather)
- Gemini API: Retry with exponential backoff, return cached explanation if unavailable
- BigQuery: Use cached predictions if query fails

### Frontend Error Handling

**Error Boundaries**:
```typescript
class ErrorBoundary extends React.Component {
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    logErrorToService(error, errorInfo);
    this.setState({ hasError: true });
  }
  
  render() {
    if (this.state.hasError) {
      return <ErrorFallbackUI />;
    }
    return this.props.children;
  }
}
```

**API Error Handling**:
```typescript
// RTK Query error handling
const { data, error, isLoading } = useGetCurrentRiskQuery(locationId);

if (error) {
  if ('status' in error) {
    // HTTP error
    if (error.status === 503) {
      return <ServiceUnavailableMessage />;
    }
  }
  return <ErrorMessage error={error} />;
}
```

**User-Friendly Error Messages**:
- "Unable to fetch weather data. Showing last known information."
- "Prediction service temporarily unavailable. Please try again in a few minutes."
- "AI explanation unavailable. Risk assessment is still accurate."

## Testing Strategy

### Unit Testing

**Backend (Jest + Supertest)**:
```typescript
describe('PredictionEngine', () => {
  it('should calculate pressure drop correctly', () => {
    const weatherData = mockWeatherData();
    const features = predictionEngine.calculateFeatures(weatherData);
    expect(features.pressure_drop_7d).toBeCloseTo(15.5, 1);
  });
  
  it('should trigger alert when risk exceeds threshold', async () => {
    const prediction = { risk_score: 85, ...mockPrediction() };
    await predictionEngine.getCurrentPrediction('loc_123');
    expect(alertManager.triggerAlert).toHaveBeenCalled();
  });
});
```

**Frontend (Jest + React Testing Library)**:
```typescript
describe('RiskOverviewCard', () => {
  it('should display high risk with red color', () => {
    render(<RiskOverviewCard risk_score={85} disaster_type="flood" />);
    expect(screen.getByText(/85%/)).toBeInTheDocument();
    expect(screen.getByTestId('risk-bar')).toHaveStyle({ backgroundColor: 'red' });
  });
});
```

### Integration Testing

**API Integration Tests**:
```typescript
describe('GET /api/v1/risk/current', () => {
  it('should return current risk for valid location', async () => {
    const response = await request(app)
      .get('/api/v1/risk/current?location=loc_123')
      .expect(200);
    
    expect(response.body).toHaveProperty('risk_score');
    expect(response.body).toHaveProperty('ai_explanation');
  });
});
```

**BigQuery ML Tests**:
```python
def test_model_prediction():
    features = {
        'pressure_drop_7d': 15.0,
        'wind_spike_max': 45.0,
        'rain_accumulation_7d': 120.0,
        # ...
    }
    prediction = run_ml_prediction(features)
    assert 0 <= prediction['risk_score'] <= 100
    assert prediction['disaster_type'] in ['flood', 'storm', 'hurricane', 'extreme_rainfall']
```

### End-to-End Testing

**Cypress Tests**:
```typescript
describe('Dashboard Flow', () => {
  it('should display risk and allow AI explanation request', () => {
    cy.visit('/dashboard');
    cy.get('[data-testid="risk-score"]').should('be.visible');
    cy.get('[data-testid="explain-button"]').click();
    cy.get('[data-testid="ai-explanation"]').should('contain', 'rainfall');
  });
});
```

### Performance Testing

**Load Testing (k6)**:
```javascript
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  vus: 1000,  // 1000 virtual users
  duration: '5m',
};

export default function() {
  let response = http.get('https://[BACKEND_URL]/api/v1/risk/current?location=loc_123');
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 5s': (r) => r.timings.duration < 5000,
  });
}
```

### ML Model Testing

**Model Evaluation**:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, test_data):
    predictions = model.predict(test_data)
    metrics = {
        'accuracy': accuracy_score(test_data.labels, predictions),
        'precision': precision_score(test_data.labels, predictions),
        'recall': recall_score(test_data.labels, predictions),
        'f1': f1_score(test_data.labels, predictions)
    }
    assert metrics['accuracy'] >= 0.75, "Model accuracy below threshold"
    return metrics
```

## Security Considerations

### Authentication & Authorization
- Firebase Authentication with secure token validation
- API endpoints protected with middleware checking Firebase tokens
- Role-based access: regular users vs emergency teams
- Session timeout: 1 hour (configurable)

### Data Protection
- HTTPS only for all communications
- API keys stored in Google Secret Manager
- User data encrypted at rest in Firestore
- PII (email, phone) handled according to GDPR/CCPA

### API Security
- Rate limiting: 100 requests/minute per user
- CORS configuration: whitelist frontend domains only
- Input validation and sanitization
- SQL injection prevention (parameterized queries)

### Cloud Security
- Service accounts with least-privilege IAM roles
- VPC for internal service communication
- Cloud Armor for DDoS protection
- Audit logging enabled for all services

## Deployment Strategy

### CI/CD Pipeline

**GitHub Actions Workflow**:
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: npm test
      
  deploy-backend:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy disaster-backend \
            --image gcr.io/$PROJECT_ID/backend:$GITHUB_SHA \
            --platform managed \
            --region us-central1
  
  deploy-frontend:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build and deploy to Firebase Hosting
        run: |
          npm run build
          firebase deploy --only hosting
```

### Environment Configuration

**Development**:
- Cloud Run: 1-5 instances
- BigQuery: sandbox dataset
- Gemini API: rate-limited tier

**Production**:
- Cloud Run: 0-100 instances, autoscaling
- BigQuery: production dataset with backups
- Gemini API: production tier
- CDN enabled for static assets
- Multi-region deployment for high availability

### Monitoring & Observability

**Google Cloud Monitoring**:
- Cloud Run metrics: request latency, error rate, instance count
- BigQuery metrics: query performance, slot usage
- Custom metrics: prediction accuracy, alert delivery rate

**Logging**:
- Structured logging with Cloud Logging
- Log levels: ERROR, WARN, INFO, DEBUG
- Log retention: 30 days

**Alerting**:
- Alert on error rate > 5%
- Alert on API latency > 10 seconds
- Alert on prediction service downtime

## Scalability Considerations

### Horizontal Scaling
- Cloud Run auto-scales based on CPU and request count
- BigQuery automatically scales for query workload
- Firebase Hosting uses global CDN

### Data Partitioning
- BigQuery tables partitioned by date for query performance
- Firestore collections indexed for common queries

### Caching Strategy
- Redis cache for frequently accessed predictions (5-minute TTL)
- Browser caching for static assets (1-day TTL)
- CDN caching for API responses (30-second TTL)

### Cost Optimization
- Cloud Run scales to zero when idle
- BigQuery uses partitioning and clustering to reduce query costs
- Gemini API calls batched when possible
- Weather API calls limited to 30-minute intervals

## Future Enhancements

1. **Mobile Apps**: Native iOS and Android apps
2. **SMS Alerts**: Twilio integration for SMS notifications
3. **Multi-language Support**: i18n for global users
4. **Advanced ML Models**: LSTM for time-series prediction, ensemble models
5. **Satellite Imagery**: Integration with NASA satellite data for visual analysis
6. **Community Reports**: User-submitted disaster reports
7. **Emergency Response Integration**: API for emergency services
8. **Offline Mode**: PWA with offline prediction capability
9. **Voice Alerts**: Integration with smart speakers
10. **Predictive Routing**: Safe route recommendations during disasters
