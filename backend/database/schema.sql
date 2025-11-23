-- Disaster Early Warning System Database Schema
-- SQLite Database

-- Table: disasters_historical
-- Stores historical disaster event data
CREATE TABLE IF NOT EXISTS disasters_historical (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disaster_id TEXT UNIQUE NOT NULL,
    disaster_type TEXT NOT NULL CHECK(disaster_type IN ('flood', 'storm', 'hurricane', 'extreme_rainfall')),
    event_date DATE NOT NULL,
    location_id TEXT,
    country TEXT,
    region TEXT,
    latitude REAL,
    longitude REAL,
    severity TEXT CHECK(severity IN ('low', 'moderate', 'high', 'extreme')),
    impact_description TEXT,
    casualties INTEGER DEFAULT 0,
    economic_damage REAL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_disasters_date ON disasters_historical(event_date);
CREATE INDEX IF NOT EXISTS idx_disasters_type ON disasters_historical(disaster_type);
CREATE INDEX IF NOT EXISTS idx_disasters_location ON disasters_historical(location_id);

-- Table: weather_historical
-- Stores historical weather observation data
CREATE TABLE IF NOT EXISTS weather_historical (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    location_id TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    temperature REAL,           -- Celsius
    pressure REAL,              -- hPa
    humidity REAL,              -- percentage (0-100)
    wind_speed REAL,            -- mph
    wind_direction REAL,        -- degrees (0-360)
    rainfall_1h REAL,           -- mm
    rainfall_24h REAL,          -- mm
    cloud_cover REAL,           -- percentage (0-100)
    visibility REAL,            -- km
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_weather_hist_location ON weather_historical(location_id);
CREATE INDEX IF NOT EXISTS idx_weather_hist_timestamp ON weather_historical(timestamp);

-- Table: features_training
-- Stores engineered features for ML training
CREATE TABLE IF NOT EXISTS features_training (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_id TEXT UNIQUE NOT NULL,
    disaster_id TEXT,
    location_id TEXT NOT NULL,
    disaster_date DATE,
    window_start_date DATE,         -- 7 days before disaster
    pressure_drop_7d REAL,          -- Maximum pressure drop over 7 days (hPa)
    wind_spike_max REAL,            -- Maximum wind speed spike (mph)
    rain_accumulation_7d REAL,      -- Total rainfall over 7 days (mm)
    humidity_trend REAL,            -- Linear regression slope of humidity
    temp_deviation REAL,            -- Standard deviation from historical average (°C)
    pressure_velocity REAL,         -- Rate of pressure change (hPa/hour)
    wind_gust_ratio REAL,           -- Max gust / average wind speed
    disaster_occurred INTEGER NOT NULL CHECK(disaster_occurred IN (0, 1)),  -- Boolean: 0 or 1
    disaster_type TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (disaster_id) REFERENCES disasters_historical(disaster_id)
);

CREATE INDEX IF NOT EXISTS idx_features_disaster ON features_training(disaster_id);
CREATE INDEX IF NOT EXISTS idx_features_location ON features_training(location_id);

-- Table: predictions_log
-- Stores ML prediction results
CREATE TABLE IF NOT EXISTS predictions_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT UNIQUE NOT NULL,
    timestamp DATETIME NOT NULL,
    location_id TEXT NOT NULL,
    risk_score REAL NOT NULL,                   -- 0-100%
    confidence_interval_lower REAL,
    confidence_interval_upper REAL,
    predicted_disaster_type TEXT,
    feature_values TEXT,                        -- JSON string of input features
    model_version TEXT,
    ai_explanation TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_location ON predictions_log(location_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions_log(timestamp);

-- Table: users
-- Stores user account information
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT,
    display_name TEXT,
    home_location_id TEXT,
    home_latitude REAL,
    home_longitude REAL,
    home_address TEXT,
    alert_threshold REAL DEFAULT 70.0 CHECK(alert_threshold BETWEEN 50 AND 90),
    notification_channels TEXT,                 -- JSON array: ["email", "push"]
    theme TEXT DEFAULT 'light' CHECK(theme IN ('light', 'dark')),
    language TEXT DEFAULT 'en',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Table: alerts_history
-- Stores alert notification history
CREATE TABLE IF NOT EXISTS alerts_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT UNIQUE NOT NULL,
    user_id TEXT NOT NULL,
    prediction_id TEXT,
    timestamp DATETIME NOT NULL,
    risk_score REAL NOT NULL,
    disaster_type TEXT,
    alert_message TEXT,
    channels_sent TEXT,                         -- JSON array of channels used
    acknowledged BOOLEAN DEFAULT 0,
    acknowledged_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (prediction_id) REFERENCES predictions_log(prediction_id)
);

CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts_history(user_id);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts_history(timestamp);
