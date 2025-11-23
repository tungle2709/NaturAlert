"""
Feature Engineering Module for Disaster Early Warning System

This module creates engineered features from weather data for ML training.
It calculates rolling window statistics and derived features that help predict disasters.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Class for engineering features from weather data for disaster prediction
    """
    
    def __init__(self, window_size: int = 7):
        """
        Initialize FeatureEngineer
        
        Args:
            window_size: Size of rolling window for calculations (default 7 days)
        """
        self.window_size = window_size
        self.feature_definitions = {
            'pressure_drop_7d': 'Maximum pressure drop over 7-day window (hPa)',
            'wind_spike_max': 'Maximum wind speed spike in 7-day window (mph)',
            'rain_accumulation_7d': 'Total rainfall accumulation over 7 days (mm)',
            'humidity_trend': 'Linear trend of humidity over 7 days (slope)',
            'temp_deviation': 'Temperature deviation from 7-day average (°C)',
            'pressure_velocity': 'Rate of pressure change over 7 days (hPa/day)',
            'wind_gust_ratio': 'Ratio of max to average wind speed over 7 days',
            'temp_range_7d': 'Temperature range (max - min) over 7 days (°C)',
            'pressure_stability': 'Pressure stability index (inverse of std dev)',
            'weather_volatility': 'Combined volatility index of all weather variables'
        }
    
    def engineer_features(self, weather_df: pd.DataFrame, 
                         datetime_col: Optional[str] = None,
                         location_col: Optional[str] = None) -> pd.DataFrame:
        """
        Engineer features from weather data
        
        Args:
            weather_df: DataFrame with weather data
            datetime_col: Name of datetime column (if available)
            location_col: Name of location column (if available)
            
        Returns:
            DataFrame with engineered features added
        """
        logger.info(f"Engineering features for {len(weather_df)} weather records")
        
        # Make a copy to avoid modifying original data
        df = weather_df.copy()
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Sort by datetime if available
        if datetime_col and datetime_col in df.columns:
            df = df.sort_values(datetime_col)
        
        # Group by location if available
        if location_col and location_col in df.columns:
            logger.info(f"Processing features by location: {df[location_col].nunique()} unique locations")
            df = df.groupby(location_col).apply(self._calculate_features_for_group).reset_index(drop=True)
        else:
            df = self._calculate_features_for_group(df)
        
        # Calculate composite features
        df = self._calculate_composite_features(df)
        
        # Handle edge cases and missing values
        df = self._handle_feature_edge_cases(df)
        
        logger.info(f"Feature engineering complete. Added {len(self.feature_definitions)} features")
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize weather column names"""
        
        # Create mapping for common column name variations
        column_mapping = {}
        
        # Temperature
        temp_cols = [col for col in df.columns if any(x in col.lower() for x in ['temp', 'celsius', 'fahrenheit'])]
        if temp_cols and 'temperature' not in df.columns:
            column_mapping[temp_cols[0]] = 'temperature'
        
        # Pressure
        pressure_cols = [col for col in df.columns if 'pressure' in col.lower()]
        if pressure_cols and 'pressure' not in df.columns:
            column_mapping[pressure_cols[0]] = 'pressure'
        
        # Wind speed
        wind_cols = [col for col in df.columns if any(x in col.lower() for x in ['wind_speed', 'wind'])]
        if wind_cols and 'wind_speed' not in df.columns:
            column_mapping[wind_cols[0]] = 'wind_speed'
        
        # Precipitation
        precip_cols = [col for col in df.columns if any(x in col.lower() for x in ['precip', 'rain', 'rainfall'])]
        if precip_cols and 'precipitation' not in df.columns:
            column_mapping[precip_cols[0]] = 'precipitation'
        
        # Humidity
        humidity_cols = [col for col in df.columns if 'humidity' in col.lower()]
        if humidity_cols and 'humidity' not in df.columns:
            column_mapping[humidity_cols[0]] = 'humidity'
        
        # Apply mapping
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        return df
    
    def _calculate_features_for_group(self, group_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling window features for a group (location or entire dataset)"""
        
        df = group_df.copy()
        
        # Ensure we have the required columns
        required_cols = ['temperature', 'pressure', 'wind_speed', 'precipitation', 'humidity']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < 3:
            logger.warning(f"Insufficient weather columns for feature engineering. Available: {available_cols}")
            return df
        
        # Calculate rolling window features
        if 'pressure' in df.columns:
            df = self._calculate_pressure_features(df)
        
        if 'wind_speed' in df.columns:
            df = self._calculate_wind_features(df)
        
        if 'precipitation' in df.columns:
            df = self._calculate_precipitation_features(df)
        
        if 'humidity' in df.columns:
            df = self._calculate_humidity_features(df)
        
        if 'temperature' in df.columns:
            df = self._calculate_temperature_features(df)
        
        return df
    
    def _calculate_pressure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pressure-related features"""
        
        # Pressure drop over window
        pressure_rolling = df['pressure'].rolling(window=self.window_size, min_periods=1)
        df['pressure_drop_7d'] = pressure_rolling.max() - pressure_rolling.min()
        
        # Pressure velocity (rate of change)
        df['pressure_velocity'] = np.abs(df['pressure'].diff()).rolling(
            window=self.window_size, min_periods=1
        ).mean()
        
        # Pressure stability (inverse of standard deviation)
        pressure_std = pressure_rolling.std()
        df['pressure_stability'] = 1 / (pressure_std + 0.1)  # Add small constant to avoid division by zero
        
        return df
    
    def _calculate_wind_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate wind-related features"""
        
        # Wind spike maximum
        df['wind_spike_max'] = df['wind_speed'].rolling(
            window=self.window_size, min_periods=1
        ).max()
        
        # Wind gust ratio (max to average)
        wind_rolling = df['wind_speed'].rolling(window=self.window_size, min_periods=1)
        wind_avg = wind_rolling.mean()
        wind_max = wind_rolling.max()
        df['wind_gust_ratio'] = wind_max / (wind_avg + 0.1)  # Avoid division by zero
        
        # Wind acceleration (change in wind speed)
        df['wind_acceleration'] = df['wind_speed'].diff().rolling(
            window=self.window_size, min_periods=1
        ).mean()
        
        return df
    
    def _calculate_precipitation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate precipitation-related features"""
        
        # Rain accumulation over window
        df['rain_accumulation_7d'] = df['precipitation'].rolling(
            window=self.window_size, min_periods=1
        ).sum()
        
        # Rain intensity (average daily rainfall)
        df['rain_intensity'] = df['precipitation'].rolling(
            window=self.window_size, min_periods=1
        ).mean()
        
        # Rain variability
        df['rain_variability'] = df['precipitation'].rolling(
            window=self.window_size, min_periods=1
        ).std()
        
        return df
    
    def _calculate_humidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate humidity-related features"""
        
        # Humidity trend (linear regression slope)
        df['humidity_trend'] = df['humidity'].rolling(
            window=self.window_size, min_periods=2
        ).apply(self._calculate_trend_slope, raw=False)
        
        # Humidity range
        humidity_rolling = df['humidity'].rolling(window=self.window_size, min_periods=1)
        df['humidity_range'] = humidity_rolling.max() - humidity_rolling.min()
        
        return df
    
    def _calculate_temperature_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate temperature-related features"""
        
        # Temperature deviation from rolling average
        temp_rolling_mean = df['temperature'].rolling(
            window=self.window_size, min_periods=1
        ).mean()
        df['temp_deviation'] = np.abs(df['temperature'] - temp_rolling_mean)
        
        # Temperature range over window
        temp_rolling = df['temperature'].rolling(window=self.window_size, min_periods=1)
        df['temp_range_7d'] = temp_rolling.max() - temp_rolling.min()
        
        # Temperature volatility
        df['temp_volatility'] = df['temperature'].rolling(
            window=self.window_size, min_periods=1
        ).std()
        
        return df
    
    def _calculate_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite features combining multiple weather variables"""
        
        # Weather volatility index (normalized combination of all volatilities)
        volatility_features = []
        
        if 'pressure_velocity' in df.columns:
            volatility_features.append('pressure_velocity')
        if 'wind_acceleration' in df.columns:
            volatility_features.append('wind_acceleration')
        if 'temp_volatility' in df.columns:
            volatility_features.append('temp_volatility')
        if 'rain_variability' in df.columns:
            volatility_features.append('rain_variability')
        
        if volatility_features:
            # Normalize each volatility feature and combine
            volatility_normalized = pd.DataFrame()
            for feature in volatility_features:
                if df[feature].std() > 0:
                    volatility_normalized[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
                else:
                    volatility_normalized[feature] = 0
            
            df['weather_volatility'] = volatility_normalized.mean(axis=1)
        
        # Extreme weather index (combination of extreme conditions)
        extreme_conditions = []
        
        if 'pressure_drop_7d' in df.columns:
            extreme_conditions.append(df['pressure_drop_7d'] > df['pressure_drop_7d'].quantile(0.9))
        if 'wind_spike_max' in df.columns:
            extreme_conditions.append(df['wind_spike_max'] > df['wind_spike_max'].quantile(0.9))
        if 'rain_accumulation_7d' in df.columns:
            extreme_conditions.append(df['rain_accumulation_7d'] > df['rain_accumulation_7d'].quantile(0.9))
        
        if extreme_conditions:
            df['extreme_weather_index'] = sum(extreme_conditions) / len(extreme_conditions)
        
        # Atmospheric instability index
        if all(col in df.columns for col in ['pressure_drop_7d', 'wind_gust_ratio', 'temp_range_7d']):
            # Normalize and combine pressure drop, wind variability, and temperature range
            pressure_norm = (df['pressure_drop_7d'] - df['pressure_drop_7d'].min()) / (df['pressure_drop_7d'].max() - df['pressure_drop_7d'].min() + 0.1)
            wind_norm = (df['wind_gust_ratio'] - df['wind_gust_ratio'].min()) / (df['wind_gust_ratio'].max() - df['wind_gust_ratio'].min() + 0.1)
            temp_norm = (df['temp_range_7d'] - df['temp_range_7d'].min()) / (df['temp_range_7d'].max() - df['temp_range_7d'].min() + 0.1)
            
            df['atmospheric_instability'] = (pressure_norm + wind_norm + temp_norm) / 3
        
        return df
    
    def _calculate_trend_slope(self, series: pd.Series) -> float:
        """Calculate linear regression slope for trend analysis"""
        
        if len(series) < 2:
            return 0.0
        
        # Remove NaN values
        clean_series = series.dropna()
        if len(clean_series) < 2:
            return 0.0
        
        # Calculate linear regression slope
        x = np.arange(len(clean_series))
        y = clean_series.values
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        except (np.linalg.LinAlgError, ValueError):
            return 0.0
    
    def _handle_feature_edge_cases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle edge cases and missing values in engineered features"""
        
        # List of engineered features
        engineered_features = [
            'pressure_drop_7d', 'wind_spike_max', 'rain_accumulation_7d',
            'humidity_trend', 'temp_deviation', 'pressure_velocity',
            'wind_gust_ratio', 'temp_range_7d', 'pressure_stability',
            'weather_volatility', 'extreme_weather_index', 'atmospheric_instability'
        ]
        
        for feature in engineered_features:
            if feature in df.columns:
                # Handle infinite values
                df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
                
                # Fill NaN values with median
                if df[feature].isnull().any():
                    median_value = df[feature].median()
                    if pd.isna(median_value):
                        median_value = 0.0
                    df[feature] = df[feature].fillna(median_value)
                
                # Cap extreme outliers (beyond 5 standard deviations)
                if df[feature].std() > 0:
                    mean_val = df[feature].mean()
                    std_val = df[feature].std()
                    lower_bound = mean_val - 5 * std_val
                    upper_bound = mean_val + 5 * std_val
                    
                    df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def get_feature_importance_ranking(self, df: pd.DataFrame, 
                                     target_col: str = 'disaster_occurred') -> pd.DataFrame:
        """
        Calculate feature importance ranking using correlation with target
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            DataFrame with feature importance rankings
        """
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found")
            return pd.DataFrame()
        
        # Get engineered features
        engineered_features = [col for col in df.columns if col in self.feature_definitions.keys()]
        
        if not engineered_features:
            logger.warning("No engineered features found")
            return pd.DataFrame()
        
        # Calculate correlations with target
        correlations = []
        for feature in engineered_features:
            if df[feature].dtype in ['int64', 'float64']:
                corr = df[feature].corr(df[target_col])
                correlations.append({
                    'feature': feature,
                    'correlation': abs(corr) if not pd.isna(corr) else 0,
                    'description': self.feature_definitions.get(feature, '')
                })
        
        # Sort by correlation strength
        importance_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
        
        logger.info(f"Feature importance calculated for {len(importance_df)} features")
        return importance_df
    
    def create_feature_summary(self, df: pd.DataFrame) -> Dict:
        """
        Create summary statistics for engineered features
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Dictionary with feature summary statistics
        """
        engineered_features = [col for col in df.columns if col in self.feature_definitions.keys()]
        
        summary = {
            'total_features': len(engineered_features),
            'feature_statistics': {},
            'feature_descriptions': self.feature_definitions
        }
        
        for feature in engineered_features:
            if feature in df.columns and df[feature].dtype in ['int64', 'float64']:
                summary['feature_statistics'][feature] = {
                    'mean': float(df[feature].mean()),
                    'std': float(df[feature].std()),
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max()),
                    'median': float(df[feature].median()),
                    'missing_count': int(df[feature].isnull().sum())
                }
        
        return summary


def engineer_features(weather_df: pd.DataFrame, 
                     window_size: int = 7,
                     datetime_col: Optional[str] = None,
                     location_col: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to engineer features from weather data
    
    Args:
        weather_df: DataFrame with weather data
        window_size: Size of rolling window for calculations
        datetime_col: Name of datetime column (if available)
        location_col: Name of location column (if available)
        
    Returns:
        DataFrame with engineered features added
    """
    engineer = FeatureEngineer(window_size)
    return engineer.engineer_features(weather_df, datetime_col, location_col)


if __name__ == "__main__":
    # Example usage and testing
    
    # Create sample weather data
    np.random.seed(42)
    n_samples = 100
    
    sample_data = pd.DataFrame({
        'temperature': np.random.normal(20, 10, n_samples),
        'pressure': np.random.normal(1013, 20, n_samples),
        'wind_speed': np.random.exponential(15, n_samples),
        'precipitation': np.random.exponential(5, n_samples),
        'humidity': np.random.normal(70, 15, n_samples).clip(0, 100),
        'datetime': pd.date_range('2023-01-01', periods=n_samples, freq='D')
    })
    
    # Engineer features
    engineer = FeatureEngineer()
    featured_data = engineer.engineer_features(sample_data, datetime_col='datetime')
    
    print("Feature Engineering Test Results:")
    print("=" * 50)
    print(f"Original columns: {len(sample_data.columns)}")
    print(f"After feature engineering: {len(featured_data.columns)}")
    
    # Show engineered features
    engineered_cols = [col for col in featured_data.columns if col in engineer.feature_definitions.keys()]
    print(f"\nEngineered features ({len(engineered_cols)}):")
    for col in engineered_cols:
        if col in featured_data.columns:
            print(f"  {col}: {featured_data[col].mean():.2f} ± {featured_data[col].std():.2f}")
    
    # Create feature summary
    summary = engineer.create_feature_summary(featured_data)
    print(f"\nFeature Summary:")
    print(f"Total engineered features: {summary['total_features']}")
    print(f"Features with statistics: {len(summary['feature_statistics'])}")