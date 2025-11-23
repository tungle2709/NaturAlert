"""
Data Loading Module for Disaster Early Warning System

This module provides functions to load and clean weather datasets from CSV files.
It handles data cleaning, unit conversions, and standardization across different datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader class for handling weather datasets
    """
    
    def __init__(self, dataset_dir: str = "dataset"):
        """
        Initialize DataLoader
        
        Args:
            dataset_dir: Path to directory containing CSV datasets
        """
        self.dataset_dir = Path(dataset_dir)
        self.datasets = {}
        
    def load_weather_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available weather datasets from CSV files
        
        Returns:
            Dictionary containing loaded datasets with standardized column names
        """
        logger.info("Loading weather datasets...")
        
        # Define dataset files and their expected structures
        dataset_files = {
            'global': 'GlobalWeatherRepository.csv',
            'classification': 'weather_classification_data.csv', 
            'rain_prediction': 'rain_prediction_2500observations.csv',
            'weather_large': 'weather_data.csv',
            'top_cities': 'top100cities_weather_data.csv',
            'seattle': 'seattle-weather.csv'
        }
        
        loaded_datasets = {}
        
        for name, filename in dataset_files.items():
            file_path = self.dataset_dir / filename
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    logger.info(f"✓ Loaded {name}: {df.shape}")
                    
                    # Clean and standardize the dataset
                    df_cleaned = self._clean_dataset(df, name)
                    loaded_datasets[name] = df_cleaned
                    
                except Exception as e:
                    logger.error(f"✗ Failed to load {name}: {e}")
            else:
                logger.warning(f"⚠ Dataset file not found: {filename}")
        
        self.datasets = loaded_datasets
        logger.info(f"Successfully loaded {len(loaded_datasets)} datasets")
        return loaded_datasets
    
    def _clean_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Clean and standardize a dataset
        
        Args:
            df: Raw dataframe
            dataset_name: Name of the dataset for specific cleaning rules
            
        Returns:
            Cleaned dataframe with standardized columns
        """
        logger.info(f"Cleaning dataset: {dataset_name}")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Standardize column names (lowercase, replace spaces with underscores)
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('%', 'pct')
        
        # Dataset-specific cleaning
        if dataset_name == 'global':
            df_clean = self._clean_global_dataset(df_clean)
        elif dataset_name == 'classification':
            df_clean = self._clean_classification_dataset(df_clean)
        elif dataset_name == 'rain_prediction':
            df_clean = self._clean_rain_prediction_dataset(df_clean)
        
        # General cleaning steps
        df_clean = self._apply_general_cleaning(df_clean)
        
        logger.info(f"Cleaned {dataset_name}: {df_clean.shape}")
        return df_clean
    
    def _clean_global_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean GlobalWeatherRepository dataset"""
        # Standardize temperature to Celsius
        if 'temperature_celsius' in df.columns:
            df['temperature'] = df['temperature_celsius']
        elif 'temperature_fahrenheit' in df.columns:
            df['temperature'] = self._fahrenheit_to_celsius(df['temperature_fahrenheit'])
        
        # Standardize pressure to hPa (mb)
        if 'pressure_mb' in df.columns:
            df['pressure'] = df['pressure_mb']
        elif 'pressure_in' in df.columns:
            df['pressure'] = self._inches_to_hpa(df['pressure_in'])
        
        # Standardize wind speed to mph
        if 'wind_mph' in df.columns:
            df['wind_speed'] = df['wind_mph']
        elif 'wind_kph' in df.columns:
            df['wind_speed'] = self._kph_to_mph(df['wind_kph'])
        
        # Standardize precipitation to mm
        if 'precip_mm' in df.columns:
            df['precipitation'] = df['precip_mm']
        elif 'precip_in' in df.columns:
            df['precipitation'] = self._inches_to_mm(df['precip_in'])
        
        # Add other standard columns
        if 'humidity' not in df.columns and 'humidity' in df.columns:
            df['humidity'] = df['humidity']
        
        return df
    
    def _clean_classification_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean weather classification dataset"""
        # Standardize column names
        column_mapping = {
            'precipitation_pct': 'precipitation',
            'wind_speed': 'wind_speed',
            'atmospheric_pressure': 'pressure',
            'weather_type': 'weather_type'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        return df
    
    def _clean_rain_prediction_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean rain prediction dataset"""
        # Standardize column names
        if 'wind_speed' in df.columns:
            df['wind_speed'] = df['wind_speed']
        if 'cloud_cover' in df.columns:
            df['cloud_cover'] = df['cloud_cover']
        
        # Convert rain column to binary
        if 'rain' in df.columns:
            df['has_rain'] = (df['rain'] == 'rain').astype(int)
        
        return df
    
    def _apply_general_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply general cleaning steps to any dataset"""
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Validate and clean numeric columns
        df = self._validate_numeric_columns(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        
        # Get numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # For numeric columns, use forward fill then backward fill
        for col in numeric_cols:
            if df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                logger.info(f"Filled {missing_count} missing values in {col}")
        
        # For categorical columns, use mode (most frequent value)
        for col in categorical_cols:
            if df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown'
                df[col] = df[col].fillna(mode_value)
                logger.info(f"Filled {missing_count} missing values in {col} with '{mode_value}'")
        
        return df
    
    def _validate_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean numeric columns"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Remove infinite values
            if np.isinf(df[col]).any():
                inf_count = np.isinf(df[col]).sum()
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(df[col].median())
                logger.info(f"Replaced {inf_count} infinite values in {col}")
            
            # Handle extreme outliers (beyond 5 standard deviations)
            if col in ['temperature', 'pressure', 'wind_speed', 'humidity']:
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                # Define reasonable bounds for weather variables
                bounds = {
                    'temperature': (-50, 60),  # Celsius
                    'pressure': (800, 1100),   # hPa
                    'wind_speed': (0, 200),    # mph
                    'humidity': (0, 100)       # percentage
                }
                
                if col in bounds:
                    min_bound, max_bound = bounds[col]
                    outliers = (df[col] < min_bound) | (df[col] > max_bound)
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        # Cap outliers at reasonable bounds
                        df.loc[df[col] < min_bound, col] = min_bound
                        df.loc[df[col] > max_bound, col] = max_bound
                        logger.info(f"Capped {outlier_count} outliers in {col}")
        
        return df
    
    # Unit conversion functions
    @staticmethod
    def _fahrenheit_to_celsius(fahrenheit: pd.Series) -> pd.Series:
        """Convert Fahrenheit to Celsius"""
        return (fahrenheit - 32) * 5/9
    
    @staticmethod
    def _celsius_to_fahrenheit(celsius: pd.Series) -> pd.Series:
        """Convert Celsius to Fahrenheit"""
        return celsius * 9/5 + 32
    
    @staticmethod
    def _kph_to_mph(kph: pd.Series) -> pd.Series:
        """Convert km/h to mph"""
        return kph * 0.621371
    
    @staticmethod
    def _mph_to_kph(mph: pd.Series) -> pd.Series:
        """Convert mph to km/h"""
        return mph / 0.621371
    
    @staticmethod
    def _inches_to_hpa(inches: pd.Series) -> pd.Series:
        """Convert inches of mercury to hPa"""
        return inches * 33.8639
    
    @staticmethod
    def _hpa_to_inches(hpa: pd.Series) -> pd.Series:
        """Convert hPa to inches of mercury"""
        return hpa / 33.8639
    
    @staticmethod
    def _inches_to_mm(inches: pd.Series) -> pd.Series:
        """Convert inches to millimeters"""
        return inches * 25.4
    
    @staticmethod
    def _mm_to_inches(mm: pd.Series) -> pd.Series:
        """Convert millimeters to inches"""
        return mm / 25.4
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """
        Get information about loaded datasets
        
        Returns:
            Dictionary with dataset information
        """
        info = {}
        
        for name, df in self.datasets.items():
            info[name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object']).columns),
                'missing_values': df.isnull().sum().sum(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            }
        
        return info
    
    def get_combined_dataset(self, datasets_to_combine: List[str] = None) -> pd.DataFrame:
        """
        Combine multiple datasets into a single dataframe
        
        Args:
            datasets_to_combine: List of dataset names to combine. If None, combines all.
            
        Returns:
            Combined dataframe with source column indicating origin
        """
        if datasets_to_combine is None:
            datasets_to_combine = list(self.datasets.keys())
        
        combined_dfs = []
        
        for name in datasets_to_combine:
            if name in self.datasets:
                df = self.datasets[name].copy()
                df['source_dataset'] = name
                combined_dfs.append(df)
        
        if combined_dfs:
            # Find common columns
            common_cols = set(combined_dfs[0].columns)
            for df in combined_dfs[1:]:
                common_cols = common_cols.intersection(set(df.columns))
            
            # Keep only common columns for each dataframe
            aligned_dfs = []
            for df in combined_dfs:
                aligned_dfs.append(df[list(common_cols)])
            
            combined_df = pd.concat(aligned_dfs, ignore_index=True)
            logger.info(f"Combined {len(datasets_to_combine)} datasets into shape {combined_df.shape}")
            return combined_df
        
        return pd.DataFrame()


def load_weather_datasets(dataset_dir: str = "dataset") -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load all weather datasets
    
    Args:
        dataset_dir: Path to directory containing CSV datasets
        
    Returns:
        Dictionary containing loaded and cleaned datasets
    """
    loader = DataLoader(dataset_dir)
    return loader.load_weather_datasets()


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    datasets = loader.load_weather_datasets()
    
    print("\nDataset Information:")
    print("=" * 50)
    info = loader.get_dataset_info()
    for name, details in info.items():
        print(f"\n{name.upper()}:")
        print(f"  Shape: {details['shape']}")
        print(f"  Columns: {len(details['columns'])}")
        print(f"  Missing values: {details['missing_values']}")
        print(f"  Memory: {details['memory_usage_mb']:.2f} MB")