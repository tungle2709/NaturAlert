"""
Disaster Labeling Module for Disaster Early Warning System

This module creates synthetic disaster labels from extreme weather patterns.
It defines criteria for different disaster types based on meteorological thresholds.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DisasterLabeler:
    """
    Class for creating disaster labels from weather data based on extreme conditions
    """
    
    def __init__(self):
        """Initialize DisasterLabeler with default thresholds"""
        
        # Define disaster criteria thresholds
        self.disaster_thresholds = {
            'flood': {
                'precipitation_mm': 100,      # Heavy rainfall threshold
                'pressure_hpa': 1000,         # Low pressure threshold
                'description': 'High precipitation with low pressure'
            },
            'storm': {
                'wind_speed_mph': 40,         # High wind threshold
                'pressure_hpa': 1005,         # Low pressure threshold
                'description': 'High wind speed with low pressure'
            },
            'extreme_rainfall': {
                'precipitation_mm': 150,      # Very heavy rainfall
                'description': 'Extreme precipitation levels'
            },
            'hurricane': {
                'wind_speed_mph': 74,         # Hurricane force winds
                'pressure_hpa': 980,          # Very low pressure
                'description': 'Hurricane-force winds with very low pressure'
            }
        }
        
        # Regional adjustment factors (can be customized based on location)
        self.regional_factors = {
            'tropical': {
                'precipitation_multiplier': 1.2,  # Higher rainfall tolerance
                'wind_multiplier': 0.9,           # Lower wind tolerance
                'pressure_adjustment': -5         # Adjust pressure thresholds
            },
            'temperate': {
                'precipitation_multiplier': 1.0,
                'wind_multiplier': 1.0,
                'pressure_adjustment': 0
            },
            'arid': {
                'precipitation_multiplier': 0.7,  # Lower rainfall for disaster
                'wind_multiplier': 1.1,           # Higher wind tolerance
                'pressure_adjustment': 5
            }
        }
    
    def create_disaster_labels(self, weather_df: pd.DataFrame, 
                             region_type: str = 'temperate',
                             custom_thresholds: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create disaster labels based on extreme weather conditions
        
        Args:
            weather_df: DataFrame with weather data
            region_type: Type of region ('tropical', 'temperate', 'arid')
            custom_thresholds: Optional custom threshold dictionary
            
        Returns:
            DataFrame with added disaster columns
        """
        logger.info(f"Creating disaster labels for {len(weather_df)} weather records")
        
        # Make a copy to avoid modifying original data
        df = weather_df.copy()
        
        # Initialize disaster columns
        df['disaster_occurred'] = 0
        df['disaster_type'] = 'none'
        df['disaster_severity'] = 'none'
        df['disaster_confidence'] = 0.0
        
        # Get adjusted thresholds for the region
        thresholds = self._get_adjusted_thresholds(region_type, custom_thresholds)
        
        # Standardize column names for processing
        df = self._standardize_weather_columns(df)
        
        # Apply disaster labeling rules
        df = self._apply_hurricane_labels(df, thresholds)
        df = self._apply_flood_labels(df, thresholds)
        df = self._apply_storm_labels(df, thresholds)
        df = self._apply_extreme_rainfall_labels(df, thresholds)
        
        # Calculate disaster statistics
        disaster_stats = self._calculate_disaster_statistics(df)
        logger.info(f"Disaster labeling complete: {disaster_stats}")
        
        return df
    
    def _standardize_weather_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize weather column names for consistent processing"""
        
        # Create mapping for common column name variations
        column_mapping = {}
        
        # Temperature columns
        temp_variations = ['temperature', 'temp', 'temperature_celsius', 'temp_c']
        for col in df.columns:
            if any(var in col.lower() for var in temp_variations):
                column_mapping[col] = 'temperature'
                break
        
        # Pressure columns
        pressure_variations = ['pressure', 'atmospheric_pressure', 'pressure_mb', 'pressure_hpa']
        for col in df.columns:
            if any(var in col.lower() for var in pressure_variations):
                column_mapping[col] = 'pressure'
                break
        
        # Wind speed columns
        wind_variations = ['wind_speed', 'wind', 'wind_mph', 'wind_kph']
        for col in df.columns:
            if any(var in col.lower() for var in wind_variations):
                column_mapping[col] = 'wind_speed'
                break
        
        # Precipitation columns
        precip_variations = ['precipitation', 'precip', 'rainfall', 'rain', 'precip_mm']
        for col in df.columns:
            if any(var in col.lower() for var in precip_variations):
                column_mapping[col] = 'precipitation'
                break
        
        # Humidity columns
        humidity_variations = ['humidity', 'relative_humidity', 'rh']
        for col in df.columns:
            if any(var in col.lower() for var in humidity_variations):
                column_mapping[col] = 'humidity'
                break
        
        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        return df
    
    def _get_adjusted_thresholds(self, region_type: str, 
                               custom_thresholds: Optional[Dict] = None) -> Dict:
        """Get disaster thresholds adjusted for region type"""
        
        if custom_thresholds:
            return custom_thresholds
        
        thresholds = self.disaster_thresholds.copy()
        
        if region_type in self.regional_factors:
            factors = self.regional_factors[region_type]
            
            # Apply regional adjustments
            for disaster_type in thresholds:
                if 'precipitation_mm' in thresholds[disaster_type]:
                    thresholds[disaster_type]['precipitation_mm'] *= factors['precipitation_multiplier']
                
                if 'wind_speed_mph' in thresholds[disaster_type]:
                    thresholds[disaster_type]['wind_speed_mph'] *= factors['wind_multiplier']
                
                if 'pressure_hpa' in thresholds[disaster_type]:
                    thresholds[disaster_type]['pressure_hpa'] += factors['pressure_adjustment']
        
        logger.info(f"Using {region_type} region thresholds")
        return thresholds
    
    def _apply_hurricane_labels(self, df: pd.DataFrame, thresholds: Dict) -> pd.DataFrame:
        """Apply hurricane disaster labels (highest priority)"""
        
        if 'wind_speed' not in df.columns or 'pressure' not in df.columns:
            logger.warning("Missing wind_speed or pressure columns for hurricane detection")
            return df
        
        hurricane_threshold = thresholds['hurricane']
        
        # Hurricane conditions: Very high wind + very low pressure
        hurricane_mask = (
            (df['wind_speed'] >= hurricane_threshold['wind_speed_mph']) &
            (df['pressure'] <= hurricane_threshold['pressure_hpa'])
        )
        
        # Apply hurricane labels (overrides other disaster types)
        df.loc[hurricane_mask, 'disaster_occurred'] = 1
        df.loc[hurricane_mask, 'disaster_type'] = 'hurricane'
        df.loc[hurricane_mask, 'disaster_severity'] = self._calculate_hurricane_severity(
            df.loc[hurricane_mask], hurricane_threshold
        )
        df.loc[hurricane_mask, 'disaster_confidence'] = self._calculate_confidence(
            df.loc[hurricane_mask], 'hurricane', hurricane_threshold
        )
        
        hurricane_count = hurricane_mask.sum()
        if hurricane_count > 0:
            logger.info(f"Identified {hurricane_count} hurricane events")
        
        return df
    
    def _apply_flood_labels(self, df: pd.DataFrame, thresholds: Dict) -> pd.DataFrame:
        """Apply flood disaster labels"""
        
        if 'precipitation' not in df.columns or 'pressure' not in df.columns:
            logger.warning("Missing precipitation or pressure columns for flood detection")
            return df
        
        flood_threshold = thresholds['flood']
        
        # Flood conditions: High precipitation + low pressure
        # Only apply if not already labeled as hurricane
        flood_mask = (
            (df['precipitation'] >= flood_threshold['precipitation_mm']) &
            (df['pressure'] <= flood_threshold['pressure_hpa']) &
            (df['disaster_type'] == 'none')
        )
        
        df.loc[flood_mask, 'disaster_occurred'] = 1
        df.loc[flood_mask, 'disaster_type'] = 'flood'
        df.loc[flood_mask, 'disaster_severity'] = self._calculate_flood_severity(
            df.loc[flood_mask], flood_threshold
        )
        df.loc[flood_mask, 'disaster_confidence'] = self._calculate_confidence(
            df.loc[flood_mask], 'flood', flood_threshold
        )
        
        flood_count = flood_mask.sum()
        if flood_count > 0:
            logger.info(f"Identified {flood_count} flood events")
        
        return df
    
    def _apply_storm_labels(self, df: pd.DataFrame, thresholds: Dict) -> pd.DataFrame:
        """Apply storm disaster labels"""
        
        if 'wind_speed' not in df.columns or 'pressure' not in df.columns:
            logger.warning("Missing wind_speed or pressure columns for storm detection")
            return df
        
        storm_threshold = thresholds['storm']
        
        # Storm conditions: High wind + low pressure (but not hurricane level)
        # Only apply if not already labeled as hurricane or flood
        storm_mask = (
            (df['wind_speed'] >= storm_threshold['wind_speed_mph']) &
            (df['pressure'] <= storm_threshold['pressure_hpa']) &
            (df['disaster_type'] == 'none')
        )
        
        df.loc[storm_mask, 'disaster_occurred'] = 1
        df.loc[storm_mask, 'disaster_type'] = 'storm'
        df.loc[storm_mask, 'disaster_severity'] = self._calculate_storm_severity(
            df.loc[storm_mask], storm_threshold
        )
        df.loc[storm_mask, 'disaster_confidence'] = self._calculate_confidence(
            df.loc[storm_mask], 'storm', storm_threshold
        )
        
        storm_count = storm_mask.sum()
        if storm_count > 0:
            logger.info(f"Identified {storm_count} storm events")
        
        return df
    
    def _apply_extreme_rainfall_labels(self, df: pd.DataFrame, thresholds: Dict) -> pd.DataFrame:
        """Apply extreme rainfall disaster labels"""
        
        if 'precipitation' not in df.columns:
            logger.warning("Missing precipitation column for extreme rainfall detection")
            return df
        
        rainfall_threshold = thresholds['extreme_rainfall']
        
        # Extreme rainfall conditions: Very high precipitation
        # Only apply if not already labeled with other disaster types
        rainfall_mask = (
            (df['precipitation'] >= rainfall_threshold['precipitation_mm']) &
            (df['disaster_type'] == 'none')
        )
        
        df.loc[rainfall_mask, 'disaster_occurred'] = 1
        df.loc[rainfall_mask, 'disaster_type'] = 'extreme_rainfall'
        df.loc[rainfall_mask, 'disaster_severity'] = self._calculate_rainfall_severity(
            df.loc[rainfall_mask], rainfall_threshold
        )
        df.loc[rainfall_mask, 'disaster_confidence'] = self._calculate_confidence(
            df.loc[rainfall_mask], 'extreme_rainfall', rainfall_threshold
        )
        
        rainfall_count = rainfall_mask.sum()
        if rainfall_count > 0:
            logger.info(f"Identified {rainfall_count} extreme rainfall events")
        
        return df
    
    def _calculate_hurricane_severity(self, df_subset: pd.DataFrame, threshold: Dict) -> pd.Series:
        """Calculate hurricane severity based on wind speed and pressure"""
        
        wind_speed = df_subset['wind_speed']
        pressure = df_subset['pressure']
        
        # Saffir-Simpson Hurricane Wind Scale categories
        conditions = [
            wind_speed >= 157,  # Category 5
            wind_speed >= 130,  # Category 4
            wind_speed >= 111,  # Category 3
            wind_speed >= 96,   # Category 2
            wind_speed >= 74    # Category 1
        ]
        
        choices = ['extreme', 'high', 'high', 'moderate', 'moderate']
        
        return pd.Series(np.select(conditions, choices, default='low'), index=df_subset.index)
    
    def _calculate_flood_severity(self, df_subset: pd.DataFrame, threshold: Dict) -> pd.Series:
        """Calculate flood severity based on precipitation and pressure"""
        
        precipitation = df_subset['precipitation']
        base_threshold = threshold['precipitation_mm']
        
        conditions = [
            precipitation >= base_threshold * 2.5,  # Extreme
            precipitation >= base_threshold * 2.0,  # High
            precipitation >= base_threshold * 1.5,  # Moderate
            precipitation >= base_threshold         # Low
        ]
        
        choices = ['extreme', 'high', 'moderate', 'low']
        
        return pd.Series(np.select(conditions, choices, default='low'), index=df_subset.index)
    
    def _calculate_storm_severity(self, df_subset: pd.DataFrame, threshold: Dict) -> pd.Series:
        """Calculate storm severity based on wind speed and pressure"""
        
        wind_speed = df_subset['wind_speed']
        base_threshold = threshold['wind_speed_mph']
        
        conditions = [
            wind_speed >= base_threshold * 1.8,  # Extreme
            wind_speed >= base_threshold * 1.5,  # High
            wind_speed >= base_threshold * 1.2,  # Moderate
            wind_speed >= base_threshold         # Low
        ]
        
        choices = ['extreme', 'high', 'moderate', 'low']
        
        return pd.Series(np.select(conditions, choices, default='low'), index=df_subset.index)
    
    def _calculate_rainfall_severity(self, df_subset: pd.DataFrame, threshold: Dict) -> pd.Series:
        """Calculate extreme rainfall severity"""
        
        precipitation = df_subset['precipitation']
        base_threshold = threshold['precipitation_mm']
        
        conditions = [
            precipitation >= base_threshold * 2.0,  # Extreme
            precipitation >= base_threshold * 1.7,  # High
            precipitation >= base_threshold * 1.3,  # Moderate
            precipitation >= base_threshold         # Low
        ]
        
        choices = ['extreme', 'high', 'moderate', 'low']
        
        return pd.Series(np.select(conditions, choices, default='low'), index=df_subset.index)
    
    def _calculate_confidence(self, df_subset: pd.DataFrame, disaster_type: str, 
                            threshold: Dict) -> pd.Series:
        """Calculate confidence score for disaster classification"""
        
        confidence_scores = []
        
        for idx, row in df_subset.iterrows():
            score = 0.5  # Base confidence
            
            if disaster_type == 'hurricane':
                # Higher confidence for more extreme conditions
                wind_ratio = row['wind_speed'] / threshold['wind_speed_mph']
                pressure_ratio = threshold['pressure_hpa'] / row['pressure']
                score = min(0.95, 0.5 + (wind_ratio - 1) * 0.2 + (pressure_ratio - 1) * 0.2)
            
            elif disaster_type == 'flood':
                precip_ratio = row['precipitation'] / threshold['precipitation_mm']
                pressure_ratio = threshold['pressure_hpa'] / row['pressure']
                score = min(0.95, 0.5 + (precip_ratio - 1) * 0.15 + (pressure_ratio - 1) * 0.15)
            
            elif disaster_type == 'storm':
                wind_ratio = row['wind_speed'] / threshold['wind_speed_mph']
                pressure_ratio = threshold['pressure_hpa'] / row['pressure']
                score = min(0.95, 0.5 + (wind_ratio - 1) * 0.15 + (pressure_ratio - 1) * 0.15)
            
            elif disaster_type == 'extreme_rainfall':
                precip_ratio = row['precipitation'] / threshold['precipitation_mm']
                score = min(0.95, 0.5 + (precip_ratio - 1) * 0.25)
            
            confidence_scores.append(max(0.1, score))  # Minimum confidence of 0.1
        
        return pd.Series(confidence_scores, index=df_subset.index)
    
    def _calculate_disaster_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics about identified disasters"""
        
        total_records = len(df)
        disaster_records = (df['disaster_occurred'] == 1).sum()
        
        stats = {
            'total_records': total_records,
            'disaster_records': disaster_records,
            'disaster_percentage': (disaster_records / total_records) * 100 if total_records > 0 else 0
        }
        
        # Count by disaster type
        disaster_counts = df[df['disaster_occurred'] == 1]['disaster_type'].value_counts()
        stats['disaster_types'] = disaster_counts.to_dict()
        
        # Count by severity
        severity_counts = df[df['disaster_occurred'] == 1]['disaster_severity'].value_counts()
        stats['severity_distribution'] = severity_counts.to_dict()
        
        return stats
    
    def test_labeling_logic(self, sample_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Test disaster labeling logic with sample data
        
        Args:
            sample_data: Optional sample data. If None, creates test data.
            
        Returns:
            Dictionary with test results
        """
        logger.info("Testing disaster labeling logic...")
        
        if sample_data is None:
            # Create test data with known extreme conditions
            test_data = pd.DataFrame({
                'temperature': [25, 30, 15, 35, 20],
                'pressure': [970, 1015, 995, 975, 1010],  # Very low, normal, low, very low, normal
                'wind_speed': [80, 15, 45, 90, 25],       # Hurricane, calm, storm, hurricane, moderate
                'precipitation': [200, 5, 120, 180, 50],  # Extreme, light, heavy, extreme, moderate
                'humidity': [85, 60, 75, 90, 65]
            })
        else:
            test_data = sample_data.copy()
        
        # Apply labeling
        labeled_data = self.create_disaster_labels(test_data)
        
        # Analyze results
        results = {
            'input_records': len(test_data),
            'labeled_disasters': (labeled_data['disaster_occurred'] == 1).sum(),
            'disaster_types': labeled_data['disaster_type'].value_counts().to_dict(),
            'severity_levels': labeled_data['disaster_severity'].value_counts().to_dict(),
            'sample_results': labeled_data[['temperature', 'pressure', 'wind_speed', 
                                          'precipitation', 'disaster_type', 'disaster_severity']].to_dict('records')
        }
        
        logger.info(f"Test completed: {results['labeled_disasters']}/{results['input_records']} records labeled as disasters")
        return results


def create_disaster_labels(weather_df: pd.DataFrame, 
                         region_type: str = 'temperate',
                         custom_thresholds: Optional[Dict] = None) -> pd.DataFrame:
    """
    Convenience function to create disaster labels from weather data
    
    Args:
        weather_df: DataFrame with weather data
        region_type: Type of region ('tropical', 'temperate', 'arid')
        custom_thresholds: Optional custom threshold dictionary
        
    Returns:
        DataFrame with added disaster columns
    """
    labeler = DisasterLabeler()
    return labeler.create_disaster_labels(weather_df, region_type, custom_thresholds)


if __name__ == "__main__":
    # Example usage and testing
    labeler = DisasterLabeler()
    
    # Test with sample data
    test_results = labeler.test_labeling_logic()
    
    print("\nDisaster Labeling Test Results:")
    print("=" * 50)
    print(f"Input records: {test_results['input_records']}")
    print(f"Labeled disasters: {test_results['labeled_disasters']}")
    print(f"Disaster types: {test_results['disaster_types']}")
    print(f"Severity levels: {test_results['severity_levels']}")
    
    print("\nSample Results:")
    for i, result in enumerate(test_results['sample_results']):
        print(f"Record {i+1}: {result['disaster_type']} ({result['disaster_severity']})")
        print(f"  Conditions: T={result['temperature']}°C, P={result['pressure']}hPa, "
              f"W={result['wind_speed']}mph, R={result['precipitation']}mm")