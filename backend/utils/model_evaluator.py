"""
Model Evaluation and Feature Importance Analysis

This module provides comprehensive evaluation tools for the disaster prediction models,
including confusion matrices, feature importance analysis, and performance reporting.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)


class ModelEvaluator:
    """Comprehensive model evaluation and analysis"""
    
    def __init__(self, models_dir: str = 'models', db_path: str = 'disaster_data.db'):
        """
        Initialize the model evaluator
        
        Args:
            models_dir: Directory containing trained models
            db_path: Path to SQLite database with test data
        """
        self.models_dir = Path(models_dir)
        self.db_path = db_path
        self.binary_model = None
        self.multiclass_model = None
        self.scaler = None
        self.metadata = None
        
    def load_models(self) -> bool:
        """Load trained models and metadata"""
        try:
            # Load binary classification model
            binary_path = self.models_dir / 'disaster_prediction_model.pkl'
            if binary_path.exists():
                self.binary_model = joblib.load(binary_path)
                print(f"✓ Loaded binary classification model from {binary_path}")
            
            # Load multiclass model (if available)
            multiclass_path = self.models_dir / 'disaster_type_model.pkl'
            if multiclass_path.exists():
                self.multiclass_model = joblib.load(multiclass_path)
                print(f"✓ Loaded disaster type classification model from {multiclass_path}")
            
            # Load feature scaler
            scaler_path = self.models_dir / 'feature_scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print(f"✓ Loaded feature scaler from {scaler_path}")
            
            # Load metadata
            metadata_path = self.models_dir / 'model_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✓ Loaded model metadata from {metadata_path}")
            
            return True
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            return False
    
    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load test data from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            test_df = pd.read_sql_query("SELECT * FROM test_data", conn)
            conn.close()
            
            # Extract features and target
            feature_columns = [col for col in test_df.columns if col != 'disaster_occurred']
            X_test = test_df[feature_columns]
            y_test = test_df['disaster_occurred']
            
            print(f"✓ Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            return X_test, y_test
        except Exception as e:
            print(f"✗ Error loading test data: {e}")
            return None, None
    
    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 model_name: str) -> Dict:
        """
        Generate confusion matrix and related metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary with confusion matrix and metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate metrics from confusion matrix
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            result = {
                'model_name': model_name,
                'confusion_matrix': cm.tolist(),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'specificity': float(specificity),
                'sensitivity': float(sensitivity)
            }
        else:
            result = {
                'model_name': model_name,
                'confusion_matrix': cm.tolist(),
                'matrix_shape': cm.shape
            }
        
        return result
    
    def calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='binary', zero_division=0))
        }
        
        # Add ROC-AUC if probabilities are provided and we have both classes
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
            except:
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def analyze_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Analyze and rank feature importance
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance rankings
        """
        if not hasattr(model, 'feature_importances_'):
            print("⚠ Model does not have feature_importances_ attribute")
            return pd.DataFrame()
        
        importance = model.feature_importances_
        
        # Create DataFrame with feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'importance_pct': importance * 100
        }).sort_values('importance', ascending=False)
        
        # Add cumulative importance
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        importance_df['cumulative_pct'] = importance_df['cumulative_importance'] * 100
        
        # Add rank
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def evaluate_binary_model(self) -> Dict:
        """Evaluate the binary classification model"""
        if self.binary_model is None:
            print("✗ Binary model not loaded")
            return {}
        
        print("\n" + "="*80)
        print("BINARY CLASSIFICATION MODEL EVALUATION")
        print("="*80)
        
        # Load test data
        X_test, y_test = self.load_test_data()
        if X_test is None:
            return {}
        
        # Make predictions
        y_pred = self.binary_model.predict(X_test)
        y_pred_proba = self.binary_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate confusion matrix
        cm_result = self.generate_confusion_matrix(y_test, y_pred, 'Binary Classification')
        
        # Analyze feature importance
        feature_names = X_test.columns.tolist()
        importance_df = self.analyze_feature_importance(self.binary_model, feature_names)
        
        # Print results
        print(f"\nPerformance Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric.upper():15s}: {value:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {cm_result.get('true_negatives', 'N/A')}")
        print(f"  False Positives: {cm_result.get('false_positives', 'N/A')}")
        print(f"  False Negatives: {cm_result.get('false_negatives', 'N/A')}")
        print(f"  True Positives:  {cm_result.get('true_positives', 'N/A')}")
        print(f"  Specificity:     {cm_result.get('specificity', 0):.4f}")
        print(f"  Sensitivity:     {cm_result.get('sensitivity', 0):.4f}")
        
        print(f"\nTop 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['rank']:2d}. {row['feature']:30s}: {row['importance_pct']:6.2f}% (cumulative: {row['cumulative_pct']:6.2f}%)")
        
        # Compile results
        results = {
            'model_type': 'binary_classification',
            'algorithm': 'RandomForestClassifier',
            'metrics': metrics,
            'confusion_matrix': cm_result,
            'feature_importance': importance_df.to_dict('records'),
            'top_features': importance_df.head(10).to_dict('records')
        }
        
        return results
    
    def evaluate_multiclass_model(self) -> Dict:
        """Evaluate the disaster type classification model"""
        if self.multiclass_model is None:
            print("\n⚠ Disaster type classification model not available")
            return {}
        
        print("\n" + "="*80)
        print("DISASTER TYPE CLASSIFICATION MODEL EVALUATION")
        print("="*80)
        
        try:
            # Load full processed data to get disaster events
            conn = sqlite3.connect(self.db_path)
            full_df = pd.read_sql_query("SELECT * FROM processed_weather_data", conn)
            conn.close()
            
            # Filter for disaster events only
            if 'disaster_type' not in full_df.columns:
                print("⚠ No disaster_type column found in data")
                return {}
            
            disaster_df = full_df[full_df['disaster_occurred'] == 1].copy()
            
            if len(disaster_df) == 0:
                print("⚠ No disaster events found for evaluation")
                return {}
            
            # Extract features and target
            feature_columns = [col for col in disaster_df.columns 
                             if col not in ['disaster_occurred', 'disaster_type']]
            X_disaster = disaster_df[feature_columns]
            y_disaster = disaster_df['disaster_type']
            
            # Make predictions
            y_pred = self.multiclass_model.predict(X_disaster)
            y_pred_proba = self.multiclass_model.predict_proba(X_disaster)
            
            # Calculate metrics
            accuracy = accuracy_score(y_disaster, y_pred)
            
            # Generate confusion matrix
            cm_result = self.generate_confusion_matrix(y_disaster, y_pred, 'Disaster Type Classification')
            
            # Analyze feature importance
            importance_df = self.analyze_feature_importance(self.multiclass_model, feature_columns)
            
            # Print results
            print(f"\nPerformance Metrics:")
            print(f"  ACCURACY: {accuracy:.4f}")
            
            print(f"\nDisaster Type Distribution:")
            type_counts = y_disaster.value_counts()
            for disaster_type, count in type_counts.items():
                print(f"  {disaster_type:20s}: {count:4d} samples")
            
            print(f"\nConfusion Matrix Shape: {cm_result.get('matrix_shape', 'N/A')}")
            
            print(f"\nTop 10 Most Important Features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"  {row['rank']:2d}. {row['feature']:30s}: {row['importance_pct']:6.2f}%")
            
            # Compile results
            results = {
                'model_type': 'multiclass_classification',
                'algorithm': 'GradientBoostingClassifier',
                'accuracy': float(accuracy),
                'classes': self.multiclass_model.classes_.tolist(),
                'confusion_matrix': cm_result,
                'feature_importance': importance_df.to_dict('records'),
                'top_features': importance_df.head(10).to_dict('records')
            }
            
            return results
            
        except Exception as e:
            print(f"✗ Error evaluating multiclass model: {e}")
            return {}
    
    def generate_performance_report(self, output_path: str = 'models/model_performance_report.json') -> bool:
        """
        Generate comprehensive model performance report
        
        Args:
            output_path: Path to save the report
            
        Returns:
            True if successful, False otherwise
        """
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE MODEL PERFORMANCE REPORT")
        print("="*80)
        
        # Load models
        if not self.load_models():
            return False
        
        # Evaluate binary model
        binary_results = self.evaluate_binary_model()
        
        # Evaluate multiclass model
        multiclass_results = self.evaluate_multiclass_model()
        
        # Compile comprehensive report
        report = {
            'report_metadata': {
                'generated_at': pd.Timestamp.now().isoformat(),
                'models_directory': str(self.models_dir),
                'database_path': self.db_path
            },
            'binary_classification_model': binary_results,
            'disaster_type_classification_model': multiclass_results,
            'model_metadata': self.metadata
        }
        
        # Save report
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n✓ Performance report saved to: {output_file}")
            print(f"  Report size: {output_file.stat().st_size / 1024:.2f} KB")
            
            return True
        except Exception as e:
            print(f"✗ Error saving report: {e}")
            return False
    
    def print_summary(self):
        """Print a summary of model evaluation"""
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY")
        print("="*80)
        
        if self.metadata:
            binary_meta = self.metadata.get('binary_model', {})
            print(f"\nBinary Classification Model:")
            print(f"  Algorithm: {binary_meta.get('model_type', 'N/A')}")
            
            training_samples = binary_meta.get('training_samples', 0)
            test_samples = binary_meta.get('test_samples', 0)
            print(f"  Training Samples: {training_samples:,}" if isinstance(training_samples, int) else f"  Training Samples: {training_samples}")
            print(f"  Test Samples: {test_samples:,}" if isinstance(test_samples, int) else f"  Test Samples: {test_samples}")
            print(f"  Features: {len(binary_meta.get('features', []))}")
            
            perf = binary_meta.get('performance', {})
            print(f"  Test Accuracy: {perf.get('accuracy', 0):.4f}")
            print(f"  Test F1-Score: {perf.get('f1', 0):.4f}")
            print(f"  Test ROC-AUC: {perf.get('roc_auc', 0):.4f}")
            
            multiclass_meta = self.metadata.get('multiclass_model', {})
            if multiclass_meta:
                print(f"\nDisaster Type Classification Model:")
                print(f"  Algorithm: {multiclass_meta.get('model_type', 'N/A')}")
                print(f"  Classes: {', '.join(multiclass_meta.get('classes', []))}")
                
                mc_training_samples = multiclass_meta.get('training_samples', 0)
                print(f"  Training Samples: {mc_training_samples:,}" if isinstance(mc_training_samples, int) else f"  Training Samples: {mc_training_samples}")
                print(f"  Accuracy: {multiclass_meta.get('accuracy', 0):.4f}")
        
        print("\n" + "="*80)


def main():
    """Main function to run model evaluation"""
    evaluator = ModelEvaluator()
    
    # Generate comprehensive performance report
    success = evaluator.generate_performance_report()
    
    if success:
        # Print summary
        evaluator.print_summary()
        print("\n✓ Model evaluation completed successfully!")
    else:
        print("\n✗ Model evaluation failed!")
    
    return success


if __name__ == '__main__':
    main()
