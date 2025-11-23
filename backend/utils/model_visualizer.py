"""
Model Visualization Tools

This module provides visualization tools for model evaluation including
confusion matrices, feature importance charts, and performance plots.
"""

import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve


class ModelVisualizer:
    """Generate visualizations for model evaluation"""
    
    def __init__(self, models_dir: str = 'models', db_path: str = 'disaster_data.db'):
        """
        Initialize the model visualizer
        
        Args:
            models_dir: Directory containing trained models
            db_path: Path to SQLite database with test data
        """
        self.models_dir = Path(models_dir)
        self.db_path = db_path
        self.binary_model = None
        self.multiclass_model = None
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette('husl')
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def load_models(self):
        """Load trained models"""
        binary_path = self.models_dir / 'disaster_prediction_model.pkl'
        if binary_path.exists():
            self.binary_model = joblib.load(binary_path)
            print(f"✓ Loaded binary model")
        
        multiclass_path = self.models_dir / 'disaster_type_model.pkl'
        if multiclass_path.exists():
            self.multiclass_model = joblib.load(multiclass_path)
            print(f"✓ Loaded multiclass model")
    
    def load_test_data(self):
        """Load test data from database"""
        conn = sqlite3.connect(self.db_path)
        test_df = pd.read_sql_query("SELECT * FROM test_data", conn)
        conn.close()
        
        feature_columns = [col for col in test_df.columns if col != 'disaster_occurred']
        X_test = test_df[feature_columns]
        y_test = test_df['disaster_occurred']
        
        return X_test, y_test
    
    def plot_confusion_matrix(self, y_true, y_pred, title, output_path):
        """
        Plot confusion matrix heatmap
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            output_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['No Disaster', 'Disaster'],
                   yticklabels=['No Disaster', 'Disaster'])
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved confusion matrix to: {output_file}")
    
    def plot_feature_importance(self, model, feature_names, title, output_path, top_n=15):
        """
        Plot feature importance bar chart
        
        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
            title: Plot title
            output_path: Path to save the plot
            top_n: Number of top features to display
        """
        if not hasattr(model, 'feature_importances_'):
            print("⚠ Model does not have feature_importances_")
            return
        
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(importance_df.iterrows()):
            plt.text(row['importance'], i, f" {row['importance']:.4f}", 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved feature importance to: {output_file}")
    
    def plot_roc_curve(self, y_true, y_pred_proba, title, output_path):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            output_path: Path to save the plot
        """
        if len(np.unique(y_true)) < 2:
            print("⚠ Cannot plot ROC curve: only one class present")
            return
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved ROC curve to: {output_file}")
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, title, output_path):
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            output_path: Path to save the plot
        """
        if len(np.unique(y_true)) < 2:
            print("⚠ Cannot plot PR curve: only one class present")
            return
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved Precision-Recall curve to: {output_file}")
    
    def plot_performance_comparison(self, metrics_dict, output_path):
        """
        Plot performance metrics comparison
        
        Args:
            metrics_dict: Dictionary with metric names and values
            output_path: Path to save the plot
        """
        metrics_names = list(metrics_dict.keys())
        metrics_values = list(metrics_dict.values())
        
        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_names)))
        bars = plt.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=1.5)
        
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
        plt.ylim([0, 1.0])
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved performance comparison to: {output_file}")
    
    def generate_all_visualizations(self, output_dir='models/visualizations'):
        """Generate all model visualizations"""
        print("\n" + "="*80)
        print("GENERATING MODEL VISUALIZATIONS")
        print("="*80)
        
        # Load models and data
        self.load_models()
        X_test, y_test = self.load_test_data()
        
        if self.binary_model is None:
            print("✗ Binary model not loaded")
            return False
        
        # Make predictions
        y_pred = self.binary_model.predict(X_test)
        y_pred_proba = self.binary_model.predict_proba(X_test)[:, 1]
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Confusion Matrix
        self.plot_confusion_matrix(
            y_test, y_pred,
            'Binary Classification - Confusion Matrix',
            output_path / 'confusion_matrix_binary.png'
        )
        
        # 2. Feature Importance
        feature_names = X_test.columns.tolist()
        self.plot_feature_importance(
            self.binary_model, feature_names,
            'Binary Classification - Feature Importance',
            output_path / 'feature_importance_binary.png'
        )
        
        # 3. ROC Curve
        if len(np.unique(y_test)) > 1:
            self.plot_roc_curve(
                y_test, y_pred_proba,
                'Binary Classification - ROC Curve',
                output_path / 'roc_curve_binary.png'
            )
            
            # 4. Precision-Recall Curve
            self.plot_precision_recall_curve(
                y_test, y_pred_proba,
                'Binary Classification - Precision-Recall Curve',
                output_path / 'precision_recall_curve_binary.png'
            )
        
        # 5. Performance Metrics Comparison
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        self.plot_performance_comparison(
            metrics,
            output_path / 'performance_metrics_binary.png'
        )
        
        print(f"\n✓ All visualizations saved to: {output_path}")
        return True


def main():
    """Main function to generate visualizations"""
    visualizer = ModelVisualizer()
    success = visualizer.generate_all_visualizations()
    
    if success:
        print("\n✓ Visualization generation completed successfully!")
    else:
        print("\n✗ Visualization generation failed!")
    
    return success


if __name__ == '__main__':
    main()
