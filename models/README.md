# Disaster Early Warning System - Model Documentation

This directory contains the trained machine learning models and evaluation artifacts for the Disaster Early Warning System.

## 📁 Directory Contents

### Model Files
- **`disaster_prediction_model.pkl`** (2.5 MB) - Binary classification model (Random Forest)
- **`disaster_type_model.pkl`** (500 KB) - Multi-class disaster type classifier (Gradient Boosting)
- **`feature_scaler.pkl`** (5 KB) - StandardScaler for feature normalization
- **`model_metadata.json`** (11 KB) - Complete model configuration and training metadata

### Evaluation Reports
- **`MODEL_EVALUATION_REPORT.md`** - Comprehensive evaluation report with metrics and analysis
- **`model_performance_report.json`** - Machine-readable performance metrics and feature importance

### Visualizations
The `visualizations/` directory contains:
- `confusion_matrix_binary.png` - Confusion matrix for binary classifier
- `feature_importance_binary.png` - Feature importance rankings
- `roc_curve_binary.png` - ROC curve analysis
- `precision_recall_curve_binary.png` - Precision-Recall curve
- `performance_metrics_binary.png` - Performance metrics comparison

## 🎯 Model Overview

### Binary Classification Model
- **Purpose**: Predict disaster occurrence from weather patterns
- **Algorithm**: Random Forest Classifier
- **Features**: 10 weather-related features
- **Performance**: 99.96% accuracy (see limitations below)

### Disaster Type Classification Model
- **Purpose**: Classify disaster type when disaster is predicted
- **Algorithm**: Gradient Boosting Classifier
- **Classes**: extreme_weather, storm
- **Performance**: 100% accuracy on small dataset

## 📊 Key Metrics

### Binary Classifier Performance
```
Accuracy:    99.96%
Precision:   0.00%
Recall:      0.00%
F1-Score:    0.00%
ROC-AUC:     0.5475
Specificity: 100.00%
Sensitivity: 0.00%
```

### Top 5 Most Important Features
1. **wind_speed** (35.58%) - Current wind speed
2. **wind_spike_max** (28.62%) - Maximum wind spike in 7-day window
3. **pressure** (13.71%) - Atmospheric pressure
4. **wind_gust_ratio** (13.15%) - Ratio of max gust to average wind
5. **pressure_drop_7d** (3.84%) - Maximum pressure drop over 7 days

## ⚠️ Important Limitations

**The current model is NOT production-ready due to:**

1. **Class Imbalance**: Severe imbalance (2,500:1 normal:disaster ratio)
2. **Zero Recall**: Model fails to identify any disaster events
3. **Limited Data**: Only 50 disaster events in training dataset
4. **Overfitting**: Model learned to always predict "no disaster"

## 🔧 Usage

### Loading Models

```python
import joblib

# Load binary classification model
binary_model = joblib.load('models/disaster_prediction_model.pkl')

# Load feature scaler
scaler = joblib.load('models/feature_scaler.pkl')

# Load disaster type model
type_model = joblib.load('models/disaster_type_model.pkl')
```

### Making Predictions

```python
import numpy as np

# Prepare features (10 features required)
features = np.array([[
    temperature,        # °C
    pressure,          # hPa
    wind_speed,        # mph
    humidity,          # %
    pressure_drop_7d,  # hPa
    wind_spike_max,    # mph
    humidity_trend,    # slope
    temp_deviation,    # °C
    pressure_velocity, # hPa/hour
    wind_gust_ratio    # ratio
]])

# Scale features
features_scaled = scaler.transform(features)

# Predict disaster probability
disaster_prob = binary_model.predict_proba(features_scaled)[0, 1]

# If high probability, classify disaster type
if disaster_prob > 0.5:
    disaster_type = type_model.predict(features_scaled)[0]
    print(f"Disaster Type: {disaster_type}")
```

## 📈 Evaluation Tools

### Generate Performance Report
```bash
python backend/utils/model_evaluator.py
```

This generates:
- `model_performance_report.json` - Detailed metrics in JSON format
- Console output with evaluation summary

### Generate Visualizations
```bash
python backend/utils/model_visualizer.py
```

This creates visualization files in `models/visualizations/`:
- Confusion matrices
- Feature importance charts
- ROC and Precision-Recall curves
- Performance metric comparisons

## 🔄 Model Versioning

### Current Version: 1.0.0 (2024-11-23)

**Changes:**
- Initial model training
- Established baseline performance
- Created evaluation framework

**Known Issues:**
- Class imbalance causing zero recall
- Limited disaster event data
- Not production-ready

### Next Version Goals (2.0.0)
- [ ] Address class imbalance with SMOTE/ADASYN
- [ ] Improve recall to >70%
- [ ] Expand disaster event dataset to 500+ samples
- [ ] Implement cost-sensitive learning
- [ ] Add model monitoring capabilities

## 🛠️ Retraining Models

To retrain the models with new data:

1. Update the database with new weather and disaster data
2. Run the preprocessing notebook: `notebooks/02_data_preprocessing.ipynb`
3. Run the training notebook: `notebooks/03_model_training.ipynb`
4. Evaluate the new models: `python backend/utils/model_evaluator.py`
5. Generate visualizations: `python backend/utils/model_visualizer.py`

## 📚 Additional Resources

- **Full Evaluation Report**: See `MODEL_EVALUATION_REPORT.md`
- **Training Notebook**: `notebooks/03_model_training.ipynb`
- **Evaluation Scripts**: `backend/utils/model_evaluator.py`, `backend/utils/model_visualizer.py`
- **Design Document**: `.kiro/specs/disaster-early-warning-system/design.md`

## 🤝 Contributing

When updating models:
1. Update version number in metadata
2. Document changes in this README
3. Regenerate evaluation reports
4. Update visualizations
5. Test with backend prediction engine

## 📞 Support

For questions or issues with the models:
- Review the evaluation report for detailed analysis
- Check the training notebook for implementation details
- Consult the design document for system architecture

---

**Last Updated**: 2024-11-23  
**Model Version**: 1.0.0  
**Status**: Development (Not Production Ready)
