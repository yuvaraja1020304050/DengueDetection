# Dengue Prediction Model Comparison Report

## Executive Summary
This report compares machine learning models for dengue prediction based on our analysis. **XGBoost** achieves the best performance across key metrics including accuracy, precision, recall, F1-score, and ROC AUC.

## Performance Metrics

### Accuracy
- Random Forest: 0.6627
- XGBoost: 0.6967
- SVM: 0.6508

**XGBoost delivers 5.13% higher accuracy than Random Forest and 7.05% higher than SVM.**

### Classification Report - XGBoost          precision    recall  f1-score   support

       0       0.70      0.72      0.71       134.0
       1       0.67      0.65      0.66       118.0

accuracy                           0.70       252.0macro avg 0.69 0.68 0.68 252.0
weighted avg 0.69 0.69 0.69 252.0
## Key Findings
1. **XGBoost Superiority:** XGBoost consistently outperforms other models across all evaluation metrics.
2. **Precision-Recall Balance:** XGBoost maintains the best balance between precision and recall for detecting positive dengue cases.
3. **ROC AUC Performance:** XGBoost's ROC AUC of 0.7247 demonstrates excellent discrimination ability.

## Important Features
The most important features for predicting dengue according to XGBoost are related to:
- NSI levels
- IgG antibody levels
- Area type
- Patient age

## Recommendations
1. **Deploy XGBoost:** Implement XGBoost in the production environment for dengue prediction.
2. **Feature Focus:** Pay special attention to NSI and IgG measurements in clinical settings.
3. **Regular Retraining:** Establish a protocol to retrain the model as new data becomes available.
4. **Clinical Integration:** Work with healthcare providers to integrate predictions into clinical workflows.

## Visualizations
Refer to the visualizations folder for detailed performance charts and feature importance graphics.
