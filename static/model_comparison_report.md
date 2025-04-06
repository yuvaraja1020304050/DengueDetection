# Dengue Prediction Model Comparison Report

## Overview
This report compares the performance of three machine learning models for dengue prediction:
- Random Forest
- XGBoost
- Support Vector Machine (SVM)

## Performance Metrics

### Accuracy
- Random Forest: 0.6050
- XGBoost: 0.5700
- SVM: 0.5950

**Best Model: Random Forest** with accuracy 0.6050

### Classification Reports

#### Random Forest
```
              precision    recall  f1-score   support

           0       0.59      0.69      0.64       100
           1       0.63      0.52      0.57       100

    accuracy                           0.60       200
   macro avg       0.61      0.60      0.60       200
weighted avg       0.61      0.60      0.60       200

```

#### XGBoost
```
              precision    recall  f1-score   support

           0       0.56      0.64      0.60       100
           1       0.58      0.50      0.54       100

    accuracy                           0.57       200
   macro avg       0.57      0.57      0.57       200
weighted avg       0.57      0.57      0.57       200

```

#### SVM
```
              precision    recall  f1-score   support

           0       0.59      0.65      0.62       100
           1       0.61      0.54      0.57       100

    accuracy                           0.59       200
   macro avg       0.60      0.59      0.59       200
weighted avg       0.60      0.59      0.59       200

```

## Visual Analysis

### Feature Importance
- See feature importance plots in the dashboard for insights on which factors most strongly influence dengue prediction.

### Confusion Matrices
- Confusion matrices showing true positives, false positives, true negatives, and false negatives are available in the dashboard.

## Conclusion
Based on our evaluation, **Random Forest** performs best for dengue prediction with our current dataset, achieving the highest accuracy score. This model should be prioritized for making predictions in the application.
