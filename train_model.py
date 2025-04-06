import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_selection import SelectFromModel
import markdown

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load dataset
def load_data():
    try:
        data = pd.read_csv('dengue_dataset.csv')
        print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns")
        return data
    except FileNotFoundError:
        print("Dataset not found. Creating a sample dataset...")
        
        # Create sample dataset if file doesn't exist
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample data
        genders = np.random.choice(['male', 'female'], size=n_samples)
        ages = np.random.randint(1, 100, size=n_samples)
        nsi_values = np.random.uniform(0, 10, size=n_samples)
        igg_values = np.random.uniform(0, 15, size=n_samples)
        areas = np.random.choice(['urban', 'rural', 'suburban'], size=n_samples)
        area_types = np.random.choice(['residential', 'commercial', 'industrial'], size=n_samples)
        house_types = np.random.choice(['apartment', 'house', 'slum'], size=n_samples)
        districts = np.random.choice(['district1', 'district2', 'district3', 'district4', 'district5'], size=n_samples)
        
        # Generate outcomes based on some rules to make it more realistic
        outcomes = []
        for i in range(n_samples):
            # Higher NSI and IgG values increase likelihood of dengue
            base_prob = 0.2
            
            if nsi_values[i] > 7:
                base_prob += 0.3
            if igg_values[i] > 10:
                base_prob += 0.3
                
            # Age factors
            if ages[i] < 15 or ages[i] > 65:
                base_prob += 0.1
                
            # Area factors
            if areas[i] == 'urban' and house_types[i] == 'slum':
                base_prob += 0.2
                
            # Cap probability at 0.9
            base_prob = min(base_prob, 0.9)
            
            outcome = np.random.choice([0, 1], p=[1-base_prob, base_prob])
            outcomes.append(outcome)
        
        # Create DataFrame
        data = pd.DataFrame({
            'gender': genders,
            'age': ages,
            'nsi': nsi_values,
            'igg': igg_values,
            'area': areas,
            'area_type': area_types,
            'house_type': house_types,
            'district': districts,
            'outcome': outcomes
        })
        
        # Save to CSV
        data.to_csv('dengue_dataset.csv', index=False)
        print(f"Sample dataset created with {n_samples} rows")
        return data

# Preprocess data
def preprocess_data(data):
    # Handle categorical variables
    categorical_cols = ['gender', 'area', 'area_type', 'house_type', 'district']
    
    # One-hot encode categorical variables
    X_categorical = pd.get_dummies(data[categorical_cols], drop_first=False)
    
    # Combine with numerical features
    X_numerical = data[['age', 'nsi', 'igg']]
    X = pd.concat([X_numerical, X_categorical], axis=1)
    
    # Target variable
    y = data['outcome']
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, feature_names

# Train models
def train_models(X_train, X_test, y_train, y_test, feature_names):
    models = {}
    performance = {}
    
    # Random Forest
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    models['random_forest_model'] = rf_model
    performance['Random Forest'] = {
        'accuracy': rf_accuracy,
        'confusion_matrix': confusion_matrix(y_test, rf_pred),
        'classification_report': classification_report(y_test, rf_pred)
    }
    print(f"Random Forest accuracy: {rf_accuracy:.4f}")
    
    # XGBoost
    print("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    models['xgboost_model'] = xgb_model
    performance['XGBoost'] = {
        'accuracy': xgb_accuracy,
        'confusion_matrix': confusion_matrix(y_test, xgb_pred),
        'classification_report': classification_report(y_test, xgb_pred)
    }
    print(f"XGBoost accuracy: {xgb_accuracy:.4f}")
    
    # SVM
    print("Training SVM model...")
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    models['svm_model'] = svm_model
    performance['SVM'] = {
        'accuracy': svm_accuracy,
        'confusion_matrix': confusion_matrix(y_test, svm_pred),
        'classification_report': classification_report(y_test, svm_pred)
    }
    print(f"SVM accuracy: {svm_accuracy:.4f}")
    
    # Save feature names
    models['feature_names'] = feature_names
    
    # Determine best model
    best_model = max(performance.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\nBest performing model: {best_model} with accuracy {performance[best_model]['accuracy']:.4f}")
    
    return models, performance, best_model

# Generate visualizations
def generate_visualizations(models, X_test, y_test, feature_names):
    for model_name, model in models.items():
        if model_name == 'feature_names':
            continue
            
        # Get display name
        if model_name == 'random_forest_model':
            display_name = 'Random Forest'
        elif model_name == 'xgboost_model':
            display_name = 'XGBoost'
        else:
            display_name = 'SVM'
            
        # Feature importance (for tree-based models)
        if model_name in ['random_forest_model', 'xgboost_model']:
            plt.figure(figsize=(10, 6))
            feat_importances = pd.Series(model.feature_importances_, index=feature_names)
            feat_importances.nlargest(10).plot(kind='barh')
            plt.title(f'Top 10 Feature Importance - {display_name}')
            plt.tight_layout()
            plt.savefig(f'static/feature_importance_{display_name}.png')
            plt.close()
        
        # For SVM, we can use absolute values of coefficients if linear kernel
        elif model_name == 'svm_model' and getattr(model, 'kernel', None) == 'linear':
            plt.figure(figsize=(10, 6))
            feat_importances = pd.Series(np.abs(model.coef_[0]), index=feature_names)
            feat_importances.nlargest(10).plot(kind='barh')
            plt.title(f'Top 10 Feature Importance - {display_name}')
            plt.tight_layout()
            plt.savefig(f'static/feature_importance_{display_name}.png')
            plt.close()
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {display_name}')
        plt.tight_layout()
        plt.savefig(f'static/confusion_matrix_{display_name}.png')
        plt.close()

# Save models
def save_models(models):
    for model_name, model in models.items():
        with open(f'models/{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)
    print("Models saved successfully!")

# Generate model comparison report
def generate_report(performance, best_model):
    report = f"""# Dengue Prediction Model Comparison Report

## Overview
This report compares the performance of three machine learning models for dengue prediction:
- Random Forest
- XGBoost
- Support Vector Machine (SVM)

## Performance Metrics

### Accuracy
- Random Forest: {performance['Random Forest']['accuracy']:.4f}
- XGBoost: {performance['XGBoost']['accuracy']:.4f}
- SVM: {performance['SVM']['accuracy']:.4f}

**Best Model: {best_model}** with accuracy {performance[best_model]['accuracy']:.4f}

### Classification Reports

#### Random Forest
```
{performance['Random Forest']['classification_report']}
```

#### XGBoost
```
{performance['XGBoost']['classification_report']}
```

#### SVM
```
{performance['SVM']['classification_report']}
```

## Visual Analysis

### Feature Importance
- See feature importance plots in the dashboard for insights on which factors most strongly influence dengue prediction.

### Confusion Matrices
- Confusion matrices showing true positives, false positives, true negatives, and false negatives are available in the dashboard.

## Conclusion
Based on our evaluation, **{best_model}** performs best for dengue prediction with our current dataset, achieving the highest accuracy score. This model should be prioritized for making predictions in the application.
"""

    with open('static/model_comparison_report.md', 'w') as f:
        f.write(report)
    print("Model comparison report generated successfully!")

def main():
    print("Starting model training process...")
    
    # Load data
    data = load_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(data)
    
    # Train models
    models, performance, best_model = train_models(X_train, X_test, y_train, y_test, feature_names)
    
    # Generate visualizations
    generate_visualizations(models, X_test, y_test, feature_names)
    
    # Save models
    save_models(models)
    
    # Generate comparison report
    generate_report(performance, best_model)
    
    print("\nTraining process completed successfully!")
    print(f"The best performing model is {best_model} with accuracy {performance[best_model]['accuracy']:.4f}")

if __name__ == "__main__":
    main()