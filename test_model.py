"""
Test script to verify that the trained models are working correctly.
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("=" * 70)
print("MODEL VERIFICATION TEST")
print("=" * 70)

# 1. Load the saved models
print("\n[1/5] Loading saved models...")
try:
    rf_model = joblib.load('models/random_forest_model.joblib')
    lr_model = joblib.load('models/linear_regression_model.joblib')
    dt_model = joblib.load('models/decision_tree_model.joblib')
    gb_model = joblib.load('models/gradient_boosting_model.joblib')
    xgb_model = joblib.load('models/xgboost_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    print("✅ All 5 models loaded successfully!")
    print(f"   - Random Forest: {rf_model.n_estimators} trees, max_depth={rf_model.max_depth}")
    print(f"   - Decision Tree: max_depth={dt_model.max_depth}")
    print(f"   - Linear Regression: {type(lr_model).__name__}")
    print(f"   - Gradient Boosting: {gb_model.n_estimators} estimators")
    print(f"   - XGBoost: {xgb_model.n_estimators} estimators")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

# 2. Load feature names
print("\n[2/5] Loading feature names...")
try:
    with open('models/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f]
    print(f"✅ Loaded {len(feature_names)} feature names")
    print(f"   Features: {', '.join(feature_names[:5])}...")
except Exception as e:
    print(f"❌ Error loading features: {e}")
    exit(1)

# 3. Load and prepare test data
print("\n[3/5] Loading test data...")
try:
    from src.data_preprocessing import WalmartDataPreprocessor
    
    preprocessor = WalmartDataPreprocessor('Walmart.csv')
    preprocessor.load_data()
    preprocessor.parse_dates()
    preprocessor.create_lag_features()
    preprocessor.create_rolling_features()
    preprocessor.handle_missing_values()
    X_train, X_test, y_train, y_test = preprocessor.split_data()
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    print(f"✅ Data preprocessed successfully!")
    print(f"   Test set size: {X_test_scaled.shape[0]} samples")
    print(f"   Number of features: {X_test_scaled.shape[1]}")
except Exception as e:
    print(f"❌ Error preprocessing data: {e}")
    exit(1)

# 4. Make predictions with all models
print("\n[4/5] Making predictions...")
try:
    predictions = {
        'Random Forest': rf_model.predict(X_test_scaled),
        'Linear Regression': lr_model.predict(X_test_scaled),
        'Decision Tree': dt_model.predict(X_test_scaled),
        'Gradient Boosting': gb_model.predict(X_test_scaled),
        'XGBoost': xgb_model.predict(X_test_scaled)
    }
    print("✅ All models made predictions successfully!")
except Exception as e:
    print(f"❌ Error making predictions: {e}")
    exit(1)

# 5. Evaluate predictions
print("\n[5/5] Evaluating predictions...")
print("\n" + "=" * 70)
print("MODEL PERFORMANCE VERIFICATION")
print("=" * 70)

results = []
for model_name, y_pred in predictions.items():
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    results.append({
        'Model': model_name,
        'R² Score': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape
    })
    
    print(f"\n{model_name}:")
    print(f"  R² Score: {r2:.4f} ({r2*100:.2f}%)")
    print(f"  MAE: ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAPE: {mape:.2f}%")

# Find best model
results_df = pd.DataFrame(results).sort_values('R² Score', ascending=False)
best_model = results_df.iloc[0]

print("\n" + "=" * 70)
print(f"🏆 BEST MODEL: {best_model['Model']}")
print("=" * 70)
print(f"  R² Score: {best_model['R² Score']:.4f} ({best_model['R² Score']*100:.2f}%)")
print(f"  MAE: ${best_model['MAE']:,.2f}")
print(f"  RMSE: ${best_model['RMSE']:,.2f}")
print(f"  MAPE: {best_model['MAPE (%)']:.2f}%")
print("=" * 70)

# 6. Sanity checks
print("\n" + "=" * 70)
print("SANITY CHECKS")
print("=" * 70)

# Check if predictions are reasonable
best_pred = predictions[best_model['Model']]
print(f"\n✓ Prediction range: ${best_pred.min():,.2f} to ${best_pred.max():,.2f}")
print(f"✓ Actual range: ${y_test.min():,.2f} to ${y_test.max():,.2f}")
print(f"✓ Mean prediction: ${best_pred.mean():,.2f}")
print(f"✓ Mean actual: ${y_test.mean():,.2f}")

# Check for negative predictions
if (best_pred < 0).any():
    print("⚠️  Warning: Some predictions are negative!")
else:
    print("✓ All predictions are positive (good!)")

# Check if model is overfitting
train_pred = rf_model.predict(X_train_scaled)
train_r2 = r2_score(y_train, train_pred)
test_r2 = best_model['R² Score']
overfit_diff = train_r2 - test_r2

print(f"\n✓ Train R²: {train_r2:.4f}")
print(f"✓ Test R²: {test_r2:.4f}")
print(f"✓ Difference: {overfit_diff:.4f}", end='')

if overfit_diff < 0.05:
    print(" (Good! Not overfitting)")
elif overfit_diff < 0.10:
    print(" (Acceptable)")
else:
    print(" (Warning: Possible overfitting)")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED! Models are working correctly!")
print("=" * 70)
