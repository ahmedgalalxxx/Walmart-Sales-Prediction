"""
Run comprehensive validation on the Walmart Sales Prediction model
"""
import sys
sys.path.append('src') import pandas as pd
import numpy as np
from data_preprocessing import WalmartDataPreprocessor
from model_validation import ModelValidator print("="*70)
print("WALMART SALES PREDICTION - MODEL VALIDATION")
print("="*70) # 1. Load and preprocess data
print("\n[Step 1/5] Loading and preprocessing data...")
preprocessor = WalmartDataPreprocessor('Walmart.csv')
preprocessor.load_data()
preprocessor.parse_dates()
preprocessor.create_lag_features()
preprocessor.create_rolling_features()
preprocessor.handle_missing_values()
preprocessor.prepare_features() # Split data
X_train, X_test, y_train, y_test = preprocessor.split_data(test_size=0.2, random_state=42) # Scale features
exclude_cols = ['Store', 'Holiday_Flag', 'Year']
X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test, exclude_cols=exclude_cols) print(f" Data prepared: {X_train_scaled.shape[0]} training, {X_test_scaled.shape[0]} test samples") # 2. Load the best model (Gradient Boosting)
print("\n[Step 2/5] Loading best model...")
validator = ModelValidator(model_path='models/gradient_boosting_model.joblib') # 3. Data Quality Validation
print("\n[Step 3/5] Validating data quality...")
data_quality_ok = validator.validate_data_quality(X_test_scaled, y_test) if not data_quality_ok: print("\n WARNING: Data quality issues detected. Proceeding with caution...") # 4. Cross-Validation
print("\n[Step 4/5] Performing cross-validation...")
cv_scores = validator.cross_validate(X_train_scaled, y_train, cv=5, scoring='r2') # 5. Time Series Cross-Validation
print("\n[Step 5/5] Performing time series cross-validation...")
ts_scores = validator.time_series_validation(X_train_scaled, y_train, n_splits=5) # 6. Prediction Validation
print("\n[Step 6/7] Validating predictions on test set...")
y_pred = validator.validate_predictions(X_test_scaled, y_test, confidence_level=0.95) # 7. Residual Analysis
print("\n[Step 7/7] Performing residual analysis...")
validator.residual_analysis(X_test_scaled, y_test, y_pred) # 8. Generate Report
print("\n[Final] Generating validation report...")
validator.generate_validation_report(save_path='results/validation_report.txt') print("\n" + "="*70)
print(" VALIDATION COMPLETE!")
print("="*70)
print("\nValidation Summary:")
print(f" Data Quality: {'PASSED' if data_quality_ok else 'WARNING'}")
print(f" Cross-Validation Mean R²: {cv_scores.mean():.4f}")
print(f" Time Series CV Mean R²: {ts_scores.mean():.4f}")
print(f" Test Set R²: {validator.validation_results['r2']:.4f}")
print(f" Test Set MAPE: {validator.validation_results['mape']:.2f}%")
print(f" Predictions within 10%: {validator.validation_results['within_10_pct']:.2f}%")
print("="*70)
print("\n Check 'results/validation_report.txt' for detailed report")
