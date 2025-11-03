"""
Comprehensive Overfitting Detection Test
Tests for overfitting by comparing training vs test performance and using cross-validation.
""" import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt
from src.data_preprocessing import WalmartDataPreprocessor print("=" * 80)
print("OVERFITTING DETECTION TEST")
print("=" * 80) # Load models
print("\n[1/4] Loading models...")
models = { 'Random Forest': joblib.load('models/random_forest_model.joblib'), 'Linear Regression': joblib.load('models/linear_regression_model.joblib'), 'Decision Tree': joblib.load('models/decision_tree_model.joblib'), 'Gradient Boosting': joblib.load('models/gradient_boosting_model.joblib'), 'XGBoost': joblib.load('models/xgboost_model.joblib')
}
print(" All models loaded") # Prepare data
print("\n[2/4] Preparing data...")
preprocessor = WalmartDataPreprocessor('Walmart.csv')
preprocessor.load_data()
preprocessor.parse_dates()
preprocessor.create_lag_features()
preprocessor.create_rolling_features()
preprocessor.handle_missing_values()
X_train, X_test, y_train, y_test = preprocessor.split_data()
X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
print(" Data prepared") # Test 1: Train vs Test Performance
print("\n" + "=" * 80)
print("TEST 1: TRAIN VS TEST PERFORMANCE (Overfitting Indicator)")
print("=" * 80)
print("\nA large gap between train and test scores indicates overfitting.")
print("Ideal difference: < 5% (Good), 5-10% (Acceptable), > 10% (Overfitting)\n") results = []
for name, model in models.items(): # Train predictions y_train_pred = model.predict(X_train_scaled) train_r2 = r2_score(y_train, y_train_pred) train_mae = mean_absolute_error(y_train, y_train_pred) # Test predictions y_test_pred = model.predict(X_test_scaled) test_r2 = r2_score(y_test, y_test_pred) test_mae = mean_absolute_error(y_test, y_test_pred) # Calculate differences r2_diff = (train_r2 - test_r2) * 100 # Convert to percentage points mae_diff_pct = ((test_mae - train_mae) / train_mae) * 100 # Determine status if r2_diff < 5: status = " GOOD (No overfitting)" elif r2_diff < 10: status = " ACCEPTABLE (Slight overfitting)" else: status = " OVERFITTING DETECTED" results.append({ 'Model': name, 'Train R²': train_r2, 'Test R²': test_r2, 'R² Difference': r2_diff, 'Train MAE': train_mae, 'Test MAE': test_mae, 'Status': status }) print(f"\n{name}:") print(f" Train R²: {train_r2:.4f} ({train_r2*100:.2f}%)") print(f" Test R²: {test_r2:.4f} ({test_r2*100:.2f}%)") print(f" Difference: {r2_diff:.2f} percentage points") print(f" Train MAE: ${train_mae:,.2f}") print(f" Test MAE: ${test_mae:,.2f}") print(f" {status}") results_df = pd.DataFrame(results) # Test 2: Cross-Validation
print("\n" + "=" * 80)
print("TEST 2: CROSS-VALIDATION (5-Fold)")
print("=" * 80)
print("\nCross-validation tests model stability across different data splits.")
print("Low standard deviation indicates consistent, generalizable model.\n") # Combine all data for CV
X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test]) # We need to re-scale the combined data
from sklearn.preprocessing import StandardScaler
scaler_cv = StandardScaler()
numeric_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
X_all_scaled = X_all.copy()
X_all_scaled[numeric_cols] = scaler_cv.fit_transform(X_all[numeric_cols]) cv_results = []
for name, model in models.items(): print(f"\n{name}:") # Perform 5-fold cross-validation cv_scores = cross_val_score(model, X_all_scaled, y_all, cv=5, scoring='r2', n_jobs=-1) mean_cv = cv_scores.mean() std_cv = cv_scores.std() # Compare with test score test_r2 = results_df[results_df['Model'] == name]['Test R²'].values[0] cv_vs_test_diff = abs(mean_cv - test_r2) * 100 if std_cv < 0.02: stability = " Very Stable" elif std_cv < 0.05: stability = " Stable" else: stability = " Variable" print(f" CV R² Scores: {cv_scores}") print(f" Mean CV R²: {mean_cv:.4f} ({mean_cv*100:.2f}%)") print(f" Std Dev: {std_cv:.4f} ({stability})") print(f" Test R²: {test_r2:.4f}") print(f" CV vs Test Difference: {cv_vs_test_diff:.2f} percentage points") cv_results.append({ 'Model': name, 'Mean CV R²': mean_cv, 'Std Dev': std_cv, 'Test R²': test_r2, 'Stability': stability }) cv_df = pd.DataFrame(cv_results) # Test 3: Residual Analysis
print("\n" + "=" * 80)
print("TEST 3: RESIDUAL ANALYSIS (Best Model)")
print("=" * 80) # Use best model (Random Forest)
best_model = models['Random Forest']
y_train_pred = best_model.predict(X_train_scaled)
y_test_pred = best_model.predict(X_test_scaled) train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred print("\nRandom Forest Residual Statistics:")
print(f"\nTraining Set:")
print(f" Mean Residual: ${train_residuals.mean():,.2f}")
print(f" Std Dev: ${train_residuals.std():,.2f}")
print(f" Min: ${train_residuals.min():,.2f}")
print(f" Max: ${train_residuals.max():,.2f}") print(f"\nTest Set:")
print(f" Mean Residual: ${test_residuals.mean():,.2f}")
print(f" Std Dev: ${test_residuals.std():,.2f}")
print(f" Min: ${test_residuals.min():,.2f}")
print(f" Max: ${test_residuals.max():,.2f}") # Check if residuals are similar
std_ratio = test_residuals.std() / train_residuals.std()
print(f"\nStd Dev Ratio (Test/Train): {std_ratio:.2f}")
if 0.8 <= std_ratio <= 1.2: print(" GOOD: Similar error distribution on train and test sets")
elif 0.6 <= std_ratio <= 1.5: print(" ACCEPTABLE: Moderately different error distributions")
else: print(" WARNING: Very different error distributions (possible overfitting)") # Test 4: Prediction Distribution
print("\n" + "=" * 80)
print("TEST 4: PREDICTION DISTRIBUTION CHECK")
print("=" * 80) print("\nChecking if predictions cover similar ranges on train and test sets...")
print(f"\nRandom Forest Predictions:")
print(f" Train Range: ${y_train_pred.min():,.2f} to ${y_train_pred.max():,.2f}")
print(f" Test Range: ${y_test_pred.min():,.2f} to ${y_test_pred.max():,.2f}")
print(f" Train Mean: ${y_train_pred.mean():,.2f}")
print(f" Test Mean: ${y_test_pred.mean():,.2f}") print(f"\nActual Values:")
print(f" Train Range: ${y_train.min():,.2f} to ${y_train.max():,.2f}")
print(f" Test Range: ${y_test.min():,.2f} to ${y_test.max():,.2f}")
print(f" Train Mean: ${y_train.mean():,.2f}")
print(f" Test Mean: ${y_test.mean():,.2f}") # Check if test predictions are within training range (good sign)
if y_test_pred.min() >= y_train.min() * 0.8 and y_test_pred.max() <= y_train.max() * 1.2: print("\n GOOD: Test predictions are within reasonable range of training data")
else: print("\n WARNING: Test predictions extend beyond training data range") # Final Summary
print("\n" + "=" * 80)
print("OVERFITTING TEST SUMMARY")
print("=" * 80) overfitting_count = 0
for _, row in results_df.iterrows(): if "OVERFITTING" in row['Status']: overfitting_count += 1 print(f"\nModels with overfitting: {overfitting_count} out of {len(models)}") if overfitting_count == 0: print("\n EXCELLENT: No overfitting detected in any model!") print(" Your models are generalizing well to unseen data.")
elif overfitting_count <= 2: print("\n ACCEPTABLE: Some models show slight overfitting.") print(" Consider using the models with better train/test balance.")
else: print("\n CONCERN: Multiple models showing overfitting.") print(" Consider: reducing model complexity, adding regularization, or getting more data.") print("\nBest Model for Production: Random Forest")
rf_result = results_df[results_df['Model'] == 'Random Forest'].iloc[0]
print(f" R² Difference: {rf_result['R² Difference']:.2f} percentage points")
print(f" Status: {rf_result['Status']}") print("\n" + "=" * 80)
print(" OVERFITTING TEST COMPLETE!")
print("=" * 80)
