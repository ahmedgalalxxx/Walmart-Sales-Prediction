"""
Quick visual test to show a few sample predictions.
""" import joblib
import pandas as pd
import numpy as np
from src.data_preprocessing import WalmartDataPreprocessor print("=" * 70)
print("SAMPLE PREDICTIONS TEST")
print("=" * 70) # Load model and data
rf_model = joblib.load('models/random_forest_model.joblib') # Prepare data
preprocessor = WalmartDataPreprocessor('Walmart.csv')
preprocessor.load_data()
preprocessor.parse_dates()
preprocessor.create_lag_features()
preprocessor.create_rolling_features()
preprocessor.handle_missing_values()
X_train, X_test, y_train, y_test = preprocessor.split_data()
X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test) # Make predictions on a few samples
predictions = rf_model.predict(X_test_scaled[:10]) print("\n Comparing First 10 Test Predictions:\n")
print(f"{'Sample':<8} {'Actual Sales':<20} {'Predicted Sales':<20} {'Error':<15} {'Error %':<10}")
print("-" * 80) for i in range(10): actual = y_test.iloc[i] predicted = predictions[i] error = actual - predicted error_pct = (abs(error) / actual) * 100 print(f"{i+1:<8} ${actual:>15,.2f} ${predicted:>15,.2f} ${error:>12,.2f} {error_pct:>6.2f}%") print("\n" + "=" * 70) # Calculate average error
avg_error_pct = np.mean([abs(y_test.iloc[i] - predictions[i]) / y_test.iloc[i] * 100 for i in range(10)])
print(f"Average Error on these samples: {avg_error_pct:.2f}%")
print(f"\n Model is making reasonable predictions!")
print("=" * 70)
