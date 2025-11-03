"""
Quick run script for Walmart Sales Prediction
"""
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from data_preprocessing import WalmartDataPreprocessor
from model_training import WalmartModelTrainer
from model_evaluation import WalmartModelEvaluator

print("=" * 70)
print("WALMART SALES PREDICTION - AUTOMATED RUN")
print("=" * 70)

# 1. Data Preprocessing
print("\n[1/3] PREPROCESSING DATA...")
print("-" * 70)
preprocessor = WalmartDataPreprocessor('Walmart.csv')
preprocessor.load_data()
preprocessor.parse_dates()
preprocessor.create_lag_features()
preprocessor.create_rolling_features()
preprocessor.handle_missing_values()
preprocessor.prepare_features()

# Split data
X_train, X_test, y_train, y_test = preprocessor.split_data(test_size=0.2, random_state=42)

# Scale (exclude datetime and categorical columns)
exclude_cols = ['Store', 'Holiday_Flag', 'Year']
X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test, exclude_cols=exclude_cols)

print(f"\n✓ Preprocessing Complete!")
print(f"  Training set: {X_train_scaled.shape}")
print(f"  Test set: {X_test_scaled.shape}")

# 2. Model Training
print("\n[2/3] TRAINING MODELS...")
print("-" * 70)
trainer = WalmartModelTrainer(models_dir='models')
trainer.initialize_models()
trained_models = trainer.train_all_models(X_train_scaled, y_train)

# 3. Model Evaluation
print("\n[3/3] EVALUATING MODELS...")
print("-" * 70)
evaluator = WalmartModelEvaluator()
evaluator.evaluate_all_models(trained_models, X_train_scaled, y_train, X_test_scaled, y_test)

# Display results
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
results_df = evaluator.get_results_dataframe()
test_results = results_df[results_df.index.str.contains('Test')]
print("\nTest Set Performance:")
print(test_results.to_string())

# Find best model
best_model_name = test_results['R² Score'].idxmax().replace(' (Test)', '')
best_r2 = test_results['R² Score'].max()
best_mae = test_results.loc[f'{best_model_name} (Test)', 'MAE']
best_rmse = test_results.loc[f'{best_model_name} (Test)', 'RMSE']

print(f"\n{'=' * 70}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"{'=' * 70}")
print(f"  R² Score: {best_r2:.4f}")
print(f"  MAE: ${best_mae:,.2f}")
print(f"  RMSE: ${best_rmse:,.2f}")
print(f"{'=' * 70}")

# Save models
print("\n[4/4] SAVING MODELS...")
print("-" * 70)
trainer.save_all_models()

# Save scaler
import joblib
joblib.dump(preprocessor.scaler, 'models/scaler.joblib')
print("✓ Scaler saved: models/scaler.joblib")

# Save feature names
with open('models/feature_names.txt', 'w') as f:
    for feature in X_train_scaled.columns:
        f.write(f"{feature}\n")
print("✓ Feature names saved: models/feature_names.txt")

print("\n" + "=" * 70)
print("✅ PROJECT RUN COMPLETE!")
print("=" * 70)
print("\nNext steps:")
print("  1. Check models/ folder for trained models")
print("  2. Run: python predict.py --data Walmart.csv --output predictions.csv")
print("  3. Open Jupyter notebooks for detailed visualizations")
print("=" * 70)
