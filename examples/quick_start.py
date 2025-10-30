"""
Quick Start Example for Walmart Sales Prediction

This script demonstrates the basic usage of the project.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import WalmartDataPreprocessor
from src.model_training import WalmartModelTrainer
from src.visualization import WalmartVisualizer
from src.predict import WalmartPredictor
import pandas as pd
import numpy as np


def quick_start_example():
    """
    Quick start example demonstrating the complete workflow
    """
    print("="*70)
    print("WALMART SALES PREDICTION - QUICK START EXAMPLE")
    print("="*70)
    
    # Check if data file exists
    data_path = '../data/raw/Walmart.csv'
    if not os.path.exists(data_path):
        print("\n❌ Dataset not found!")
        print("Please download the dataset from Kaggle and place it in data/raw/")
        print("URL: https://www.kaggle.com/code/yasserh/walmart-sales-prediction-best-ml-algorithms/input")
        return
    
    print("\n✅ Dataset found!")
    
    # Step 1: Initialize components
    print("\n" + "="*70)
    print("STEP 1: Initializing Components")
    print("="*70)
    preprocessor = WalmartDataPreprocessor()
    trainer = WalmartModelTrainer()
    visualizer = WalmartVisualizer()
    print("✅ Components initialized")
    
    # Step 2: Load data
    print("\n" + "="*70)
    print("STEP 2: Loading Data")
    print("="*70)
    df = preprocessor.load_data(data_path)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 3: Preprocess
    print("\n" + "="*70)
    print("STEP 3: Preprocessing")
    print("="*70)
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.feature_engineering(df)
    
    # Identify target column (adjust based on your dataset)
    target_column = 'Weekly_Sales'
    if target_column not in df.columns:
        print(f"\n⚠️  Target column '{target_column}' not found in dataset")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df, target_column)
    print(f"✅ Data preprocessed and split")
    
    # Step 4: Train models (using a subset for quick demo)
    print("\n" + "="*70)
    print("STEP 4: Training Models (Top 3 for quick demo)")
    print("="*70)
    
    # Initialize with subset of models for quick demo
    trainer.initialize_models()
    trainer.models = {
        'Linear Regression': trainer.models['Linear Regression'],
        'Random Forest': trainer.models['Random Forest'],
        'XGBoost': trainer.models['XGBoost']
    }
    
    trainer.train_all_models(X_train, y_train, X_test, y_test)
    print("✅ Models trained")
    
    # Step 5: Results
    print("\n" + "="*70)
    print("STEP 5: Results")
    print("="*70)
    results_df = trainer.get_results_dataframe()
    print("\nModel Performance:")
    print(results_df.to_string(index=False))
    
    # Step 6: Make predictions
    print("\n" + "="*70)
    print("STEP 6: Sample Predictions")
    print("="*70)
    
    # Make predictions on test set
    best_model = trainer.best_model
    y_pred = best_model.predict(X_test[:5])
    
    print("\nFirst 5 predictions vs actual:")
    for i in range(5):
        print(f"  Sample {i+1}: Predicted={y_pred[i]:.2f}, Actual={y_test.iloc[i]:.2f}, "
              f"Error={abs(y_pred[i] - y_test.iloc[i]):.2f}")
    
    print("\n" + "="*70)
    print("✅ QUICK START COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run 'python main.py' for complete analysis with all models")
    print("  2. Explore notebooks in the notebooks/ directory")
    print("  3. Use src/predict.py for making new predictions")
    print("="*70)


if __name__ == "__main__":
    try:
        quick_start_example()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\nPlease ensure:")
        print("  1. Dataset is placed in data/raw/Walmart.csv")
        print("  2. All dependencies are installed: pip install -r requirements.txt")
        print("  3. Column names in the script match your dataset")
