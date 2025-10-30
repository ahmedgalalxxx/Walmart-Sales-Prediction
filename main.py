"""
Main script for Walmart Sales Prediction
"""

import os
import sys
import pandas as pd
import numpy as np
from src.data_preprocessing import WalmartDataPreprocessor
from src.model_training import WalmartModelTrainer
from src.visualization import WalmartVisualizer


def main():
    """
    Main execution function for Walmart Sales Prediction
    """
    print("="*70)
    print("WALMART SALES PREDICTION - MACHINE LEARNING PROJECT")
    print("="*70)
    
    # Configuration
    DATA_PATH = 'data/raw/Walmart.csv'
    TARGET_COLUMN = 'Weekly_Sales'  # Adjust based on actual dataset
    CATEGORICAL_COLUMNS = ['Store', 'Holiday_Flag']  # Adjust based on actual dataset
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"\nError: Data file not found at {DATA_PATH}")
        print("Please download the dataset from Kaggle and place it in data/raw/")
        print("Dataset URL: https://www.kaggle.com/code/yasserh/walmart-sales-prediction-best-ml-algorithms/input")
        return
    
    # Initialize components
    preprocessor = WalmartDataPreprocessor()
    trainer = WalmartModelTrainer()
    visualizer = WalmartVisualizer()
    
    # Step 1: Load and preprocess data
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    df = preprocessor.load_data(DATA_PATH)
    if df is None:
        return
    
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df)
    
    # Feature engineering
    df = preprocessor.feature_engineering(df)
    
    # Encode categorical features
    if CATEGORICAL_COLUMNS:
        # Only encode columns that exist in the dataframe
        existing_categorical = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
        if existing_categorical:
            df = preprocessor.encode_categorical_features(df, existing_categorical)
    
    # Prepare train-test split
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df, TARGET_COLUMN)
    
    # Step 2: Train models
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)
    
    trainer.initialize_models()
    trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Get results
    results_df = trainer.get_results_dataframe()
    print("\n" + "="*70)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/model_comparison.csv', index=False)
    print("\nResults saved to results/model_comparison.csv")
    
    # Save best model
    os.makedirs('models', exist_ok=True)
    trainer.save_best_model('models/best_model.pkl')
    
    # Step 3: Visualizations
    print("\n" + "="*70)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("="*70)
    
    os.makedirs('figures', exist_ok=True)
    
    # Plot model comparison
    print("\nCreating model comparison plot...")
    visualizer.plot_model_comparison(results_df, save_name='model_comparison.png')
    
    # Plot predictions vs actual for best model
    print("\nCreating predictions vs actual plot...")
    best_model = trainer.best_model
    y_pred = best_model.predict(X_test)
    visualizer.plot_predictions_vs_actual(y_test, y_pred, 
                                         title=f'{trainer.best_model_name} - Predictions vs Actual',
                                         save_name='predictions_vs_actual.png')
    
    # Plot residuals
    print("\nCreating residual plots...")
    visualizer.plot_residuals(y_test, y_pred, save_name='residuals.png')
    
    # Plot feature importance (if applicable)
    if hasattr(best_model, 'feature_importances_'):
        print("\nCreating feature importance plot...")
        feature_names = X_train.columns.tolist()
        visualizer.plot_feature_importance(best_model, feature_names, 
                                          save_name='feature_importance.png')
    
    print("\n" + "="*70)
    print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nBest Model: {trainer.best_model_name}")
    print(f"Test R2 Score: {trainer.results[trainer.best_model_name]['test_metrics']['R2 Score']:.4f}")
    print(f"Test RMSE: {trainer.results[trainer.best_model_name]['test_metrics']['RMSE']:.4f}")
    print("\nOutput files:")
    print("  - models/best_model.pkl")
    print("  - results/model_comparison.csv")
    print("  - figures/*.png")


if __name__ == "__main__":
    main()
