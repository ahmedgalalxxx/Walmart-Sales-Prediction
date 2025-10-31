"""
Prediction Script for Walmart Sales Prediction
This script loads trained models and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import WalmartDataPreprocessor


class WalmartSalesPredictor:
    """
    A class to make sales predictions using trained models.
    """
    
    def __init__(self, model_path, scaler_path, feature_names_path):
        """
        Initialize the predictor with paths to saved model artifacts.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model file
        scaler_path : str
            Path to the saved scaler
        feature_names_path : str
            Path to the feature names file
        """
        self.model = self.load_model(model_path)
        self.scaler = self.load_scaler(scaler_path)
        self.feature_names = self.load_feature_names(feature_names_path)
        
    def load_model(self, path):
        """Load the trained model."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model = joblib.load(path)
        print(f"✓ Model loaded from: {path}")
        return model
    
    def load_scaler(self, path):
        """Load the fitted scaler."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found: {path}")
        
        scaler = joblib.load(path)
        print(f"✓ Scaler loaded from: {path}")
        return scaler
    
    def load_feature_names(self, path):
        """Load feature names."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature names file not found: {path}")
        
        with open(path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        print(f"✓ Feature names loaded: {len(feature_names)} features")
        return feature_names
    
    def preprocess_data(self, data_path):
        """
        Preprocess new data for prediction.
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file with new data
        """
        print("\nPreprocessing data...")
        
        # Initialize preprocessor
        preprocessor = WalmartDataPreprocessor(data_path)
        
        # Load and preprocess
        preprocessor.load_data()
        preprocessor.parse_dates()
        preprocessor.create_lag_features()
        preprocessor.create_rolling_features()
        preprocessor.handle_missing_values()
        df = preprocessor.prepare_features()
        
        # Separate features and target (if exists)
        if 'Weekly_Sales' in df.columns:
            y = df['Weekly_Sales']
            X = df.drop(columns=['Weekly_Sales'])
        else:
            y = None
            X = df
        
        print(f"✓ Data preprocessed: {X.shape}")
        
        return X, y, preprocessor
    
    def scale_features(self, X, exclude_cols=['Store', 'Holiday_Flag', 'Year']):
        """
        Scale features using the loaded scaler.
        
        Parameters:
        -----------
        X : DataFrame
            Features to scale
        exclude_cols : list
            Columns to exclude from scaling
        """
        X_scaled = X.copy()
        
        # Identify columns to scale
        cols_to_scale = [col for col in X.columns if col not in exclude_cols]
        
        # Scale
        X_scaled[cols_to_scale] = self.scaler.transform(X[cols_to_scale])
        
        print(f"✓ Features scaled: {len(cols_to_scale)} columns")
        
        return X_scaled
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : DataFrame
            Features for prediction
        """
        # Ensure feature order matches training
        X_ordered = X[self.feature_names]
        
        # Make predictions
        predictions = self.model.predict(X_ordered)
        
        print(f"✓ Predictions made: {len(predictions)} samples")
        
        return predictions
    
    def predict_from_file(self, data_path, output_path=None):
        """
        Make predictions from a CSV file and optionally save results.
        
        Parameters:
        -----------
        data_path : str
            Path to the input CSV file
        output_path : str, optional
            Path to save predictions
        """
        print("=" * 70)
        print("WALMART SALES PREDICTION")
        print("=" * 70)
        
        # Preprocess data
        X, y_true, preprocessor = self.preprocess_data(data_path)
        
        # Scale features
        X_scaled = self.scale_features(X)
        
        # Make predictions
        predictions = self.predict(X_scaled)
        
        # Create results DataFrame
        results = X.copy()
        results['Predicted_Weekly_Sales'] = predictions
        
        if y_true is not None:
            results['Actual_Weekly_Sales'] = y_true.values
            results['Error'] = y_true.values - predictions
            results['Abs_Error'] = np.abs(results['Error'])
            results['Percentage_Error'] = (results['Error'] / y_true.values) * 100
            
            # Calculate metrics
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            
            r2 = r2_score(y_true, predictions)
            mae = mean_absolute_error(y_true, predictions)
            rmse = np.sqrt(mean_squared_error(y_true, predictions))
            mape = np.mean(np.abs((y_true.values - predictions) / y_true.values)) * 100
            
            print("\n" + "=" * 70)
            print("PREDICTION RESULTS")
            print("=" * 70)
            print(f"R² Score: {r2:.4f}")
            print(f"MAE: ${mae:,.2f}")
            print(f"RMSE: ${rmse:,.2f}")
            print(f"MAPE: {mape:.2f}%")
            print("=" * 70)
        
        # Save results if output path provided
        if output_path:
            results.to_csv(output_path, index=False)
            print(f"\n✓ Results saved to: {output_path}")
        
        return results
    
    def predict_single(self, store, date, holiday_flag, temperature, 
                      fuel_price, cpi, unemployment, **kwargs):
        """
        Make a prediction for a single observation.
        
        Parameters:
        -----------
        store : int
            Store number
        date : str
            Date in format 'DD-MM-YYYY'
        holiday_flag : int
            1 if holiday week, 0 otherwise
        temperature : float
            Temperature in Fahrenheit
        fuel_price : float
            Fuel price per gallon
        cpi : float
            Consumer Price Index
        unemployment : float
            Unemployment rate
        **kwargs : Additional features if available (lag features, rolling features, etc.)
        """
        # Create DataFrame with input
        data = {
            'Store': [store],
            'Date': [date],
            'Holiday_Flag': [holiday_flag],
            'Temperature': [temperature],
            'Fuel_Price': [fuel_price],
            'CPI': [cpi],
            'Unemployment': [unemployment]
        }
        
        # Add any additional features
        data.update(kwargs)
        
        df = pd.DataFrame(data)
        
        # Parse date
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        
        # Extract date features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Drop date
        df = df.drop(columns=['Date'])
        
        # Ensure all required features are present (fill missing with 0)
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select and order features
        df = df[self.feature_names]
        
        # Scale features
        df_scaled = self.scale_features(df)
        
        # Make prediction
        prediction = self.model.predict(df_scaled)[0]
        
        print(f"\nPredicted Weekly Sales: ${prediction:,.2f}")
        
        return prediction


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Walmart Sales Prediction')
    parser.add_argument('--data', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--model', type=str, default='models/xgboost_model.joblib', 
                       help='Path to trained model')
    parser.add_argument('--scaler', type=str, default='models/scaler.joblib',
                       help='Path to scaler')
    parser.add_argument('--features', type=str, default='models/feature_names.txt',
                       help='Path to feature names file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = WalmartSalesPredictor(args.model, args.scaler, args.features)
    
    # Make predictions
    results = predictor.predict_from_file(args.data, args.output)
    
    # Display sample results
    print("\nSample Predictions:")
    print(results[['Store', 'Predicted_Weekly_Sales']].head(10))


if __name__ == "__main__":
    # Example usage
    print("Walmart Sales Prediction Script")
    print("\nUsage:")
    print("  python predict.py --data path/to/data.csv --output path/to/output.csv")
    print("\nOr import this module to use WalmartSalesPredictor class programmatically.")
