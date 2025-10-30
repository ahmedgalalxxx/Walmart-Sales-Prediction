"""
Prediction Module for Walmart Sales Prediction
"""

import pickle
import pandas as pd
import numpy as np


class WalmartPredictor:
    """
    Make predictions using trained Walmart sales model
    """
    
    def __init__(self, model_path='models/best_model.pkl'):
        """
        Initialize predictor with saved model
        
        Args:
            model_path: Path to the saved model pickle file
        """
        self.model = None
        self.model_path = model_path
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.model_path}")
            print("Please train a model first using main.py")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict_single(self, features):
        """
        Make prediction for a single sample
        
        Args:
            features: Dictionary or array of features
            
        Returns:
            Predicted value
        """
        if self.model is None:
            print("No model loaded. Cannot make predictions.")
            return None
        
        try:
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            elif isinstance(features, list):
                features = np.array(features).reshape(1, -1)
            
            prediction = self.model.predict(features)
            return prediction[0]
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def predict_batch(self, features_df):
        """
        Make predictions for multiple samples
        
        Args:
            features_df: DataFrame or array of features
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            print("No model loaded. Cannot make predictions.")
            return None
        
        try:
            predictions = self.model.predict(features_df)
            return predictions
        except Exception as e:
            print(f"Error making batch predictions: {e}")
            return None
    
    def predict_from_csv(self, csv_path, output_path=None):
        """
        Make predictions from a CSV file
        
        Args:
            csv_path: Path to input CSV file
            output_path: Path to save predictions (optional)
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            print("No model loaded. Cannot make predictions.")
            return None
        
        try:
            # Load data
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} samples from {csv_path}")
            
            # Make predictions
            predictions = self.model.predict(df)
            
            # Add predictions to dataframe
            df['Predicted_Sales'] = predictions
            
            # Save if output path provided
            if output_path:
                df.to_csv(output_path, index=False)
                print(f"Predictions saved to {output_path}")
            
            return df
        except Exception as e:
            print(f"Error predicting from CSV: {e}")
            return None


def main():
    """
    Example usage of WalmartPredictor
    """
    print("="*60)
    print("WALMART SALES PREDICTOR")
    print("="*60)
    
    # Initialize predictor
    predictor = WalmartPredictor()
    
    if predictor.model is not None:
        print("\nPredictor ready to make predictions!")
        print("\nUsage examples:")
        print("1. Single prediction: predictor.predict_single(features)")
        print("2. Batch prediction: predictor.predict_batch(features_df)")
        print("3. CSV prediction: predictor.predict_from_csv('input.csv', 'output.csv')")
    else:
        print("\nPlease train a model first by running: python main.py")


if __name__ == "__main__":
    main()
