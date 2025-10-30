"""
Model Training Module for Walmart Sales Prediction
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class WalmartModelTrainer:
    """
    Train multiple ML models for Walmart sales prediction
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize all ML models"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'SVR': SVR(kernel='rbf')
        }
        print(f"Initialized {len(self.models)} models")
        
    def calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r2
        }
    
    def train_single_model(self, model_name, model, X_train, y_train, X_test, y_test):
        """Train a single model and evaluate"""
        print(f"\nTraining {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        # Store results
        self.results[model_name] = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        
        print(f"{model_name} - Test R2 Score: {test_metrics['R2 Score']:.4f}, Test RMSE: {test_metrics['RMSE']:.4f}")
        
        return model
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all models and compare"""
        if not self.models:
            self.initialize_models()
        
        print("\n" + "="*60)
        print("Training All Models")
        print("="*60)
        
        for model_name, model in self.models.items():
            try:
                self.train_single_model(model_name, model, X_train, y_train, X_test, y_test)
            except Exception as e:
                print(f"Error training {model_name}: {e}")
        
        # Find best model
        self.find_best_model()
        
    def find_best_model(self):
        """Find the best performing model based on test R2 score"""
        best_score = -float('inf')
        
        for model_name, result in self.results.items():
            test_r2 = result['test_metrics']['R2 Score']
            if test_r2 > best_score:
                best_score = test_r2
                self.best_model_name = model_name
                self.best_model = result['model']
        
        print("\n" + "="*60)
        print(f"Best Model: {self.best_model_name}")
        print(f"Best Test R2 Score: {best_score:.4f}")
        print("="*60)
    
    def get_results_dataframe(self):
        """Get results as a pandas DataFrame"""
        results_list = []
        
        for model_name, result in self.results.items():
            test_metrics = result['test_metrics']
            results_list.append({
                'Model': model_name,
                'R2 Score': test_metrics['R2 Score'],
                'RMSE': test_metrics['RMSE'],
                'MAE': test_metrics['MAE'],
                'MSE': test_metrics['MSE']
            })
        
        df = pd.DataFrame(results_list)
        df = df.sort_values('R2 Score', ascending=False).reset_index(drop=True)
        
        return df
    
    def save_model(self, model, filepath):
        """Save trained model to disk"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath):
        """Load trained model from disk"""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def save_best_model(self, filepath='models/best_model.pkl'):
        """Save the best performing model"""
        if self.best_model is not None:
            self.save_model(self.best_model, filepath)
        else:
            print("No best model found. Train models first.")


if __name__ == "__main__":
    # Example usage
    trainer = WalmartModelTrainer()
    trainer.initialize_models()
    print(f"Available models: {list(trainer.models.keys())}")
