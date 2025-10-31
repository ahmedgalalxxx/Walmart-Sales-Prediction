"""
Model Training Module for Walmart Sales Prediction
This module contains functions for training and saving machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import joblib
import os
from datetime import datetime


class WalmartModelTrainer:
    """
    A class to train and manage different regression models for sales prediction.
    """
    
    def __init__(self, models_dir='../models'):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        models_dir : str
            Directory to save trained models
        """
        self.models_dir = models_dir
        self.models = {}
        self.trained_models = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
    def initialize_models(self):
        """Initialize different regression models."""
        self.models = {
            'Linear Regression': LinearRegression(),
            
            'Decision Tree': DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            ),
            
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            
            'XGBoost': XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        }
        
        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
        
        return self.models
    
    def train_model(self, model_name, X_train, y_train, verbose=True):
        """
        Train a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to train
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        verbose : bool
            Whether to print training progress
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        if verbose:
            print(f"\nTraining {model_name}...")
        
        start_time = datetime.now()
        
        # Train the model
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Store trained model
        self.trained_models[model_name] = model
        
        if verbose:
            print(f"{model_name} trained successfully in {training_time:.2f} seconds!")
        
        return model
    
    def train_all_models(self, X_train, y_train):
        """
        Train all initialized models.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        """
        if not self.models:
            self.initialize_models()
        
        print("=" * 60)
        print("Training all models...")
        print("=" * 60)
        
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train)
        
        print("\n" + "=" * 60)
        print(f"All {len(self.models)} models trained successfully!")
        print("=" * 60)
        
        return self.trained_models
    
    def get_feature_importance(self, model_name, feature_names, top_n=10):
        """
        Get feature importance for tree-based models.
        
        Parameters:
        -----------
        model_name : str
            Name of the trained model
        feature_names : list
            List of feature names
        top_n : int
            Number of top features to return
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained yet.")
        
        model = self.trained_models[model_name]
        
        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            print(f"{model_name} doesn't support feature importance.")
            return None
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop {top_n} Features for {model_name}:")
        print(feature_importance_df.head(top_n).to_string(index=False))
        
        return feature_importance_df
    
    def save_model(self, model_name, filename=None):
        """
        Save a trained model to disk.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to save
        filename : str, optional
            Custom filename for the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained yet.")
        
        if filename is None:
            # Create filename from model name
            filename = f"{model_name.lower().replace(' ', '_')}_model.joblib"
        
        filepath = os.path.join(self.models_dir, filename)
        
        # Save the model
        joblib.dump(self.trained_models[model_name], filepath)
        print(f"Model saved: {filepath}")
        
        return filepath
    
    def save_all_models(self):
        """Save all trained models to disk."""
        print("\nSaving all trained models...")
        
        saved_paths = {}
        for model_name in self.trained_models.keys():
            filepath = self.save_model(model_name)
            saved_paths[model_name] = filepath
        
        print(f"\nAll {len(self.trained_models)} models saved successfully!")
        return saved_paths
    
    def load_model(self, model_name, filename=None):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        model_name : str
            Name to assign to the loaded model
        filename : str
            Path to the model file
        """
        if filename is None:
            filename = f"{model_name.lower().replace(' ', '_')}_model.joblib"
        
        filepath = os.path.join(self.models_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load the model
        model = joblib.load(filepath)
        self.trained_models[model_name] = model
        
        print(f"Model loaded: {filepath}")
        return model
    
    def predict(self, model_name, X):
        """
        Make predictions using a trained model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to use
        X : array-like
            Features to predict on
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained yet.")
        
        model = self.trained_models[model_name]
        predictions = model.predict(X)
        
        return predictions
    
    def predict_all_models(self, X):
        """
        Make predictions using all trained models.
        
        Parameters:
        -----------
        X : array-like
            Features to predict on
        """
        predictions = {}
        
        for model_name in self.trained_models.keys():
            predictions[model_name] = self.predict(model_name, X)
        
        return predictions


def hyperparameter_tuning_rf(X_train, y_train, X_test, y_test):
    """
    Perform hyperparameter tuning for Random Forest using GridSearchCV.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    """
    from sklearn.model_selection import GridSearchCV
    
    print("Performing hyperparameter tuning for Random Forest...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10]
    }
    
    # Initialize model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='r2',
        verbose=2,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    test_score = grid_search.score(X_test, y_test)
    print(f"Test set score: {test_score:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


if __name__ == "__main__":
    print("Model Training Module - Ready to use!")
    print("Import this module in your notebook or script to train models.")
