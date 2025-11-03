"""
Data Preprocessing Module for Walmart Sales Prediction
This module handles data loading, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class WalmartDataPreprocessor:
    """
    A class to preprocess Walmart sales data including feature engineering
    and data transformation.
    """
    
    def __init__(self, filepath):
        """
        Initialize the preprocessor with the data file path.
        
        Parameters:
        -----------
        filepath : str
            Path to the Walmart CSV file
        """
        self.filepath = filepath
        self.df = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the data from CSV file."""
        self.df = pd.read_csv(self.filepath)
        print(f"Data loaded successfully! Shape: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """Print basic information about the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n=== Dataset Info ===")
        print(self.df.info())
        print("\n=== First Few Rows ===")
        print(self.df.head())
        print("\n=== Statistical Summary ===")
        print(self.df.describe())
        print("\n=== Missing Values ===")
        print(self.df.isnull().sum())
        print("\n=== Unique Stores ===")
        print(f"Number of unique stores: {self.df['Store'].nunique()}")
        
    def parse_dates(self):
        """Parse date strings and extract date features."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Convert Date column to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%m-%Y')
        
        # Extract date features
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Week'] = self.df['Date'].dt.isocalendar().week
        self.df['Day'] = self.df['Date'].dt.day
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
        self.df['Quarter'] = self.df['Date'].dt.quarter
        
        # Create cyclical features for month (seasonal patterns)
        self.df['Month_Sin'] = np.sin(2 * np.pi * self.df['Month'] / 12)
        self.df['Month_Cos'] = np.cos(2 * np.pi * self.df['Month'] / 12)
        
        print("Date features extracted successfully!")
        return self.df
    
    def create_lag_features(self, lags=[1, 2]):
        """
        Create lag features for Weekly_Sales.
        
        Parameters:
        -----------
        lags : list
            List of lag periods to create
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Sort by Store and Date
        self.df = self.df.sort_values(['Store', 'Date'])
        
        # Create lag features for each store
        for lag in lags:
            self.df[f'Sales_Lag_{lag}'] = self.df.groupby('Store')['Weekly_Sales'].shift(lag)
        
        print(f"Lag features created for lags: {lags}")
        return self.df
    
    def create_rolling_features(self, windows=[4]):
        """
        Create rolling statistics features.
        
        Parameters:
        -----------
        windows : list
            List of window sizes for rolling calculations
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Sort by Store and Date
        self.df = self.df.sort_values(['Store', 'Date'])
        
        # Create rolling mean for each store
        for window in windows:
            self.df[f'Sales_RollingMean_{window}'] = (
                self.df.groupby('Store')['Weekly_Sales']
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )
        
        print(f"Rolling features created for windows: {windows}")
        return self.df
    
    def handle_missing_values(self):
        """Handle missing values in the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Check for missing values
        missing_before = self.df.isnull().sum().sum()
        
        # For lag and rolling features, forward fill within each store
        lag_cols = [col for col in self.df.columns if 'Lag' in col or 'Rolling' in col]
        if lag_cols:
            for col in lag_cols:
                self.df[col] = self.df.groupby('Store')[col].fillna(method='ffill')
                # If still missing, fill with 0 (for initial periods)
                self.df[col] = self.df[col].fillna(0)
        
        missing_after = self.df.isnull().sum().sum()
        print(f"Missing values before: {missing_before}, after: {missing_after}")
        
        return self.df
    
    def prepare_features(self, drop_date=True):
        """
        Prepare final feature set for modeling.
        
        Parameters:
        -----------
        drop_date : bool
            Whether to drop the Date column
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Columns to drop
        drop_cols = ['Date'] if drop_date else []
        
        # Prepare features
        feature_df = self.df.drop(columns=drop_cols, errors='ignore')
        
        print(f"Final feature set shape: {feature_df.shape}")
        print(f"Features: {list(feature_df.columns)}")
        
        return feature_df
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Separate features and target
        X = self.df.drop(columns=['Weekly_Sales'])
        y = self.df['Weekly_Sales']
        
        # Drop any datetime columns that might still be present
        datetime_cols = X.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
        if datetime_cols:
            X = X.drop(columns=datetime_cols)
            print(f"Dropped datetime columns: {datetime_cols}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test, exclude_cols=['Store', 'Holiday_Flag', 'Year']):
        """
        Scale numerical features using StandardScaler.
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        X_test : DataFrame
            Testing features
        exclude_cols : list
            Columns to exclude from scaling
        """
        # Identify numerical columns only
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Identify columns to scale (numerical and not excluded)
        cols_to_scale = [col for col in numerical_cols if col not in exclude_cols]
        
        # Scale training data
        X_train_scaled = X_train.copy()
        X_train_scaled[cols_to_scale] = self.scaler.fit_transform(X_train[cols_to_scale])
        
        # Scale test data
        X_test_scaled = X_test.copy()
        X_test_scaled[cols_to_scale] = self.scaler.transform(X_test[cols_to_scale])
        
        print(f"Scaled {len(cols_to_scale)} features")
        
        return X_train_scaled, X_test_scaled
    
    def full_preprocessing_pipeline(self, create_lags=True, create_rolling=True, 
                                   scale=True, test_size=0.2):
        """
        Execute full preprocessing pipeline.
        
        Parameters:
        -----------
        create_lags : bool
            Whether to create lag features
        create_rolling : bool
            Whether to create rolling features
        scale : bool
            Whether to scale features
        test_size : float
            Proportion of data for testing
        """
        print("Starting full preprocessing pipeline...")
        
        # Load data
        self.load_data()
        
        # Parse dates and extract features
        self.parse_dates()
        
        # Create lag features
        if create_lags:
            self.create_lag_features()
        
        # Create rolling features
        if create_rolling:
            self.create_rolling_features()
        
        # Handle missing values
        self.handle_missing_values()
        
        # Prepare features
        self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(test_size=test_size)
        
        # Scale features
        if scale:
            X_train, X_test = self.scale_features(X_train, X_test)
        
        print("\nPreprocessing pipeline completed successfully!")
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    preprocessor = WalmartDataPreprocessor("../Walmart.csv")
    X_train, X_test, y_train, y_test = preprocessor.full_preprocessing_pipeline()
    print("\nPreprocessing complete!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
