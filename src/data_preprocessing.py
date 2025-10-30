"""
Data Preprocessing Module for Walmart Sales Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class WalmartDataPreprocessor:
    """
    Preprocessor for Walmart sales data
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load the Walmart sales dataset"""
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("\nMissing values before handling:")
        print(df.isnull().sum())
        
        # Fill numerical columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        print("\nMissing values after handling:")
        print(df.isnull().sum())
        
        return df
    
    def encode_categorical_features(self, df, categorical_columns):
        """Encode categorical features using Label Encoding"""
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded column: {col}")
        
        return df_encoded
    
    def feature_engineering(self, df):
        """Create additional features from existing ones"""
        df_engineered = df.copy()
        
        # If Date column exists, extract date features
        if 'Date' in df_engineered.columns:
            df_engineered['Date'] = pd.to_datetime(df_engineered['Date'])
            df_engineered['Year'] = df_engineered['Date'].dt.year
            df_engineered['Month'] = df_engineered['Date'].dt.month
            df_engineered['Day'] = df_engineered['Date'].dt.day
            df_engineered['WeekOfYear'] = df_engineered['Date'].dt.isocalendar().week
            df_engineered['DayOfWeek'] = df_engineered['Date'].dt.dayofweek
            df_engineered.drop('Date', axis=1, inplace=True)
            print("Date features extracted")
        
        return df_engineered
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def prepare_data(self, df, target_column, test_size=0.2, random_state=42):
        """Prepare data for modeling"""
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nTraining set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def full_preprocessing_pipeline(self, filepath, target_column, categorical_columns=None):
        """Complete preprocessing pipeline"""
        # Load data
        df = self.load_data(filepath)
        if df is None:
            return None, None, None, None
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Encode categorical features
        if categorical_columns:
            df = self.encode_categorical_features(df, categorical_columns)
        
        # Prepare train-test split
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_column)
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    preprocessor = WalmartDataPreprocessor()
    print("Walmart Data Preprocessor initialized")
