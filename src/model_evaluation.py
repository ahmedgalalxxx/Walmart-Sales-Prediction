"""
Model Evaluation Module for Walmart Sales Prediction
This module contains functions for evaluating model performance and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_squared_error,
    mean_absolute_percentage_error
)


class WalmartModelEvaluator:
    """
    A class to evaluate and visualize model performance.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.results = {}
        
    def calculate_metrics(self, y_true, y_pred, model_name):
        """
        Calculate regression metrics for a model.
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        model_name : str
            Name of the model
        """
        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        # Store results
        self.results[model_name] = {
            'R² Score': r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE (%)': mape
        }
        
        return self.results[model_name]
    
    def evaluate_model(self, model_name, y_train_true, y_train_pred, 
                      y_test_true, y_test_pred, verbose=True):
        """
        Evaluate model on both training and test sets.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        y_train_true : array-like
            True training values
        y_train_pred : array-like
            Predicted training values
        y_test_true : array-like
            True test values
        y_test_pred : array-like
            Predicted test values
        verbose : bool
            Whether to print results
        """
        # Calculate metrics for training set
        train_metrics = self.calculate_metrics(y_train_true, y_train_pred, 
                                               f"{model_name} (Train)")
        
        # Calculate metrics for test set
        test_metrics = self.calculate_metrics(y_test_true, y_test_pred, 
                                              f"{model_name} (Test)")
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"{model_name} Performance")
            print(f"{'=' * 60}")
            print("\nTraining Set:")
            for metric, value in train_metrics.items():
                print(f"  {metric}: {value:,.4f}")
            
            print("\nTest Set:")
            for metric, value in test_metrics.items():
                print(f"  {metric}: {value:,.4f}")
            print(f"{'=' * 60}")
        
        return train_metrics, test_metrics
    
    def evaluate_all_models(self, models_dict, X_train, y_train, X_test, y_test):
        """
        Evaluate all models and compare their performance.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of trained models
        X_train, y_train : Training data
        X_test, y_test : Test data
        """
        print("\n" + "=" * 70)
        print("EVALUATING ALL MODELS")
        print("=" * 70)
        
        for model_name, model in models_dict.items():
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Evaluate
            self.evaluate_model(model_name, y_train, y_train_pred, 
                              y_test, y_test_pred)
        
        print("\nAll models evaluated successfully!")
    
    def get_results_dataframe(self):
        """
        Get evaluation results as a DataFrame.
        """
        if not self.results:
            print("No results available. Run evaluate_model first.")
            return None
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        
        return results_df
    
    def plot_comparison(self, metric='R² Score', figsize=(12, 6)):
        """
        Plot comparison of models based on a specific metric.
        
        Parameters:
        -----------
        metric : str
            Metric to compare (e.g., 'R² Score', 'RMSE', 'MAE')
        figsize : tuple
            Figure size
        """
        if not self.results:
            print("No results available. Run evaluate_model first.")
            return
        
        # Get results DataFrame
        df = self.get_results_dataframe()
        
        # Filter for test set results only
        test_df = df[df.index.str.contains('Test')]
        
        # Remove '(Test)' from index for cleaner labels
        test_df.index = test_df.index.str.replace(' (Test)', '')
        
        # Sort by metric
        test_df = test_df.sort_values(metric, ascending=(metric != 'R² Score'))
        
        # Create plot
        plt.figure(figsize=figsize)
        colors = sns.color_palette("husl", len(test_df))
        
        bars = plt.bar(range(len(test_df)), test_df[metric], color=colors)
        plt.xlabel('Model', fontsize=12, fontweight='bold')
        plt.ylabel(metric, fontsize=12, fontweight='bold')
        plt.title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
        plt.xticks(range(len(test_df)), test_df.index, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name, 
                                   figsize=(12, 5)):
        """
        Plot predicted vs actual values.
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        model_name : str
            Name of the model
        figsize : tuple
            Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidths=0.5)
        axes[0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Sales', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Predicted Sales', fontsize=11, fontweight='bold')
        axes[0].set_title(f'{model_name}: Predicted vs Actual', 
                         fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Sales', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Residuals', fontsize=11, fontweight='bold')
        axes[1].set_title(f'{model_name}: Residual Plot', 
                         fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_error_distribution(self, y_true, y_pred, model_name, 
                               figsize=(12, 5)):
        """
        Plot error distribution for a model.
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        model_name : str
            Name of the model
        figsize : tuple
            Figure size
        """
        errors = y_true - y_pred
        percentage_errors = (errors / y_true) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram of errors
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Error (Actual - Predicted)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title(f'{model_name}: Error Distribution', 
                         fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of percentage errors
        axes[1].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Percentage Error (%)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1].set_title(f'{model_name}: Percentage Error Distribution', 
                         fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, importance_df, top_n=15, figsize=(10, 8)):
        """
        Plot feature importance.
        
        Parameters:
        -----------
        importance_df : DataFrame
            DataFrame with 'Feature' and 'Importance' columns
        top_n : int
            Number of top features to plot
        figsize : tuple
            Figure size
        """
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=figsize)
        colors = sns.color_palette("viridis", len(top_features))
        
        plt.barh(range(len(top_features)), top_features['Importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def create_evaluation_report(self, save_path=None):
        """
        Create a comprehensive evaluation report.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the report
        """
        if not self.results:
            print("No results available. Run evaluate_model first.")
            return None
        
        results_df = self.get_results_dataframe()
        
        # Create report
        report = []
        report.append("=" * 70)
        report.append("WALMART SALES PREDICTION - MODEL EVALUATION REPORT")
        report.append("=" * 70)
        report.append("\n")
        report.append(results_df.to_string())
        report.append("\n\n")
        
        # Best model per metric
        report.append("=" * 70)
        report.append("BEST MODELS PER METRIC (Test Set)")
        report.append("=" * 70)
        
        test_df = results_df[results_df.index.str.contains('Test')]
        
        for metric in test_df.columns:
            if metric == 'R² Score':
                best_model = test_df[metric].idxmax()
                best_value = test_df[metric].max()
            else:
                best_model = test_df[metric].idxmin()
                best_value = test_df[metric].min()
            
            report.append(f"\n{metric}: {best_model}")
            report.append(f"  Value: {best_value:,.4f}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {save_path}")
        
        return report_text


if __name__ == "__main__":
    print("Model Evaluation Module - Ready to use!")
    print("Import this module in your notebook or script to evaluate models.")
