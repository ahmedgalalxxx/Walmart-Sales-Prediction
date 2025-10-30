"""
Visualization Module for Walmart Sales Prediction
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class WalmartVisualizer:
    """
    Visualization utilities for Walmart sales data and model results
    """
    
    def __init__(self, save_dir='figures'):
        self.save_dir = save_dir
        
    def plot_data_distribution(self, df, column, title=None, save_name=None):
        """Plot distribution of a column"""
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True, bins=30)
        plt.title(title or f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        
        if save_name:
            plt.savefig(f'{self.save_dir}/{save_name}', bbox_inches='tight', dpi=300)
        plt.show()
        
    def plot_correlation_matrix(self, df, save_name=None):
        """Plot correlation matrix heatmap"""
        plt.figure(figsize=(14, 10))
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        correlation = numeric_df.corr()
        
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Matrix')
        
        if save_name:
            plt.savefig(f'{self.save_dir}/{save_name}', bbox_inches='tight', dpi=300)
        plt.show()
        
    def plot_feature_importance(self, model, feature_names, top_n=15, save_name=None):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(top_n), importances[indices])
            plt.yticks(range(top_n), [feature_names[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Important Features')
            plt.gca().invert_yaxis()
            
            if save_name:
                plt.savefig(f'{self.save_dir}/{save_name}', bbox_inches='tight', dpi=300)
            plt.show()
        else:
            print("Model does not have feature_importances_ attribute")
    
    def plot_predictions_vs_actual(self, y_true, y_pred, title='Predictions vs Actual', save_name=None):
        """Plot predicted vs actual values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Plot perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.legend()
        
        if save_name:
            plt.savefig(f'{self.save_dir}/{save_name}', bbox_inches='tight', dpi=300)
        plt.show()
        
    def plot_residuals(self, y_true, y_pred, save_name=None):
        """Plot residuals"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residual plot
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residual Plot')
        
        # Residual distribution
        axes[1].hist(residuals, bins=30, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f'{self.save_dir}/{save_name}', bbox_inches='tight', dpi=300)
        plt.show()
        
    def plot_model_comparison(self, results_df, save_name=None):
        """Plot comparison of different models"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = ['R2 Score', 'RMSE', 'MAE', 'MSE']
        
        for idx, metric in enumerate(metrics):
            row = idx // 2
            col = idx % 2
            
            ax = axes[row, col]
            
            sorted_df = results_df.sort_values(metric, ascending=(metric != 'R2 Score'))
            
            ax.barh(sorted_df['Model'], sorted_df[metric])
            ax.set_xlabel(metric)
            ax.set_title(f'Model Comparison - {metric}')
            ax.invert_yaxis()
            
            # Add values on bars
            for i, v in enumerate(sorted_df[metric]):
                ax.text(v, i, f' {v:.4f}', va='center')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f'{self.save_dir}/{save_name}', bbox_inches='tight', dpi=300)
        plt.show()
        
    def plot_time_series(self, df, date_column, value_column, save_name=None):
        """Plot time series data"""
        if date_column in df.columns and value_column in df.columns:
            plt.figure(figsize=(14, 6))
            
            df_sorted = df.sort_values(date_column)
            plt.plot(df_sorted[date_column], df_sorted[value_column])
            plt.xlabel(date_column)
            plt.ylabel(value_column)
            plt.title(f'{value_column} Over Time')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            if save_name:
                plt.savefig(f'{self.save_dir}/{save_name}', bbox_inches='tight', dpi=300)
            plt.show()
        else:
            print(f"Columns {date_column} or {value_column} not found")
    
    def plot_boxplot(self, df, column, group_by=None, save_name=None):
        """Plot boxplot for outlier detection"""
        plt.figure(figsize=(10, 6))
        
        if group_by and group_by in df.columns:
            df.boxplot(column=column, by=group_by, figsize=(12, 6))
            plt.suptitle('')
            plt.title(f'{column} Distribution by {group_by}')
        else:
            sns.boxplot(y=df[column])
            plt.title(f'{column} Distribution')
        
        if save_name:
            plt.savefig(f'{self.save_dir}/{save_name}', bbox_inches='tight', dpi=300)
        plt.show()


if __name__ == "__main__":
    # Example usage
    visualizer = WalmartVisualizer()
    print("Walmart Visualizer initialized")
