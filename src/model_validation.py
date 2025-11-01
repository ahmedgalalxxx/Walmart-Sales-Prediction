"""
Model Validation Module for Walmart Sales Prediction
This module provides comprehensive validation for trained models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_squared_error,
    mean_absolute_percentage_error
)
import joblib
import sys
sys.path.append('src')


class ModelValidator:
    """
    Comprehensive validation for Walmart sales prediction models.
    """
    
    def __init__(self, model_path=None, model=None):
        """
        Initialize validator with either a model path or model object.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to saved model
        model : object, optional
            Trained model object
        """
        if model_path:
            self.model = joblib.load(model_path)
            print(f"✓ Model loaded from: {model_path}")
        elif model:
            self.model = model
            print("✓ Model object provided")
        else:
            raise ValueError("Either model_path or model must be provided")
        
        self.validation_results = {}
    
    def validate_data_quality(self, X, y):
        """
        Validate data quality before making predictions.
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : Series
            Target variable
        """
        print("\n" + "="*70)
        print("DATA QUALITY VALIDATION")
        print("="*70)
        
        issues = []
        
        # Check for missing values
        missing_X = X.isnull().sum().sum()
        missing_y = y.isnull().sum()
        
        if missing_X > 0:
            issues.append(f"✗ Features have {missing_X} missing values")
        else:
            print("✓ No missing values in features")
        
        if missing_y > 0:
            issues.append(f"✗ Target has {missing_y} missing values")
        else:
            print("✓ No missing values in target")
        
        # Check for infinite values
        inf_X = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        inf_y = np.isinf(y).sum() if isinstance(y, pd.Series) else np.isinf(y).sum()
        
        if inf_X > 0:
            issues.append(f"✗ Features have {inf_X} infinite values")
        else:
            print("✓ No infinite values in features")
        
        if inf_y > 0:
            issues.append(f"✗ Target has {inf_y} infinite values")
        else:
            print("✓ No infinite values in target")
        
        # Check data types
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            issues.append(f"✗ Non-numeric columns found: {non_numeric}")
        else:
            print("✓ All features are numeric")
        
        # Check for duplicates
        duplicates = X.duplicated().sum()
        if duplicates > 0:
            print(f"⚠ Warning: {duplicates} duplicate rows found")
        else:
            print("✓ No duplicate rows")
        
        # Check target variable distribution
        if y.std() == 0:
            issues.append("✗ Target variable has zero variance")
        else:
            print(f"✓ Target variance: {y.var():.2f}")
        
        # Check for outliers using IQR
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((y < (Q1 - 3 * IQR)) | (y > (Q3 + 3 * IQR))).sum()
        outlier_pct = (outliers / len(y)) * 100
        print(f"⚠ Outliers in target: {outliers} ({outlier_pct:.2f}%)")
        
        print("="*70)
        
        if issues:
            print("\n⚠ DATA QUALITY ISSUES FOUND:")
            for issue in issues:
                print(f"  {issue}")
            return False
        else:
            print("\n✅ DATA QUALITY: PASSED")
            return True
    
    def cross_validate(self, X, y, cv=5, scoring='r2'):
        """
        Perform cross-validation on the model.
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : Series
            Target variable
        cv : int
            Number of cross-validation folds
        scoring : str
            Scoring metric
        """
        print("\n" + "="*70)
        print(f"CROSS-VALIDATION ({cv}-Fold)")
        print("="*70)
        
        # Standard K-Fold Cross-Validation
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        print(f"\nFold Scores ({scoring}):")
        for i, score in enumerate(scores, 1):
            print(f"  Fold {i}: {score:.4f}")
        
        print(f"\nCross-Validation Results:")
        print(f"  Mean Score: {scores.mean():.4f}")
        print(f"  Std Dev: {scores.std():.4f}")
        print(f"  Min Score: {scores.min():.4f}")
        print(f"  Max Score: {scores.max():.4f}")
        
        self.validation_results['cv_scores'] = scores
        self.validation_results['cv_mean'] = scores.mean()
        self.validation_results['cv_std'] = scores.std()
        
        # Visualize
        plt.figure(figsize=(10, 5))
        plt.bar(range(1, cv+1), scores, alpha=0.7, color='steelblue', edgecolor='black')
        plt.axhline(y=scores.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.4f}')
        plt.xlabel('Fold', fontweight='bold')
        plt.ylabel(f'{scoring.upper()} Score', fontweight='bold')
        plt.title(f'{cv}-Fold Cross-Validation Results', fontweight='bold', fontsize=14)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("="*70)
        
        return scores
    
    def time_series_validation(self, X, y, n_splits=5):
        """
        Perform time series cross-validation.
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : Series
            Target variable
        n_splits : int
            Number of splits
        """
        print("\n" + "="*70)
        print(f"TIME SERIES CROSS-VALIDATION ({n_splits} Splits)")
        print("="*70)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        print("\nSplit Results:")
        for i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train and evaluate
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            score = r2_score(y_test, y_pred)
            scores.append(score)
            
            print(f"  Split {i}: R² = {score:.4f} (Train: {len(train_idx)}, Test: {len(test_idx)})")
        
        scores = np.array(scores)
        
        print(f"\nTime Series CV Results:")
        print(f"  Mean R²: {scores.mean():.4f}")
        print(f"  Std Dev: {scores.std():.4f}")
        print(f"  Min R²: {scores.min():.4f}")
        print(f"  Max R²: {scores.max():.4f}")
        
        self.validation_results['ts_scores'] = scores
        self.validation_results['ts_mean'] = scores.mean()
        
        # Visualize
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, n_splits+1), scores, marker='o', linewidth=2, markersize=8, color='darkgreen')
        plt.axhline(y=scores.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.4f}')
        plt.xlabel('Split', fontweight='bold')
        plt.ylabel('R² Score', fontweight='bold')
        plt.title('Time Series Cross-Validation Results', fontweight='bold', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("="*70)
        
        return scores
    
    def validate_predictions(self, X_test, y_test, confidence_level=0.95):
        """
        Validate model predictions with comprehensive metrics.
        
        Parameters:
        -----------
        X_test : DataFrame
            Test features
        y_test : Series
            True test values
        confidence_level : float
            Confidence level for intervals
        """
        print("\n" + "="*70)
        print("PREDICTION VALIDATION")
        print("="*70)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        # Additional metrics
        errors = y_test - y_pred
        abs_errors = np.abs(errors)
        pct_errors = (errors / y_test) * 100
        
        print(f"\nPerformance Metrics:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: ${mae:,.2f}")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  MAPE: {mape:.2f}%")
        
        print(f"\nError Statistics:")
        print(f"  Mean Error: ${errors.mean():,.2f}")
        print(f"  Median Error: ${np.median(errors):,.2f}")
        print(f"  Std Dev of Errors: ${errors.std():,.2f}")
        print(f"  Max Underestimation: ${errors.min():,.2f}")
        print(f"  Max Overestimation: ${errors.max():,.2f}")
        
        # Percentage within thresholds
        within_5_pct = (abs_errors / y_test <= 0.05).sum() / len(y_test) * 100
        within_10_pct = (abs_errors / y_test <= 0.10).sum() / len(y_test) * 100
        within_20_pct = (abs_errors / y_test <= 0.20).sum() / len(y_test) * 100
        
        print(f"\nPrediction Accuracy:")
        print(f"  Within 5% of actual: {within_5_pct:.2f}%")
        print(f"  Within 10% of actual: {within_10_pct:.2f}%")
        print(f"  Within 20% of actual: {within_20_pct:.2f}%")
        
        # Confidence intervals (simplified bootstrap approach)
        n_bootstrap = 1000
        bootstrap_r2 = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
            boot_r2 = r2_score(y_test.iloc[indices], y_pred[indices])
            bootstrap_r2.append(boot_r2)
        
        ci_lower = np.percentile(bootstrap_r2, (1 - confidence_level) / 2 * 100)
        ci_upper = np.percentile(bootstrap_r2, (1 + confidence_level) / 2 * 100)
        
        print(f"\nR² Confidence Interval ({int(confidence_level*100)}%):")
        print(f"  [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Store results
        self.validation_results.update({
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'within_5_pct': within_5_pct,
            'within_10_pct': within_10_pct,
            'within_20_pct': within_20_pct,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })
        
        print("="*70)
        
        return y_pred
    
    def residual_analysis(self, X_test, y_test, y_pred=None):
        """
        Perform residual analysis.
        
        Parameters:
        -----------
        X_test : DataFrame
            Test features
        y_test : Series
            True test values
        y_pred : array, optional
            Predictions (will compute if not provided)
        """
        if y_pred is None:
            y_pred = self.model.predict(X_test)
        
        residuals = y_test - y_pred
        
        print("\n" + "="*70)
        print("RESIDUAL ANALYSIS")
        print("="*70)
        
        # Normality test (Shapiro-Wilk)
        from scipy import stats
        stat, p_value = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
        
        print(f"\nResidual Distribution:")
        print(f"  Mean: ${residuals.mean():,.2f}")
        print(f"  Std Dev: ${residuals.std():,.2f}")
        print(f"  Skewness: {stats.skew(residuals):.4f}")
        print(f"  Kurtosis: {stats.kurtosis(residuals):.4f}")
        print(f"\nShapiro-Wilk Test (Normality):")
        print(f"  Statistic: {stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        if p_value > 0.05:
            print("  ✓ Residuals appear normally distributed")
        else:
            print("  ⚠ Residuals may not be normally distributed")
        
        # Visualize residuals
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Values', fontweight='bold')
        axes[0, 0].set_ylabel('Residuals', fontweight='bold')
        axes[0, 0].set_title('Residuals vs Predicted', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Histogram of residuals
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Residuals', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title('Distribution of Residuals', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Scale-Location Plot
        standardized_residuals = residuals / residuals.std()
        axes[1, 1].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), 
                          alpha=0.5, edgecolors='k', linewidths=0.5)
        axes[1, 1].set_xlabel('Predicted Values', fontweight='bold')
        axes[1, 1].set_ylabel('√|Standardized Residuals|', fontweight='bold')
        axes[1, 1].set_title('Scale-Location Plot', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("="*70)
    
    def generate_validation_report(self, save_path='results/validation_report.txt'):
        """
        Generate comprehensive validation report.
        
        Parameters:
        -----------
        save_path : str
            Path to save the report
        """
        report = []
        report.append("="*70)
        report.append("WALMART SALES PREDICTION - VALIDATION REPORT")
        report.append("="*70)
        report.append("")
        
        if 'cv_mean' in self.validation_results:
            report.append("CROSS-VALIDATION RESULTS:")
            report.append(f"  Mean Score: {self.validation_results['cv_mean']:.4f}")
            report.append(f"  Std Dev: {self.validation_results['cv_std']:.4f}")
            report.append("")
        
        if 'ts_mean' in self.validation_results:
            report.append("TIME SERIES VALIDATION:")
            report.append(f"  Mean R²: {self.validation_results['ts_mean']:.4f}")
            report.append("")
        
        if 'r2' in self.validation_results:
            report.append("PREDICTION METRICS:")
            report.append(f"  R² Score: {self.validation_results['r2']:.4f}")
            report.append(f"  MAE: ${self.validation_results['mae']:,.2f}")
            report.append(f"  RMSE: ${self.validation_results['rmse']:,.2f}")
            report.append(f"  MAPE: {self.validation_results['mape']:.2f}%")
            report.append("")
            
            report.append("ACCURACY THRESHOLDS:")
            report.append(f"  Within 5%: {self.validation_results['within_5_pct']:.2f}%")
            report.append(f"  Within 10%: {self.validation_results['within_10_pct']:.2f}%")
            report.append(f"  Within 20%: {self.validation_results['within_20_pct']:.2f}%")
            report.append("")
            
            report.append("CONFIDENCE INTERVAL (95%):")
            report.append(f"  [{self.validation_results['ci_lower']:.4f}, {self.validation_results['ci_upper']:.4f}]")
        
        report.append("")
        report.append("="*70)
        
        report_text = "\n".join(report)
        print(report_text)
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"\n✓ Validation report saved to: {save_path}")
        
        return report_text


if __name__ == "__main__":
    print("Model Validation Module - Ready to use!")
    print("\nUsage:")
    print("  from model_validation import ModelValidator")
    print("  validator = ModelValidator(model_path='models/gradient_boosting_model.joblib')")
    print("  validator.validate_data_quality(X_test, y_test)")
    print("  validator.cross_validate(X_train, y_train, cv=5)")
    print("  validator.time_series_validation(X_train, y_train)")
    print("  validator.validate_predictions(X_test, y_test)")
    print("  validator.residual_analysis(X_test, y_test)")
    print("  validator.generate_validation_report()")
