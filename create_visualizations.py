"""
Comprehensive Visualization Script for Walmart Sales Prediction
Creates publication-quality visualizations for analysis and presentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("CREATING COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

# Load data
print("\n[1/10] Loading data...")
df = pd.read_csv('Walmart.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
print(f"✅ Data loaded: {df.shape[0]:,} records")

# Load models and predictions
from src.data_preprocessing import WalmartDataPreprocessor
preprocessor = WalmartDataPreprocessor('Walmart.csv')
preprocessor.load_data()
preprocessor.parse_dates()
preprocessor.create_lag_features()
preprocessor.create_rolling_features()
preprocessor.handle_missing_values()
X_train, X_test, y_train, y_test = preprocessor.split_data()
X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

models = {
    'Random Forest': joblib.load('models/random_forest_model.joblib'),
    'Linear Regression': joblib.load('models/linear_regression_model.joblib'),
    'Decision Tree': joblib.load('models/decision_tree_model.joblib'),
    'Gradient Boosting': joblib.load('models/gradient_boosting_model.joblib'),
    'XGBoost': joblib.load('models/xgboost_model.joblib')
}

# ============================================================================
# VISUALIZATION 1: Sales Distribution and Outliers
# ============================================================================
print("\n[2/10] Creating sales distribution visualization...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Walmart Sales Distribution Analysis', fontsize=16, fontweight='bold')

# Histogram
axes[0, 0].hist(df['Weekly_Sales'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(df['Weekly_Sales'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: ${df['Weekly_Sales'].mean():,.0f}")
axes[0, 0].axvline(df['Weekly_Sales'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: ${df['Weekly_Sales'].median():,.0f}")
axes[0, 0].set_xlabel('Weekly Sales ($)', fontweight='bold', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontweight='bold', fontsize=11)
axes[0, 0].set_title('Sales Distribution (Histogram)', fontweight='bold', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Box plot
axes[0, 1].boxplot(df['Weekly_Sales'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', edgecolor='black'),
                   medianprops=dict(color='red', linewidth=2),
                   whiskerprops=dict(color='black', linewidth=1.5),
                   capprops=dict(color='black', linewidth=1.5))
axes[0, 1].set_ylabel('Weekly Sales ($)', fontweight='bold', fontsize=11)
axes[0, 1].set_title('Sales Distribution (Box Plot)', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Q-Q plot
from scipy import stats
stats.probplot(df['Weekly_Sales'], dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normality Check)', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# Violin plot by Holiday
axes[1, 1].violinplot([df[df['Holiday_Flag']==0]['Weekly_Sales'], 
                        df[df['Holiday_Flag']==1]['Weekly_Sales']],
                       positions=[0, 1], showmeans=True, showmedians=True)
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_xticklabels(['Non-Holiday', 'Holiday'])
axes[1, 1].set_ylabel('Weekly Sales ($)', fontweight='bold', fontsize=11)
axes[1, 1].set_title('Sales Distribution: Holiday vs Non-Holiday', fontweight='bold', fontsize=12)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/viz_01_sales_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/viz_01_sales_distribution.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: Time Series Analysis
# ============================================================================
print("\n[3/10] Creating time series visualization...")
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle('Time Series Analysis of Walmart Sales', fontsize=16, fontweight='bold')

# Overall trend
df_sorted = df.sort_values('Date')
axes[0].plot(df_sorted['Date'], df_sorted['Weekly_Sales'], alpha=0.5, color='steelblue', linewidth=0.5)
axes[0].plot(df_sorted.groupby('Date')['Weekly_Sales'].mean().index,
             df_sorted.groupby('Date')['Weekly_Sales'].mean().values,
             color='red', linewidth=2, label='Average Sales')
axes[0].set_xlabel('Date', fontweight='bold', fontsize=11)
axes[0].set_ylabel('Weekly Sales ($)', fontweight='bold', fontsize=11)
axes[0].set_title('Sales Over Time (All Stores)', fontweight='bold', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Monthly average
df_sorted['Month'] = df_sorted['Date'].dt.to_period('M')
monthly_sales = df_sorted.groupby('Month')['Weekly_Sales'].mean()
axes[1].bar(range(len(monthly_sales)), monthly_sales.values, color='coral', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Month', fontweight='bold', fontsize=11)
axes[1].set_ylabel('Average Weekly Sales ($)', fontweight='bold', fontsize=11)
axes[1].set_title('Average Sales by Month', fontweight='bold', fontsize=12)
axes[1].set_xticks(range(0, len(monthly_sales), 3))
axes[1].set_xticklabels([str(monthly_sales.index[i]) for i in range(0, len(monthly_sales), 3)], rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

# Year comparison
df_sorted['Year'] = df_sorted['Date'].dt.year
yearly_sales = df_sorted.groupby('Year')['Weekly_Sales'].agg(['mean', 'std'])
axes[2].bar(yearly_sales.index, yearly_sales['mean'], yerr=yearly_sales['std'], 
            color='lightgreen', edgecolor='black', alpha=0.7, capsize=5)
axes[2].set_xlabel('Year', fontweight='bold', fontsize=11)
axes[2].set_ylabel('Average Weekly Sales ($)', fontweight='bold', fontsize=11)
axes[2].set_title('Average Sales by Year (with Std Dev)', fontweight='bold', fontsize=12)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/viz_02_time_series.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/viz_02_time_series.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: Feature Correlations
# ============================================================================
print("\n[4/10] Creating correlation visualization...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Feature Correlation Analysis', fontsize=16, fontweight='bold')

# Correlation heatmap
numeric_cols = ['Weekly_Sales', 'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
corr_matrix = df[numeric_cols].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[0])
axes[0].set_title('Correlation Heatmap', fontweight='bold', fontsize=12)

# Correlation with target
target_corr = corr_matrix['Weekly_Sales'].drop('Weekly_Sales').sort_values()
colors = ['red' if x < 0 else 'green' for x in target_corr.values]
axes[1].barh(target_corr.index, target_corr.values, color=colors, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Correlation with Weekly Sales', fontweight='bold', fontsize=11)
axes[1].set_title('Feature Importance (Correlation)', fontweight='bold', fontsize=12)
axes[1].axvline(x=0, color='black', linewidth=1)
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results/viz_03_correlations.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/viz_03_correlations.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: Store Analysis
# ============================================================================
print("\n[5/10] Creating store analysis visualization...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Store-Level Sales Analysis', fontsize=16, fontweight='bold')

# Average sales by store
store_sales = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
top_10_stores = store_sales.head(10)
axes[0, 0].bar(range(len(top_10_stores)), top_10_stores.values, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Store Number', fontweight='bold', fontsize=11)
axes[0, 0].set_ylabel('Average Weekly Sales ($)', fontweight='bold', fontsize=11)
axes[0, 0].set_title('Top 10 Stores by Average Sales', fontweight='bold', fontsize=12)
axes[0, 0].set_xticks(range(len(top_10_stores)))
axes[0, 0].set_xticklabels(top_10_stores.index, rotation=45)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Sales variability by store
store_std = df.groupby('Store')['Weekly_Sales'].std().sort_values(ascending=False).head(10)
axes[0, 1].bar(range(len(store_std)), store_std.values, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Store Number', fontweight='bold', fontsize=11)
axes[0, 1].set_ylabel('Sales Std Deviation ($)', fontweight='bold', fontsize=11)
axes[0, 1].set_title('Top 10 Stores by Sales Variability', fontweight='bold', fontsize=12)
axes[0, 1].set_xticks(range(len(store_std)))
axes[0, 1].set_xticklabels(store_std.index, rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Store count distribution
axes[1, 0].hist(df.groupby('Store').size(), bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Number of Records', fontweight='bold', fontsize=11)
axes[1, 0].set_ylabel('Number of Stores', fontweight='bold', fontsize=11)
axes[1, 0].set_title('Distribution of Records per Store', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Sales range by store
store_range = df.groupby('Store')['Weekly_Sales'].agg(['min', 'max'])
store_range['range'] = store_range['max'] - store_range['min']
top_range = store_range.nlargest(10, 'range')
axes[1, 1].bar(range(len(top_range)), top_range['range'].values, color='gold', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Store Number', fontweight='bold', fontsize=11)
axes[1, 1].set_ylabel('Sales Range ($)', fontweight='bold', fontsize=11)
axes[1, 1].set_title('Top 10 Stores by Sales Range', fontweight='bold', fontsize=12)
axes[1, 1].set_xticks(range(len(top_range)))
axes[1, 1].set_xticklabels(top_range.index, rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/viz_04_store_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/viz_04_store_analysis.png")
plt.close()

# ============================================================================
# VISUALIZATION 5: Environmental Factors
# ============================================================================
print("\n[6/10] Creating environmental factors visualization...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Environmental Factors Impact on Sales', fontsize=16, fontweight='bold')

# Temperature vs Sales
axes[0, 0].scatter(df['Temperature'], df['Weekly_Sales'], alpha=0.3, s=10, color='steelblue')
z = np.polyfit(df['Temperature'], df['Weekly_Sales'], 2)
p = np.poly1d(z)
x_smooth = np.linspace(df['Temperature'].min(), df['Temperature'].max(), 100)
axes[0, 0].plot(x_smooth, p(x_smooth), "r-", linewidth=2, label='Trend')
axes[0, 0].set_xlabel('Temperature (°F)', fontweight='bold', fontsize=11)
axes[0, 0].set_ylabel('Weekly Sales ($)', fontweight='bold', fontsize=11)
axes[0, 0].set_title('Temperature vs Sales', fontweight='bold', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Fuel Price vs Sales
axes[0, 1].scatter(df['Fuel_Price'], df['Weekly_Sales'], alpha=0.3, s=10, color='coral')
z = np.polyfit(df['Fuel_Price'], df['Weekly_Sales'], 1)
p = np.poly1d(z)
x_smooth = np.linspace(df['Fuel_Price'].min(), df['Fuel_Price'].max(), 100)
axes[0, 1].plot(x_smooth, p(x_smooth), "r-", linewidth=2, label='Trend')
axes[0, 1].set_xlabel('Fuel Price ($/gallon)', fontweight='bold', fontsize=11)
axes[0, 1].set_ylabel('Weekly Sales ($)', fontweight='bold', fontsize=11)
axes[0, 1].set_title('Fuel Price vs Sales', fontweight='bold', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# CPI vs Sales
axes[1, 0].scatter(df['CPI'], df['Weekly_Sales'], alpha=0.3, s=10, color='lightgreen')
z = np.polyfit(df['CPI'], df['Weekly_Sales'], 1)
p = np.poly1d(z)
x_smooth = np.linspace(df['CPI'].min(), df['CPI'].max(), 100)
axes[1, 0].plot(x_smooth, p(x_smooth), "r-", linewidth=2, label='Trend')
axes[1, 0].set_xlabel('CPI', fontweight='bold', fontsize=11)
axes[1, 0].set_ylabel('Weekly Sales ($)', fontweight='bold', fontsize=11)
axes[1, 0].set_title('Consumer Price Index vs Sales', fontweight='bold', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Unemployment vs Sales
axes[1, 1].scatter(df['Unemployment'], df['Weekly_Sales'], alpha=0.3, s=10, color='gold')
z = np.polyfit(df['Unemployment'], df['Weekly_Sales'], 1)
p = np.poly1d(z)
x_smooth = np.linspace(df['Unemployment'].min(), df['Unemployment'].max(), 100)
axes[1, 1].plot(x_smooth, p(x_smooth), "r-", linewidth=2, label='Trend')
axes[1, 1].set_xlabel('Unemployment Rate (%)', fontweight='bold', fontsize=11)
axes[1, 1].set_ylabel('Weekly Sales ($)', fontweight='bold', fontsize=11)
axes[1, 1].set_title('Unemployment vs Sales', fontweight='bold', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/viz_05_environmental_factors.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/viz_05_environmental_factors.png")
plt.close()

# ============================================================================
# VISUALIZATION 6: Model Performance Comparison (Enhanced)
# ============================================================================
print("\n[7/10] Creating enhanced model comparison visualization...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')

# Calculate metrics for all models
results = []
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    results.append({'Model': name, 'R²': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape})

results_df = pd.DataFrame(results)

# R² Score comparison
axes[0, 0].barh(results_df['Model'], results_df['R²']*100, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('R² Score (%)', fontweight='bold', fontsize=11)
axes[0, 0].set_title('R² Score Comparison', fontweight='bold', fontsize=12)
axes[0, 0].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(results_df['R²']*100):
    axes[0, 0].text(v + 1, i, f'{v:.2f}%', va='center', fontweight='bold')

# MAE comparison
axes[0, 1].barh(results_df['Model'], results_df['MAE'], color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('MAE ($)', fontweight='bold', fontsize=11)
axes[0, 1].set_title('Mean Absolute Error', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(results_df['MAE']):
    axes[0, 1].text(v + 5000, i, f'${v:,.0f}', va='center', fontweight='bold', fontsize=9)

# RMSE comparison
axes[0, 2].barh(results_df['Model'], results_df['RMSE'], color='lightgreen', edgecolor='black', alpha=0.7)
axes[0, 2].set_xlabel('RMSE ($)', fontweight='bold', fontsize=11)
axes[0, 2].set_title('Root Mean Squared Error', fontweight='bold', fontsize=12)
axes[0, 2].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(results_df['RMSE']):
    axes[0, 2].text(v + 5000, i, f'${v:,.0f}', va='center', fontweight='bold', fontsize=9)

# MAPE comparison
axes[1, 0].barh(results_df['Model'], results_df['MAPE'], color='gold', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('MAPE (%)', fontweight='bold', fontsize=11)
axes[1, 0].set_title('Mean Absolute Percentage Error', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(results_df['MAPE']):
    axes[1, 0].text(v + 0.5, i, f'{v:.2f}%', va='center', fontweight='bold', fontsize=9)

# Radar chart for best model
best_model_name = results_df.loc[results_df['R²'].idxmax(), 'Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

categories = ['R² Score\n(scaled)', 'Low MAE\n(scaled)', 'Low RMSE\n(scaled)', 'Low MAPE\n(scaled)', 'Accuracy\n(scaled)']
values = [
    results_df.loc[results_df['Model']==best_model_name, 'R²'].values[0],
    1 - (results_df.loc[results_df['Model']==best_model_name, 'MAE'].values[0] / results_df['MAE'].max()),
    1 - (results_df.loc[results_df['Model']==best_model_name, 'RMSE'].values[0] / results_df['RMSE'].max()),
    1 - (results_df.loc[results_df['Model']==best_model_name, 'MAPE'].values[0] / results_df['MAPE'].max()),
    1 - (results_df.loc[results_df['Model']==best_model_name, 'MAPE'].values[0] / 100)
]
values += values[:1]  # Complete the circle

angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
angles += angles[:1]

ax = plt.subplot(2, 3, 5, projection='polar')
ax.plot(angles, values, 'o-', linewidth=2, color='steelblue', label=best_model_name)
ax.fill(angles, values, alpha=0.25, color='steelblue')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylim(0, 1)
ax.set_title(f'{best_model_name}\nPerformance Profile', fontweight='bold', fontsize=12, pad=20)
ax.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Summary table
axes[1, 2].axis('off')
table_data = []
for _, row in results_df.iterrows():
    table_data.append([
        row['Model'],
        f"{row['R²']*100:.2f}%",
        f"${row['MAE']:,.0f}",
        f"{row['MAPE']:.2f}%"
    ])

table = axes[1, 2].table(cellText=table_data,
                         colLabels=['Model', 'R²', 'MAE', 'MAPE'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.35, 0.2, 0.25, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color header
for i in range(4):
    table[(0, i)].set_facecolor('steelblue')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color best model row
best_idx = results_df['R²'].idxmax() + 1
for i in range(4):
    table[(best_idx, i)].set_facecolor('lightgreen')
    table[(best_idx, i)].set_text_props(weight='bold')

axes[1, 2].set_title('Performance Summary Table', fontweight='bold', fontsize=12, pad=20)

plt.tight_layout()
plt.savefig('results/viz_06_model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/viz_06_model_comparison.png")
plt.close()

# ============================================================================
# VISUALIZATION 7: Prediction Quality Analysis
# ============================================================================
print("\n[8/10] Creating prediction quality visualization...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'{best_model_name}: Prediction Quality Analysis', fontsize=16, fontweight='bold')

# Predicted vs Actual
axes[0, 0].scatter(y_test, y_pred_best, alpha=0.5, s=30, edgecolors='k', linewidths=0.5, color='steelblue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Sales ($)', fontweight='bold', fontsize=11)
axes[0, 0].set_ylabel('Predicted Sales ($)', fontweight='bold', fontsize=11)
axes[0, 0].set_title('Predicted vs Actual Sales', fontweight='bold', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Add R² annotation
r2_val = r2_score(y_test, y_pred_best)
axes[0, 0].text(0.05, 0.95, f'R² = {r2_val:.4f}', transform=axes[0, 0].transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontweight='bold')

# Residual plot
residuals = y_test - y_pred_best
axes[0, 1].scatter(y_pred_best, residuals, alpha=0.5, s=30, edgecolors='k', linewidths=0.5, color='coral')
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Sales ($)', fontweight='bold', fontsize=11)
axes[0, 1].set_ylabel('Residuals ($)', fontweight='bold', fontsize=11)
axes[0, 1].set_title('Residual Plot', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# Residual distribution
axes[1, 0].hist(residuals, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: ${residuals.mean():,.0f}')
axes[1, 0].set_xlabel('Residuals ($)', fontweight='bold', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontweight='bold', fontsize=11)
axes[1, 0].set_title('Residual Distribution', fontweight='bold', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Error percentage distribution
error_pct = np.abs(residuals / y_test) * 100
axes[1, 1].hist(error_pct, bins=50, color='gold', edgecolor='black', alpha=0.7)
axes[1, 1].axvline(error_pct.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {error_pct.mean():.2f}%')
axes[1, 1].set_xlabel('Absolute Error (%)', fontweight='bold', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontweight='bold', fontsize=11)
axes[1, 1].set_title('Error Percentage Distribution', fontweight='bold', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/viz_07_prediction_quality.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/viz_07_prediction_quality.png")
plt.close()

# ============================================================================
# VISUALIZATION 8: Feature Importance
# ============================================================================
print("\n[9/10] Creating feature importance visualization...")
if hasattr(best_model, 'feature_importances_'):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{best_model_name}: Feature Importance Analysis', fontsize=16, fontweight='bold')
    
    # Get feature names
    with open('models/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f]
    
    # Feature importance
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Bar plot
    axes[0].barh(range(len(indices)), importances[indices], color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_yticks(range(len(indices)))
    axes[0].set_yticklabels([feature_names[i] for i in indices])
    axes[0].set_xlabel('Importance Score', fontweight='bold', fontsize=11)
    axes[0].set_title('Feature Importance (All Features)', fontweight='bold', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Top 10 features pie chart
    top_10_indices = indices[:10]
    top_10_importances = importances[top_10_indices]
    top_10_names = [feature_names[i] for i in top_10_indices]
    
    colors_pie = plt.cm.Set3(range(len(top_10_names)))
    axes[1].pie(top_10_importances, labels=top_10_names, autopct='%1.1f%%',
                startangle=90, colors=colors_pie, textprops={'fontsize': 9})
    axes[1].set_title('Top 10 Features Contribution', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/viz_08_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/viz_08_feature_importance.png")
    plt.close()
else:
    print("⚠️  Best model doesn't have feature_importances_ attribute, skipping...")

# ============================================================================
# VISUALIZATION 9: Holiday Impact Analysis
# ============================================================================
print("\n[10/10] Creating holiday impact visualization...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Holiday Impact on Walmart Sales', fontsize=16, fontweight='bold')

# Sales comparison
holiday_sales = df[df['Holiday_Flag']==1]['Weekly_Sales']
non_holiday_sales = df[df['Holiday_Flag']==0]['Weekly_Sales']

axes[0, 0].boxplot([non_holiday_sales, holiday_sales], labels=['Non-Holiday', 'Holiday'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', edgecolor='black'),
                    medianprops=dict(color='red', linewidth=2))
axes[0, 0].set_ylabel('Weekly Sales ($)', fontweight='bold', fontsize=11)
axes[0, 0].set_title('Sales Distribution: Holiday vs Non-Holiday', fontweight='bold', fontsize=12)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Average sales
avg_sales = [non_holiday_sales.mean(), holiday_sales.mean()]
axes[0, 1].bar(['Non-Holiday', 'Holiday'], avg_sales, color=['lightblue', 'coral'], 
               edgecolor='black', alpha=0.7)
axes[0, 1].set_ylabel('Average Weekly Sales ($)', fontweight='bold', fontsize=11)
axes[0, 1].set_title('Average Sales Comparison', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(avg_sales):
    axes[0, 1].text(i, v + 10000, f'${v:,.0f}', ha='center', fontweight='bold')

# Holiday sales over time
df_sorted_holiday = df[df['Holiday_Flag']==1].sort_values('Date')
axes[1, 0].scatter(df_sorted_holiday['Date'], df_sorted_holiday['Weekly_Sales'], 
                   color='red', alpha=0.6, s=50, label='Holiday Sales')
axes[1, 0].plot(df_sorted_holiday['Date'], df_sorted_holiday['Weekly_Sales'], 
                color='red', alpha=0.3, linewidth=1)
axes[1, 0].set_xlabel('Date', fontweight='bold', fontsize=11)
axes[1, 0].set_ylabel('Weekly Sales ($)', fontweight='bold', fontsize=11)
axes[1, 0].set_title('Holiday Sales Over Time', fontweight='bold', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Count of records
counts = [len(non_holiday_sales), len(holiday_sales)]
axes[1, 1].pie(counts, labels=['Non-Holiday', 'Holiday'], autopct='%1.1f%%',
               colors=['lightblue', 'coral'], startangle=90)
axes[1, 1].set_title(f'Holiday vs Non-Holiday Distribution\n(Total: {len(df):,} records)', 
                     fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('results/viz_09_holiday_impact.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/viz_09_holiday_impact.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("=" * 80)
print("\nGenerated Files:")
print("  1. viz_01_sales_distribution.png - Sales distribution analysis")
print("  2. viz_02_time_series.png - Time series patterns")
print("  3. viz_03_correlations.png - Feature correlations")
print("  4. viz_04_store_analysis.png - Store-level analysis")
print("  5. viz_05_environmental_factors.png - External factors impact")
print("  6. viz_06_model_comparison.png - Model performance comparison")
print("  7. viz_07_prediction_quality.png - Prediction quality metrics")
print("  8. viz_08_feature_importance.png - Feature importance analysis")
print("  9. viz_09_holiday_impact.png - Holiday impact analysis")
print("\n📁 All files saved in: results/ folder")
print("=" * 80)
