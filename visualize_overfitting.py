"""
Visual Overfitting Analysis - Creates plots to visualize overfitting
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from src.data_preprocessing import WalmartDataPreprocessor

print("Creating overfitting visualization plots...")

# Load models
models = {
    'Random Forest': joblib.load('models/random_forest_model.joblib'),
    'Linear Regression': joblib.load('models/linear_regression_model.joblib'),
    'Decision Tree': joblib.load('models/decision_tree_model.joblib'),
    'Gradient Boosting': joblib.load('models/gradient_boosting_model.joblib'),
    'XGBoost': joblib.load('models/xgboost_model.joblib')
}

# Prepare data
preprocessor = WalmartDataPreprocessor('Walmart.csv')
preprocessor.load_data()
preprocessor.parse_dates()
preprocessor.create_lag_features()
preprocessor.create_rolling_features()
preprocessor.handle_missing_values()
X_train, X_test, y_train, y_test = preprocessor.split_data()
X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

# Calculate train and test scores
train_scores = []
test_scores = []
model_names = []

for name, model in models.items():
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_scores.append(train_r2 * 100)
    test_scores.append(test_r2 * 100)
    model_names.append(name)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Overfitting Analysis - Walmart Sales Prediction', fontsize=16, fontweight='bold')

# Plot 1: Train vs Test R² Scores
x = np.arange(len(model_names))
width = 0.35

ax1 = axes[0, 0]
bars1 = ax1.bar(x - width/2, train_scores, width, label='Train R²', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, test_scores, width, label='Test R²', color='coral', alpha=0.8)

ax1.set_ylabel('R² Score (%)', fontweight='bold', fontsize=11)
ax1.set_title('Train vs Test R² Score (Overfitting Check)', fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5)

# Add difference values on top
for i, (train, test) in enumerate(zip(train_scores, test_scores)):
    diff = abs(train - test)
    color = 'green' if diff < 5 else 'orange' if diff < 10 else 'red'
    ax1.text(i, max(train, test) + 1, f'Δ{diff:.1f}%', 
             ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

# Plot 2: Overfitting Gap
ax2 = axes[0, 1]
gaps = [train - test for train, test in zip(train_scores, test_scores)]
colors = ['green' if abs(g) < 5 else 'orange' if abs(g) < 10 else 'red' for g in gaps]
bars = ax2.bar(model_names, gaps, color=colors, alpha=0.7, edgecolor='black')

ax2.set_ylabel('Train R² - Test R² (%)', fontweight='bold', fontsize=11)
ax2.set_title('Overfitting Gap (Lower is Better)', fontweight='bold', fontsize=12)
ax2.set_xticklabels(model_names, rotation=45, ha='right')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axhline(y=5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Acceptable Threshold')
ax2.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Overfitting Threshold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, gap in zip(bars, gaps):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{gap:.2f}%', ha='center', va='bottom' if height > 0 else 'top',
             fontsize=9, fontweight='bold')

# Plot 3: Residual Plot for Best Model (Random Forest)
ax3 = axes[1, 0]
rf_model = models['Random Forest']
y_test_pred = rf_model.predict(X_test_scaled)
residuals = y_test - y_test_pred

ax3.scatter(y_test_pred, residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Predicted Sales ($)', fontweight='bold', fontsize=11)
ax3.set_ylabel('Residuals ($)', fontweight='bold', fontsize=11)
ax3.set_title('Random Forest: Residual Plot (Test Set)', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3)

# Add statistics text
mean_res = residuals.mean()
std_res = residuals.std()
ax3.text(0.05, 0.95, f'Mean: ${mean_res:,.0f}\nStd Dev: ${std_res:,.0f}',
         transform=ax3.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Predicted vs Actual for Best Model
ax4 = axes[1, 1]
ax4.scatter(y_test, y_test_pred, alpha=0.5, edgecolors='k', linewidths=0.5)
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Sales ($)', fontweight='bold', fontsize=11)
ax4.set_ylabel('Predicted Sales ($)', fontweight='bold', fontsize=11)
ax4.set_title('Random Forest: Predicted vs Actual (Test Set)', fontweight='bold', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add R² text
test_r2 = r2_score(y_test, y_test_pred)
ax4.text(0.05, 0.95, f'Test R²: {test_r2:.4f}\n({test_r2*100:.2f}%)',
         transform=ax4.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
         fontweight='bold')

plt.tight_layout()
plt.savefig('results/overfitting_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved to: results/overfitting_analysis.png")
plt.show()

print("\n" + "=" * 70)
print("INTERPRETATION GUIDE:")
print("=" * 70)
print("\n📊 Plot 1 (Train vs Test R²):")
print("   - Bars should be close in height")
print("   - Large gaps indicate overfitting")
print("\n📊 Plot 2 (Overfitting Gap):")
print("   - Green bars = Good (< 5% gap)")
print("   - Orange bars = Acceptable (5-10% gap)")
print("   - Red bars = Overfitting (> 10% gap)")
print("\n📊 Plot 3 (Residual Plot):")
print("   - Points should be randomly scattered around 0")
print("   - Patterns indicate model bias")
print("\n📊 Plot 4 (Predicted vs Actual):")
print("   - Points should follow the red diagonal line")
print("   - Scatter indicates prediction error")
print("=" * 70)
