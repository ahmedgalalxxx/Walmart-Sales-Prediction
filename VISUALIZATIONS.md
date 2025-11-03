# Walmart Sales Prediction - Visualization Gallery This document provides an overview of all visualizations created for the Walmart Sales Prediction project. ## Complete Visualization Suite ### 1. Sales Distribution Analysis
**File:** `viz_01_sales_distribution.png` **Contains:**
- Sales Histogram with mean and median lines
- Box plot showing outliers
- Q-Q plot for normality check
- Violin plot comparing Holiday vs Non-Holiday sales **Purpose:** Understand the overall distribution of sales data, identify outliers, and check data normality. --- ### 2. Time Series Analysis
**File:** `viz_02_time_series.png` **Contains:**
- Overall sales trend over time (all stores)
- Monthly average sales pattern
- Yearly sales comparison with standard deviation **Purpose:** Identify temporal patterns, seasonality, and trends in sales data over time. --- ### 3. Feature Correlations
**File:** `viz_03_correlations.png` **Contains:**
- Complete correlation heatmap of all numeric features
- Bar chart showing correlation strength with target variable **Purpose:** Understand relationships between features and identify multicollinearity issues. --- ### 4. Store Analysis
**File:** `viz_04_store_analysis.png` **Contains:**
- Top 10 stores by average sales
- Top 10 stores by sales variability
- Distribution of records per store
- Top 10 stores by sales range **Purpose:** Compare store performance and identify high/low performing locations. --- ### 5. Environmental Factors Impact
**File:** `viz_05_environmental_factors.png` **Contains:**
- Temperature vs Sales scatter plot with trend line
- Fuel Price vs Sales scatter plot with trend line
- CPI vs Sales scatter plot with trend line
- Unemployment vs Sales scatter plot with trend line **Purpose:** Analyze how external economic and environmental factors affect sales. --- ### 6. Model Performance Comparison
**File:** `viz_06_model_comparison.png` **Contains:**
- R² Score comparison across all models
- Mean Absolute Error (MAE) comparison
- Root Mean Squared Error (RMSE) comparison
- Mean Absolute Percentage Error (MAPE) comparison
- Radar chart for best model performance profile
- Summary performance table **Purpose:** Comprehensive comparison of all trained models to identify the best performer. --- ### 7. Prediction Quality Analysis
**File:** `viz_07_prediction_quality.png` **Contains:**
- Predicted vs Actual scatter plot with R² score
- Residual plot showing error distribution
- Residual histogram with mean line
- Error percentage distribution **Purpose:** Evaluate the quality and accuracy of model predictions. --- ### 8. Feature Importance Analysis
**File:** `viz_08_feature_importance.png` **Contains:**
- Horizontal bar chart of all feature importances
- Pie chart showing top 10 features contribution **Purpose:** Understand which features contribute most to model predictions. --- ### 9. Holiday Impact Analysis
**File:** `viz_09_holiday_impact.png` **Contains:**
- Box plot comparison of Holiday vs Non-Holiday sales
- Bar chart of average sales comparison
- Holiday sales over time scatter plot
- Pie chart of Holiday vs Non-Holiday distribution **Purpose:** Analyze the impact of holidays on Walmart sales. --- ### 10. Overfitting Analysis
**File:** `overfitting_analysis.png` **Contains:**
- Train vs Test R² comparison
- Overfitting gap visualization
- Residual plot for best model
- Predicted vs Actual for best model **Purpose:** Verify that models are not overfitting and are generalizing well. --- ## Using the Visualizations ### For Presentations:
All images are high-resolution (300 DPI) and suitable for:
- PowerPoint/Google Slides presentations
- Research papers and reports
- Posters and academic presentations
- Online portfolios ### For Analysis:
Each visualization provides specific insights:
- **Distribution plots** → Data quality and outliers
- **Time series plots** → Temporal patterns and seasonality
- **Correlation plots** → Feature relationships
- **Model comparison** → Best model selection
- **Prediction quality** → Model reliability
- **Feature importance** → Key drivers of sales --- ## Regenerating Visualizations To create/update all visualizations: ```bash
python create_visualizations.py
``` This will regenerate all 9 visualization files in the `results/` folder. --- ## Key Insights from Visualizations ### Sales Patterns:
- Sales show strong seasonality with peaks during holiday periods
- Average weekly sales: ~$1,050,000
- Holiday sales are generally higher than non-holiday sales ### Model Performance:
- **Best Model:** Random Forest (96.20% R²)
- All models show good generalization (no overfitting)
- Predictions are accurate with low error rates ### Important Features:
- Lag features (previous week sales) are most important
- Store ID is a strong predictor
- Economic factors (CPI, Unemployment) have moderate impact ### Store Insights:
- Top stores significantly outperform average
- High variability across different stores
- Consistent data collection across all stores --- ## File Locations All visualizations are stored in: `results/` - Total visualizations: **10 files**
- File format: PNG (high-resolution, 300 DPI)
- Total size: ~15-20 MB
- Naming convention: `viz_XX_description.png` --- **Generated:** November 1, 2025 **Project:** Walmart Sales Prediction **Author:** Ahmed Galal
