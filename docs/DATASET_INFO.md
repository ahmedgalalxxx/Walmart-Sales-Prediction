# Walmart Sales Dataset Information

## Dataset Source

**Kaggle Link**: https://www.kaggle.com/code/yasserh/walmart-sales-prediction-best-ml-algorithms/input

## Dataset Description

This dataset contains historical sales data for Walmart stores, which can be used to predict future sales based on various factors.

## Expected Features

The Walmart sales dataset typically includes the following features:

### Target Variable
- **Weekly_Sales**: The sales for the given store and week (Target variable for prediction)

### Feature Variables

1. **Store**
   - Type: Numerical
   - Description: Store number (identifier for different Walmart stores)
   - Range: Typically 1-45

2. **Date**
   - Type: Date/String
   - Description: Week of sales
   - Format: DD-MM-YYYY or similar
   - Note: Can be extracted into multiple features (Year, Month, Week, Day of Week)

3. **Holiday_Flag**
   - Type: Binary (0 or 1)
   - Description: Indicates whether the week contains a special holiday
   - Values:
     - 0: Non-holiday week
     - 1: Holiday week

4. **Temperature**
   - Type: Numerical (Float)
   - Description: Temperature on the day of sale
   - Unit: Fahrenheit
   - Range: Typically 0-100°F

5. **Fuel_Price**
   - Type: Numerical (Float)
   - Description: Cost of fuel in the region
   - Unit: Dollars per gallon
   - Impact: Higher fuel prices may affect customer shopping patterns

6. **CPI (Consumer Price Index)**
   - Type: Numerical (Float)
   - Description: Economic indicator measuring the average change in prices
   - Impact: Reflects inflation and purchasing power

7. **Unemployment**
   - Type: Numerical (Float)
   - Description: Unemployment rate in the region
   - Unit: Percentage
   - Impact: Higher unemployment may reduce sales

## Data Characteristics

- **Number of Records**: Typically 6,435+ records
- **Number of Features**: 7-8 features (before feature engineering)
- **Time Period**: Multiple years of historical data
- **Missing Values**: May contain some missing values (handled in preprocessing)

## Feature Engineering Opportunities

The preprocessing pipeline automatically creates these additional features:

### From Date Column:
- **Year**: Extracted year
- **Month**: Month of the year (1-12)
- **Day**: Day of the month
- **WeekOfYear**: Week number in the year
- **DayOfWeek**: Day of the week (0-6, Monday is 0)

### Potential Additional Features:
- **IsWeekend**: Binary flag for weekends
- **Quarter**: Quarter of the year (Q1-Q4)
- **Season**: Season (Winter, Spring, Summer, Fall)
- **MonthEnd**: Binary flag for month end
- **YearEnd**: Binary flag for year end

## Data Quality Considerations

1. **Outliers**: Sales data may contain outliers due to special events
2. **Seasonality**: Strong seasonal patterns in retail sales
3. **Trends**: Long-term trends due to economic factors
4. **Store Variations**: Different stores may have different sales patterns
5. **Holiday Effects**: Significant impact from holidays

## Target Variable Distribution

The `Weekly_Sales` typically shows:
- Right-skewed distribution
- Range: $0 to $3,000,000+
- Mean: Approximately $1,000,000
- Median: Lower than mean due to right skew

## Correlation Insights

Common correlations found:
- **Strong Positive**: Temperature (in some seasons)
- **Moderate Negative**: Unemployment, Fuel_Price
- **Strong Positive**: Holiday_Flag (higher sales during holidays)
- **Variable**: CPI effects depend on economic conditions

## Usage in This Project

The dataset is used to:
1. Predict weekly sales for Walmart stores
2. Compare multiple machine learning algorithms
3. Identify important features affecting sales
4. Understand seasonal patterns and trends

## Download Instructions

1. Visit the Kaggle dataset page
2. Click "Download" button
3. Extract the CSV file
4. Place it in `data/raw/` directory
5. Rename to `Walmart.csv` if needed

## Citation

If you use this dataset, please cite:
- Source: Kaggle
- Dataset: Walmart Sales Prediction
- URL: https://www.kaggle.com/code/yasserh/walmart-sales-prediction-best-ml-algorithms/input

## Additional Resources

- [Kaggle Walmart Sales Competition](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
- [Time Series Forecasting Resources](https://www.kaggle.com/learn/time-series)
- [Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)

## Notes

- The actual dataset structure may vary slightly
- Always inspect the data after loading to confirm column names
- Adjust preprocessing scripts if column names differ
- Consider domain knowledge when engineering features

---

**Last Updated**: 2024
**Maintained By**: Project Contributors
