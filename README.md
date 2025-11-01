# Walmart Sales Prediction 📊

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmedgalalxxx/Walmart-Sales-Prediction/blob/main/Walmart_Sales_Prediction_Colab.ipynb)

A comprehensive machine learning project to predict Walmart weekly sales based on various features including store information, holiday flags, temperature, fuel prices, CPI, and unemployment rates.

**🚀 Quick Start:** Click the "Open in Colab" badge above to run the complete project in your browser!

## 📋 Project Overview

This project uses historical sales data from Walmart stores to build predictive models that can forecast future weekly sales. The dataset includes multiple features that influence sales patterns, making it an excellent case study for regression analysis and time-series forecasting.

## 📊 Dataset

The dataset (`Walmart.csv`) contains 6,435 records with the following features:

- **Store**: Store number identifier
- **Date**: Week of sales (MM-DD-YYYY format)
- **Weekly_Sales**: Sales for the given store in that week (Target Variable)
- **Holiday_Flag**: Binary indicator (1 if special holiday week, 0 otherwise)
- **Temperature**: Average temperature in the region (°F)
- **Fuel_Price**: Cost of fuel in the region ($/gallon)
- **CPI**: Consumer Price Index
- **Unemployment**: Unemployment rate (%)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone https://github.com/ahmedgalalxxx/Walmart-Sales-Prediction.git
cd Walmart-Sales-Prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Project

1. **Exploratory Data Analysis**: Open and run `01_EDA.ipynb`
2. **Model Training**: Open and run `02_Model_Training.ipynb`
3. **Make Predictions**: Use `predict.py` for new predictions

## 📁 Project Structure

```
Walmart-Sales-Prediction/
│
├── Walmart.csv                 # Dataset
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore file
├── README.md                  # Project documentation
│
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory Data Analysis
│   └── 02_Model_Training.ipynb # Model development and evaluation
│
├── src/
│   ├── data_preprocessing.py  # Data cleaning and feature engineering
│   ├── model_training.py      # Model training functions
│   └── model_evaluation.py    # Model evaluation utilities
│
├── models/                    # Trained models (created after training)
│   └── .gitkeep
│
└── results/                   # Visualizations and reports
    └── .gitkeep
```

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Data quality assessment
- Statistical analysis of features
- Visualization of sales patterns
- Correlation analysis
- Holiday impact analysis

### 2. Data Preprocessing
- Date feature extraction (year, month, week, day)
- Handling missing values
- Feature scaling/normalization
- Train-test split

### 3. Feature Engineering
- Time-based features (seasonal patterns)
- Lag features (previous week's sales)
- Rolling statistics (moving averages)
- Interaction features

### 4. Model Development
Multiple regression models are implemented and compared:
- **Linear Regression** (Baseline)
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**

### 5. Model Evaluation
Models are evaluated using:
- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

## 📈 Results

The best performing models and their metrics will be documented here after training.

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning models and utilities
- **XGBoost**: Advanced gradient boosting
- **Jupyter**: Interactive notebooks

## 📊 Key Insights

Key insights from the analysis will be documented here, including:
- Holiday impact on sales
- Seasonal patterns
- Store-specific trends
- Economic factors influence

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Ahmed Galal**
- GitHub: [@ahmedgalalxxx](https://github.com/ahmedgalalxxx)

## 🙏 Acknowledgments

- Dataset source: Walmart Store Sales Forecasting
- Thanks to the open-source community for the amazing tools and libraries
