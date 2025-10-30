# Walmart Sales Prediction - Machine Learning Project

A comprehensive machine learning project for predicting Walmart sales using multiple ML algorithms.

## 📊 Project Overview

This project implements various machine learning algorithms to predict Walmart sales based on historical data. The project includes data preprocessing, exploratory data analysis, multiple model training, and performance comparison.

## 🎯 Features

- **Data Preprocessing**: Automated data cleaning, missing value handling, and feature engineering
- **Multiple ML Algorithms**: 
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - K-Nearest Neighbors (KNN)
  - Support Vector Regression (SVR)
- **Model Evaluation**: Comprehensive metrics (R2 Score, RMSE, MAE, MSE)
- **Visualization**: Interactive plots and charts for data analysis
- **Jupyter Notebooks**: Step-by-step analysis and modeling

## 📁 Project Structure

```
Walmart-Sales-Prediction/
├── data/
│   ├── raw/                    # Raw dataset (download from Kaggle)
│   └── processed/              # Processed data files
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data preprocessing utilities
│   ├── model_training.py       # Model training and evaluation
│   └── visualization.py        # Visualization functions
├── models/                     # Saved trained models
├── results/                    # Model results and comparisons
├── figures/                    # Generated visualizations
├── main.py                     # Main execution script
├── requirements.txt            # Project dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ahmedgalalxxx/Walmart-Sales-Prediction.git
cd Walmart-Sales-Prediction
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
   - Visit: https://www.kaggle.com/code/yasserh/walmart-sales-prediction-best-ml-algorithms/input
   - Download the Walmart sales dataset
   - Place the CSV file in `data/raw/` directory as `Walmart.csv`

## 💻 Usage

### Running the Main Script

Execute the complete pipeline (preprocessing, training, evaluation):

```bash
python main.py
```

This will:
1. Load and preprocess the data
2. Train all ML models
3. Compare model performance
4. Save the best model
5. Generate visualizations

### Using Jupyter Notebooks

1. **Start Jupyter**
```bash
jupyter notebook
```

2. **Open notebooks**
   - `01_exploratory_data_analysis.ipynb` - Data exploration and analysis
   - `02_model_training.ipynb` - Model training and evaluation

### Using Individual Modules

```python
from src.data_preprocessing import WalmartDataPreprocessor
from src.model_training import WalmartModelTrainer
from src.visualization import WalmartVisualizer

# Initialize
preprocessor = WalmartDataPreprocessor()
trainer = WalmartModelTrainer()
visualizer = WalmartVisualizer()

# Load and preprocess data
df = preprocessor.load_data('data/raw/Walmart.csv')
df = preprocessor.handle_missing_values(df)
df = preprocessor.feature_engineering(df)

# Train models
X_train, X_test, y_train, y_test = preprocessor.prepare_data(df, 'Weekly_Sales')
trainer.initialize_models()
trainer.train_all_models(X_train, y_train, X_test, y_test)

# Get results
results_df = trainer.get_results_dataframe()
print(results_df)
```

## 📈 Model Performance

The project trains and compares multiple models. Results are saved in:
- `results/model_comparison.csv` - Detailed performance metrics
- `models/best_model.pkl` - Best performing model

Example output:
```
Model                 R2 Score    RMSE        MAE         MSE
XGBoost              0.9756      1234.56     987.65      1523456.78
Random Forest        0.9723      1345.67     1012.34     1811422.89
Gradient Boosting    0.9698      1456.78     1098.76     2122188.90
...
```

## 📊 Visualizations

Generated visualizations include:
- Model comparison charts
- Predictions vs actual values
- Residual plots
- Feature importance (for tree-based models)
- Correlation matrices
- Distribution plots

All figures are saved in the `figures/` directory.

## 🔧 Customization

### Modify Target Column

Edit `main.py` and update:
```python
TARGET_COLUMN = 'Your_Target_Column'
```

### Add Custom Models

In `src/model_training.py`, add your model to the `initialize_models()` method:
```python
self.models['Your Model'] = YourModelClass()
```

### Adjust Hyperparameters

Modify model parameters in `initialize_models()`:
```python
'Random Forest': RandomForestRegressor(
    n_estimators=200,  # Increased from 100
    max_depth=15,
    random_state=42
)
```

## 📝 Dataset Information

The Walmart sales dataset typically includes:
- **Store**: Store number
- **Date**: Week of sales
- **Weekly_Sales**: Sales for the given store and date
- **Holiday_Flag**: Whether the week is a special holiday week
- **Temperature**: Temperature on the day of sale
- **Fuel_Price**: Cost of fuel in the region
- **CPI**: Consumer Price Index
- **Unemployment**: Unemployment rate

*Note: Actual columns may vary. Check the dataset documentation.*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Ahmed Galal

## 🙏 Acknowledgments

- Dataset source: [Kaggle - Walmart Sales Prediction](https://www.kaggle.com/code/yasserh/walmart-sales-prediction-best-ml-algorithms/input)
- Scikit-learn documentation
- XGBoost and LightGBM communities

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Happy Modeling! 🚀**
