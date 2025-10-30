# Usage Guide

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ahmedgalalxxx/Walmart-Sales-Prediction.git
cd Walmart-Sales-Prediction

# Run installation script (Linux/Mac)
./install.sh

# Or manually:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the Dataset

Download from: https://www.kaggle.com/code/yasserh/walmart-sales-prediction-best-ml-algorithms/input

Place as: `data/raw/Walmart.csv`

### 3. Run the Project

```bash
# Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Run complete pipeline
python main.py
```

## Detailed Usage

### Using the Main Script

The `main.py` script runs the complete pipeline:

```bash
python main.py
```

**What it does:**
1. Loads and preprocesses data
2. Trains 10 ML models
3. Compares performance
4. Saves best model
5. Generates visualizations

**Output:**
- `models/best_model.pkl` - Best trained model
- `results/model_comparison.csv` - Performance metrics
- `figures/*.png` - Visualization plots

### Using Jupyter Notebooks

#### 1. Exploratory Data Analysis

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

**Features:**
- Data overview and statistics
- Missing value analysis
- Distribution plots
- Correlation analysis
- Outlier detection

#### 2. Model Training

```bash
jupyter notebook notebooks/02_model_training.ipynb
```

**Features:**
- Step-by-step preprocessing
- Model training
- Performance comparison
- Predictions and evaluation

### Using Individual Modules

#### Data Preprocessing

```python
from src.data_preprocessing import WalmartDataPreprocessor

# Initialize
preprocessor = WalmartDataPreprocessor()

# Load data
df = preprocessor.load_data('data/raw/Walmart.csv')

# Handle missing values
df = preprocessor.handle_missing_values(df)

# Feature engineering
df = preprocessor.feature_engineering(df)

# Encode categorical features
categorical_cols = ['Store', 'Holiday_Flag']
df = preprocessor.encode_categorical_features(df, categorical_cols)

# Prepare train-test split
X_train, X_test, y_train, y_test = preprocessor.prepare_data(
    df, 
    target_column='Weekly_Sales',
    test_size=0.2
)
```

#### Model Training

```python
from src.model_training import WalmartModelTrainer

# Initialize
trainer = WalmartModelTrainer()

# Initialize models
trainer.initialize_models()

# Train all models
trainer.train_all_models(X_train, y_train, X_test, y_test)

# Get results
results_df = trainer.get_results_dataframe()
print(results_df)

# Save best model
trainer.save_best_model('models/best_model.pkl')
```

#### Making Predictions

```python
from src.predict import WalmartPredictor

# Load saved model
predictor = WalmartPredictor('models/best_model.pkl')

# Single prediction
features = {
    'Store': 1,
    'Holiday_Flag': 0,
    'Temperature': 42.5,
    'Fuel_Price': 2.57,
    'CPI': 211.096,
    'Unemployment': 8.106
}
prediction = predictor.predict_single(features)
print(f"Predicted Sales: ${prediction:.2f}")

# Batch prediction
predictions = predictor.predict_batch(X_test)

# Predict from CSV
results = predictor.predict_from_csv(
    'data/new_data.csv',
    'data/predictions.csv'
)
```

#### Visualization

```python
from src.visualization import WalmartVisualizer

# Initialize
visualizer = WalmartVisualizer(save_dir='figures')

# Plot correlation matrix
visualizer.plot_correlation_matrix(df, save_name='correlation.png')

# Plot predictions vs actual
visualizer.plot_predictions_vs_actual(
    y_test, 
    predictions,
    save_name='predictions.png'
)

# Plot model comparison
visualizer.plot_model_comparison(
    results_df,
    save_name='comparison.png'
)

# Plot feature importance
visualizer.plot_feature_importance(
    model,
    feature_names,
    save_name='importance.png'
)
```

## Advanced Usage

### Custom Model Configuration

Edit `src/model_training.py` to customize models:

```python
def initialize_models(self):
    self.models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,  # Increase trees
            max_depth=15,      # Limit depth
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        # Add more models...
    }
```

### Custom Preprocessing

Add custom preprocessing steps:

```python
def custom_feature_engineering(self, df):
    """Add custom features"""
    df_custom = df.copy()
    
    # Example: Create interaction features
    df_custom['Temp_Unemployment'] = df_custom['Temperature'] * df_custom['Unemployment']
    
    # Example: Create binned features
    df_custom['Temp_Category'] = pd.cut(
        df_custom['Temperature'],
        bins=[0, 50, 70, 100],
        labels=['Cold', 'Moderate', 'Hot']
    )
    
    return df_custom
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize model
rf = RandomForestRegressor(random_state=42)

# Grid search
grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

# Fit
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

## Troubleshooting

### Issue: Module not found

```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: Dataset not found

```bash
# Check file location
ls -la data/raw/

# Ensure file is named correctly
mv data/raw/your_file.csv data/raw/Walmart.csv
```

### Issue: Out of memory

```python
# Reduce dataset size
df = df.sample(frac=0.5, random_state=42)

# Or use fewer models
trainer.models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor()
}
```

### Issue: Poor model performance

1. **Check data quality**: Look for missing values, outliers
2. **Feature engineering**: Add more relevant features
3. **Hyperparameter tuning**: Optimize model parameters
4. **Feature selection**: Remove irrelevant features
5. **Try ensemble methods**: Combine multiple models

## Command Reference

```bash
# Installation
./install.sh

# Run main pipeline
python main.py

# Run quick start example
python examples/quick_start.py

# Start Jupyter
jupyter notebook

# Run prediction module
python src/predict.py

# Check Python syntax
python -m py_compile src/*.py
```

## Tips and Best Practices

1. **Always use virtual environment**
2. **Check data quality first**
3. **Start with simple models**
4. **Visualize your data before modeling**
5. **Save your models after training**
6. **Document your experiments**
7. **Version control your code**
8. **Test with small datasets first**

## Next Steps

1. Explore the Jupyter notebooks
2. Experiment with different models
3. Try feature engineering
4. Implement hyperparameter tuning
5. Deploy your model
6. Create a web interface
7. Add more datasets

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)

---

For more help, check the [CONTRIBUTING.md](../CONTRIBUTING.md) guide.
