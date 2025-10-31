# Walmart Sales Prediction - Quick Reference

## 🚀 Quick Commands

### Setup
```powershell
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python setup_check.py

# Start Jupyter Notebook
jupyter notebook
```

### Running Analysis
```powershell
# Option 1: Interactive (Recommended)
jupyter notebook
# Then open: notebooks/01_EDA.ipynb
# Then open: notebooks/02_Model_Training.ipynb

# Option 2: Command Line
cd src
python data_preprocessing.py
python model_training.py
```

### Making Predictions
```powershell
# Predict on new data
python predict.py --data Walmart.csv --output predictions.csv

# Use specific model
python predict.py --data Walmart.csv --model models/random_forest_model.joblib --output predictions.csv
```

## 📁 File Guide

| File/Folder | Purpose |
|------------|---------|
| `Walmart.csv` | Your dataset |
| `requirements.txt` | Python packages to install |
| `README.md` | Full project documentation |
| `GETTING_STARTED.md` | Setup instructions |
| `PROJECT_SUMMARY.md` | Complete project overview |
| `setup_check.py` | Verify setup |
| `predict.py` | Make predictions |
| `notebooks/01_EDA.ipynb` | Data exploration |
| `notebooks/02_Model_Training.ipynb` | Train models |
| `src/data_preprocessing.py` | Data processing module |
| `src/model_training.py` | Model training module |
| `src/model_evaluation.py` | Evaluation module |
| `models/` | Saved models go here |
| `results/` | Output reports go here |

## 🎯 Workflow Steps

### 1️⃣ Data Exploration
```python
# In notebooks/01_EDA.ipynb
# - Load data
# - Statistical analysis
# - Visualizations
# - Correlation analysis
# - Seasonal patterns
# - Holiday impact
```

### 2️⃣ Model Training
```python
# In notebooks/02_Model_Training.ipynb
# - Feature engineering
# - Train 5 models
# - Evaluate performance
# - Compare models
# - Save best model
```

### 3️⃣ Predictions
```python
from predict import WalmartSalesPredictor

# Load model
predictor = WalmartSalesPredictor(
    model_path='models/xgboost_model.joblib',
    scaler_path='models/scaler.joblib',
    feature_names_path='models/feature_names.txt'
)

# Predict
results = predictor.predict_from_file('Walmart.csv')
```

## 📊 Available Models

1. **Linear Regression** - Baseline model
2. **Decision Tree** - Non-linear patterns
3. **Random Forest** - Ensemble, robust
4. **Gradient Boosting** - Sequential boosting
5. **XGBoost** - Advanced gradient boosting

## 🎨 Key Visualizations

Generated in notebooks:
- Sales distribution histograms
- Time series trends
- Seasonal patterns (monthly/quarterly)
- Store performance comparison
- Correlation heatmaps
- Prediction vs Actual plots
- Residual plots
- Error distributions
- Feature importance charts

## 📈 Evaluation Metrics

All models evaluated on:
- **R² Score**: Model fit quality (0-1, higher better)
- **MAE**: Average absolute error (dollars)
- **RMSE**: Root mean squared error (dollars)
- **MAPE**: Mean absolute percentage error (%)

## 🔧 Customization Tips

### Add New Features
```python
# In src/data_preprocessing.py
# Add to WalmartDataPreprocessor class
def create_custom_feature(self):
    self.df['new_feature'] = ...
    return self.df
```

### Change Model Parameters
```python
# In src/model_training.py
# Modify initialize_models() method
'Random Forest': RandomForestRegressor(
    n_estimators=200,  # Change this
    max_depth=20,      # Change this
    random_state=42
)
```

### Add New Visualization
```python
# In src/model_evaluation.py
# Add to WalmartModelEvaluator class
def plot_custom_chart(self):
    plt.figure(figsize=(12, 6))
    # Your plotting code
    plt.show()
```

## 🐛 Troubleshooting

### Import Error
```powershell
# Solution
pip install -r requirements.txt
```

### Jupyter Not Starting
```powershell
# Solution 1
jupyter notebook --no-browser

# Solution 2
python -m jupyter notebook
```

### Module Not Found in Notebook
```python
# Add to first cell
import sys
sys.path.append('../src')
```

### Memory Issues
```python
# Use smaller datasets or reduce lag features
preprocessor.create_lag_features(lags=[1])  # Instead of [1,2,4]
```

## 💻 Code Snippets

### Load and Explore Data
```python
import pandas as pd
df = pd.read_csv('Walmart.csv')
print(df.head())
print(df.describe())
```

### Quick Preprocessing
```python
from src.data_preprocessing import WalmartDataPreprocessor

preprocessor = WalmartDataPreprocessor('Walmart.csv')
X_train, X_test, y_train, y_test = preprocessor.full_preprocessing_pipeline()
```

### Train Single Model
```python
from src.model_training import WalmartModelTrainer

trainer = WalmartModelTrainer()
trainer.initialize_models()
model = trainer.train_model('Random Forest', X_train, y_train)
```

### Evaluate Model
```python
from src.model_evaluation import WalmartModelEvaluator

evaluator = WalmartModelEvaluator()
y_pred = model.predict(X_test)
metrics = evaluator.calculate_metrics(y_test, y_pred, 'Random Forest')
print(metrics)
```

## 🎓 Learning Path

1. **Beginner**: 
   - Run notebooks as-is
   - Understand the outputs
   - Read code comments

2. **Intermediate**:
   - Modify parameters
   - Add custom features
   - Try different visualizations

3. **Advanced**:
   - Implement cross-validation
   - Add ensemble methods
   - Build prediction API
   - Deploy to cloud

## 📞 Quick Help

- **Setup Issues**: Check `GETTING_STARTED.md`
- **Project Overview**: Read `PROJECT_SUMMARY.md`
- **Full Documentation**: See `README.md`
- **Code Details**: Review module docstrings

## ✅ Checklist

Before running:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Setup verified (`python setup_check.py`)
- [ ] Jupyter working (`jupyter notebook`)

To complete:
- [ ] Run `01_EDA.ipynb`
- [ ] Run `02_Model_Training.ipynb`
- [ ] Make predictions with `predict.py`
- [ ] Review results and metrics

---

**Happy Coding! 🚀**

For detailed information, see:
- `README.md` - Complete documentation
- `GETTING_STARTED.md` - Setup guide
- `PROJECT_SUMMARY.md` - Project overview
