# Getting Started with Walmart Sales Prediction

This guide will help you set up and run the Walmart Sales Prediction project.

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

## 🚀 Installation Steps

### 1. Install Required Packages

Open PowerShell in the project directory and run:

```powershell
pip install -r requirements.txt
```

This will install all necessary dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- jupyter
- joblib

### 2. Verify Installation

Check that Jupyter is installed:

```powershell
jupyter --version
```

## 📊 Running the Project

### Option 1: Interactive Analysis with Jupyter Notebooks (Recommended)

#### Step 1: Start Jupyter Notebook

```powershell
jupyter notebook
```

This will open Jupyter in your web browser.

#### Step 2: Exploratory Data Analysis

1. Navigate to `notebooks/01_EDA.ipynb`
2. Click on the notebook to open it
3. Run cells sequentially by pressing `Shift + Enter`
4. This will:
   - Load and explore the data
   - Visualize sales patterns
   - Analyze correlations
   - Examine holiday impacts
   - Generate key insights

#### Step 3: Model Training and Evaluation

1. Navigate to `notebooks/02_Model_Training.ipynb`
2. Open and run the notebook
3. This will:
   - Preprocess the data
   - Engineer features
   - Train multiple models
   - Evaluate performance
   - Save the best model

### Option 2: Using Python Scripts Directly

#### Run Data Preprocessing

```powershell
cd src
python data_preprocessing.py
```

#### Run Model Training

```powershell
python model_training.py
```

## 🔮 Making Predictions

### On New Data

After training models, you can make predictions:

```powershell
python predict.py --data Walmart.csv --model models/xgboost_model.joblib --output predictions.csv
```

### Programmatic Usage

```python
from predict import WalmartSalesPredictor

# Initialize predictor
predictor = WalmartSalesPredictor(
    model_path='models/xgboost_model.joblib',
    scaler_path='models/scaler.joblib',
    feature_names_path='models/feature_names.txt'
)

# Make predictions from file
results = predictor.predict_from_file('Walmart.csv', 'predictions.csv')

# Make single prediction
prediction = predictor.predict_single(
    store=1,
    date='05-02-2010',
    holiday_flag=0,
    temperature=42.31,
    fuel_price=2.572,
    cpi=211.0963582,
    unemployment=8.106
)
```

## 📁 Project Workflow

1. **Data Exploration** (`01_EDA.ipynb`)
   - Understand the dataset
   - Identify patterns and trends
   - Analyze feature relationships

2. **Model Development** (`02_Model_Training.ipynb`)
   - Preprocess and engineer features
   - Train multiple models
   - Compare performance
   - Select best model

3. **Prediction** (`predict.py`)
   - Load trained model
   - Make predictions on new data
   - Evaluate results

## 📈 Expected Results

After running the notebooks, you should have:

- ✅ Comprehensive data visualizations
- ✅ 5 trained machine learning models
- ✅ Performance metrics for each model
- ✅ Feature importance analysis
- ✅ Saved models ready for deployment
- ✅ Evaluation report

## 🛠️ Troubleshooting

### Issue: Module not found error

**Solution**: Make sure you're in the correct directory and have installed all requirements:

```powershell
pip install -r requirements.txt
```

### Issue: Jupyter kernel not starting

**Solution**: Restart Jupyter:

```powershell
jupyter notebook --NotebookApp.iopub_data_rate_limit=1e10
```

### Issue: Memory errors with large datasets

**Solution**: Reduce the data size or increase system memory. You can also modify the code to process data in chunks.

### Issue: Import errors in notebooks

**Solution**: Make sure the `sys.path.append('../src')` line is executed in the notebook cells.

## 💡 Tips for Best Results

1. **Run notebooks in order**: Start with `01_EDA.ipynb`, then `02_Model_Training.ipynb`

2. **Experiment with parameters**: Try different:
   - Lag periods
   - Rolling window sizes
   - Model hyperparameters

3. **Feature engineering**: Add custom features based on domain knowledge

4. **Model tuning**: Use the hyperparameter tuning functions in `model_training.py`

5. **Cross-validation**: Implement time-series cross-validation for better evaluation

## 📚 Next Steps

After completing the basic workflow:

1. **Hyperparameter Tuning**: Fine-tune the best models
2. **Feature Selection**: Identify and remove redundant features
3. **Ensemble Methods**: Combine multiple models
4. **Deployment**: Create a web API for the model
5. **Monitoring**: Set up model performance tracking

## 🤝 Getting Help

If you encounter issues:

1. Check the README.md for detailed documentation
2. Review the code comments in each module
3. Examine error messages carefully
4. Review the Jupyter notebook outputs

## 🎯 Quick Start Command Sequence

Here's a complete sequence to get started quickly:

```powershell
# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook

# In Jupyter: Open and run notebooks/01_EDA.ipynb
# In Jupyter: Open and run notebooks/02_Model_Training.ipynb

# Make predictions (after training)
python predict.py --data Walmart.csv --output predictions.csv
```

Enjoy building your Walmart sales prediction models! 🎉
