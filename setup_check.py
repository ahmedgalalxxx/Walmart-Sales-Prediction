"""
Setup and initialization script for Walmart Sales Prediction project.
Run this script to verify everything is set up correctly.
"""

import sys
import os

def check_python_version():
    """Check if Python version is adequate."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.8 or higher is required")
        return False

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'sklearn',
        'xgboost',
        'jupyter',
        'joblib'
    ]
    
    print("\nChecking dependencies...")
    all_installed = True
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_data_file():
    """Check if the dataset exists."""
    print("\nChecking data file...")
    
    if os.path.exists('Walmart.csv'):
        print("✓ Walmart.csv found")
        return True
    else:
        print("✗ Walmart.csv not found")
        return False

def check_directory_structure():
    """Check if required directories exist."""
    print("\nChecking directory structure...")
    
    required_dirs = ['notebooks', 'src', 'models', 'results']
    all_exist = True
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ - NOT FOUND")
            all_exist = False
    
    return all_exist

def check_notebooks():
    """Check if notebook files exist."""
    print("\nChecking notebooks...")
    
    notebooks = [
        'notebooks/01_EDA.ipynb',
        'notebooks/02_Model_Training.ipynb'
    ]
    
    all_exist = True
    
    for notebook in notebooks:
        if os.path.exists(notebook):
            print(f"✓ {notebook}")
        else:
            print(f"✗ {notebook} - NOT FOUND")
            all_exist = False
    
    return all_exist

def check_source_files():
    """Check if source Python files exist."""
    print("\nChecking source files...")
    
    source_files = [
        'src/data_preprocessing.py',
        'src/model_training.py',
        'src/model_evaluation.py'
    ]
    
    all_exist = True
    
    for file in source_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - NOT FOUND")
            all_exist = False
    
    return all_exist

def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Start Jupyter Notebook:")
    print("   jupyter notebook")
    print("\n2. Open and run the notebooks in order:")
    print("   - notebooks/01_EDA.ipynb")
    print("   - notebooks/02_Model_Training.ipynb")
    print("\n3. After training, make predictions:")
    print("   python predict.py --data Walmart.csv --output predictions.csv")
    print("\n" + "="*70)
    print("\nFor detailed instructions, see GETTING_STARTED.md")
    print("="*70)

def main():
    """Main setup verification function."""
    print("="*70)
    print("WALMART SALES PREDICTION - SETUP VERIFICATION")
    print("="*70)
    
    all_checks_passed = True
    
    # Run all checks
    all_checks_passed &= check_python_version()
    all_checks_passed &= check_dependencies()
    all_checks_passed &= check_data_file()
    all_checks_passed &= check_directory_structure()
    all_checks_passed &= check_notebooks()
    all_checks_passed &= check_source_files()
    
    print("\n" + "="*70)
    if all_checks_passed:
        print("✓ ALL CHECKS PASSED - You're ready to go!")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nTo install missing dependencies, run:")
        print("  pip install -r requirements.txt")
    print("="*70)
    
    # Print next steps
    if all_checks_passed:
        print_next_steps()

if __name__ == "__main__":
    main()
