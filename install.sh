#!/bin/bash

# Installation script for Walmart Sales Prediction project

echo "=========================================="
echo "Walmart Sales Prediction - Installation"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -eq 0 ]; then
    echo "✅ Virtual environment created"
else
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p data/raw data/processed models results figures

echo ""
echo "=========================================="
echo "✅ Installation completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Download dataset from: https://www.kaggle.com/code/yasserh/walmart-sales-prediction-best-ml-algorithms/input"
echo "  3. Place dataset in: data/raw/Walmart.csv"
echo "  4. Run the project: python main.py"
echo ""
echo "For Windows users:"
echo "  - Activate venv: venv\\Scripts\\activate"
echo "=========================================="
