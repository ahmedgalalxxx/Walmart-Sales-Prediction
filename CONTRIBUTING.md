# Contributing to Walmart Sales Prediction

Thank you for your interest in contributing to this project! We welcome contributions from everyone.

## How to Contribute

### Reporting Issues

- Check if the issue already exists in the [Issues](https://github.com/ahmedgalalxxx/Walmart-Sales-Prediction/issues) section
- Provide a clear and descriptive title
- Include as much relevant information as possible
- Add code samples or screenshots if applicable

### Suggesting Enhancements

- Open an issue with the tag "enhancement"
- Clearly describe the enhancement and its benefits
- Provide examples of how it would work

### Pull Requests

1. **Fork the Repository**
   ```bash
   git clone https://github.com/ahmedgalalxxx/Walmart-Sales-Prediction.git
   cd Walmart-Sales-Prediction
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Follow the existing code style
   - Add comments where necessary
   - Update documentation if needed

4. **Test Your Changes**
   - Ensure all existing tests pass
   - Add new tests if applicable
   - Test with the sample dataset

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your fork and branch
   - Provide a clear description of your changes

## Code Style Guidelines

### Python Code

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Use type hints where applicable

Example:
```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate regression metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing metrics
    """
    # Implementation
    pass
```

### Documentation

- Update README.md if you add new features
- Add inline comments for complex logic
- Update docstrings for modified functions
- Include examples in documentation

### Commit Messages

Use clear and descriptive commit messages:

- `Add: New feature description`
- `Fix: Bug fix description`
- `Update: Changes to existing feature`
- `Docs: Documentation updates`
- `Refactor: Code refactoring`
- `Test: Adding or updating tests`

## Adding New Features

### Adding a New ML Model

1. Add the model to `src/model_training.py`:
```python
def initialize_models(self):
    self.models['Your Model'] = YourModelClass(parameters)
```

2. Ensure it follows the scikit-learn API (fit, predict methods)
3. Test with the sample dataset
4. Update documentation

### Adding New Preprocessing Steps

1. Add methods to `src/data_preprocessing.py`
2. Follow the existing pattern
3. Add error handling
4. Test with various data types

### Adding Visualizations

1. Add visualization functions to `src/visualization.py`
2. Use matplotlib/seaborn
3. Include save_name parameter for saving plots
4. Test with sample data

## Testing

Before submitting a pull request:

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test Your Code**
   ```bash
   python -m py_compile src/*.py
   ```

3. **Run the Main Script**
   ```bash
   python main.py
   ```

4. **Check for Errors**
   - Test with different datasets if possible
   - Verify all functions work as expected
   - Check for any warnings or errors

## Project Structure

```
Walmart-Sales-Prediction/
├── data/                  # Data directory
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── visualization.py
│   └── predict.py
├── examples/              # Example scripts
├── models/                # Saved models
├── results/               # Results and metrics
├── figures/               # Generated plots
└── main.py               # Main execution script
```

## Areas for Contribution

We especially welcome contributions in these areas:

- **New ML Models**: Add more machine learning algorithms
- **Feature Engineering**: Implement new feature engineering techniques
- **Visualization**: Create additional visualization functions
- **Performance**: Optimize code for better performance
- **Documentation**: Improve documentation and examples
- **Testing**: Add unit tests and integration tests
- **Deployment**: Add deployment scripts or Docker support
- **UI**: Create a web interface using Flask/Streamlit

## Questions?

If you have questions, feel free to:
- Open an issue with the "question" tag
- Contact the maintainers

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to make this project better! 🚀
