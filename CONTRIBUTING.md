# Contributing to Conformer-IEMOCAP-SER

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, PyTorch version, etc.)
- Any relevant logs or screenshots

### Suggesting Features

We welcome feature suggestions! Please open an issue with:
- A clear description of the feature
- Why this feature would be useful
- Any examples or mockups if applicable

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the code style guidelines
   - Add tests if applicable
   - Update documentation

4. **Test your changes**
   ```bash
   python setup_check.py
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add: your feature description"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**

## Code Style Guidelines

### Python

- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to all functions and classes
- Keep functions focused and small
- Use type hints where appropriate

Example:
```python
def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    emotion_labels: List[str]
) -> Dict[str, float]:
    """
    Compute metrics from predictions and labels.
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        emotion_labels: List of emotion label names
        
    Returns:
        Dictionary of metrics
    """
    # Implementation here
    pass
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions/classes
- Include examples in documentation
- Keep documentation clear and concise

### Testing

- Add tests for new features
- Ensure existing tests pass
- Test on multiple environments if possible

## Project Structure

```
CONFORMER-SER/
├── config/          # Configuration files
├── src/
│   ├── data/       # Dataset and preprocessing
│   ├── models/     # Model architectures
│   ├── utils/      # Utilities
│   ├── train.py    # Training script
│   └── evaluate.py # Evaluation script
├── notebooks/      # Jupyter notebooks
└── tests/          # Test files (to be added)
```

## Development Setup

1. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/CONFORMER-SER.git
cd CONFORMER-SER
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install development dependencies (optional):
```bash
pip install black flake8 mypy pytest
```

4. Verify setup:
```bash
python setup_check.py
```

## Commit Message Guidelines

Use clear, descriptive commit messages:

- **Add**: New features or files
- **Fix**: Bug fixes
- **Update**: Changes to existing functionality
- **Refactor**: Code restructuring without functionality changes
- **Docs**: Documentation changes
- **Test**: Adding or updating tests

Examples:
```
Add: Support for RAVDESS dataset
Fix: Memory leak in data loader
Update: Improve training speed with mixed precision
Docs: Add examples to README
```

## Areas for Contribution

We especially welcome contributions in:

1. **Data Augmentation**: SpecAugment, mixing, etc.
2. **Model Architectures**: New fusion strategies, attention mechanisms
3. **Datasets**: Support for additional datasets
4. **Optimization**: Training speed improvements
5. **Documentation**: Tutorials, examples, translations
6. **Testing**: Unit tests, integration tests
7. **Deployment**: Docker, model serving, quantization
8. **Visualization**: Attention maps, feature visualization

## Code Review Process

1. Maintainers will review your PR
2. Feedback may be provided for improvements
3. Once approved, your PR will be merged
4. You'll be credited in the CHANGELOG

## Questions?

Feel free to open an issue for any questions about contributing!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Thank you for contributing to Conformer-IEMOCAP-SER! Every contribution helps make this project better.

