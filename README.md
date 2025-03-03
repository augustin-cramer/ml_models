# ml_models
A Python package for machine learning models.

## Features
- Easy to use.
- Modular and clean structure.

## Installation
```bash
pip install ml_models
```

## Usage
```python
from ml_models import Tree

# Create a Tree instance
model = Tree(max_depth=5)

# Fit the model to data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```