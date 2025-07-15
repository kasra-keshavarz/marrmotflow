# MarrmotFlow

A Python package for marrmotflow functionality.

## Installation

You can install the package in development mode:

```bash
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from marrmotflow import example_function

# Example usage
result = example_function()
print(result)
```

## Development

### Setting up the development environment

1. Clone the repository:
```bash
git clone https://github.com/kasra-keshavarz/marrmotflow.git
cd marrmotflow
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode with dev dependencies:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

### Running tests

```bash
pytest
```

### Code formatting

```bash
black src/ tests/
```

### Type checking

```bash
mypy src/
```

## License

This project is licensed under the terms specified in the LICENSE file.
