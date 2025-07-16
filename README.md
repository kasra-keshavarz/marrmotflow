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
from marrmotflow import MARRMOTWorkflow

# Example usage
config = {
    'cat': 'path/to/catchment.shp', 
    'forcing_files': 'path/to/forcing/files',
    'forcing_vars': {
        "temperature": "temperature_variable_name",
        "precipitation": "precipitation_variable_name",
    },
    'forcing_units': {
        'temperature': 'celsius',
        'precipitation': 'meter / hour',
    },
    'forcing_time_zone': 'UTC',
}

# Build the MARRMOT workflow
marrmot_experiment = MARRMOTWorkflow(**config)

# Run the workflow
marrmot_experiment.run()

# Save the results
marrmot_experiment.save_results('path/to/save/results/directory')
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
