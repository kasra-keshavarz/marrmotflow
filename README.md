# MarrmotFlow

A Python package for building and running [MARRMoT](https://github.com/wknoben/MARRMoT) hydrological models using a model-agnostic approach. The complete documentation is available at [marrmotflow.readthedocs.io](https://marrmotflow.readthedocs.io).

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

### Python API

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

### Command Line Interface

MarrmotFlow includes a command-line interface for running workflows from JSON configuration files:

```bash
# Run a workflow from a JSON configuration file
marrmotflow run --json config.json --output ./results

# Run with verbose output
marrmotflow run --json config.json --output ./results --verbose

# Show help
marrmotflow run --help
```

#### JSON Configuration Format

Create a JSON file with your workflow configuration:

```json
{
    "name": "MyWorkflow",
    "cat": "/path/to/catchment.shp",
    "forcing_vars": {
        "precip": "precipitation",
        "temp": "temperature"
    },
    "forcing_files": "/path/to/forcing/data/",
    "forcing_units": {
        "precip": "mm/day",
        "temp": "degC"
    },
    "pet_method": "hamon",
    "model_number": [7, 37],
    "forcing_time_zone": "UTC",
    "model_time_zone": "America/Edmonton"
}
```

For detailed CLI documentation, see [CLI_DOCUMENTATION.md](CLI_DOCUMENTATION.md).

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
