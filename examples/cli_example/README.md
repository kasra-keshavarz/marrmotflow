# CLI Example for MARRMOTFlow

This example demonstrates how to use the MARRMOTFlow CLI to run a hydrological modeling workflow.

## Files in this example:

1. `workflow_config.json` - Configuration file for the workflow
2. `README.md` - This file with instructions

## Usage

### 1. Create a configuration file

Create a JSON file named `workflow_config.json`:

```json
{
    "name": "ExampleWorkflow",
    "cat": "/path/to/your/catchment.shp",
    "forcing_vars": {
        "precip": "precipitation",
        "temp": "temperature"
    },
    "forcing_files": "/path/to/your/forcing/data/",
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

### 2. Run the workflow

```bash
# Basic usage
marrmotflow run --json workflow_config.json

# With custom output directory
marrmotflow run --json workflow_config.json --output ./my_results

# With verbose output
marrmotflow run --json workflow_config.json --output ./my_results --verbose
```

### 3. Expected outputs

The CLI will create the following files in the output directory:
- `marrmot_data.mat` - MATLAB data file with forcing data and PET
- `marrmot_model.m` - MATLAB script file with the model code

## Configuration Parameters

### Required Parameters:
- `name`: Workflow identifier
- `cat`: Path to catchment shapefile
- `forcing_vars`: Mapping of variable types to their names in forcing files
- `forcing_files`: Path to forcing data files (NetCDF format)

### Optional Parameters:
- `forcing_units`: Units for forcing variables (default: auto-detected)
- `pet_method`: PET calculation method (default: "hamon")
- `model_number`: MARRMOT model number(s) (default: [7, 37])
- `forcing_time_zone`: Time zone of forcing data (default: "UTC")
- `model_time_zone`: Time zone for model execution (default: auto-detected)

## Error Handling

The CLI provides clear error messages for common issues:

- **File not found**: Check paths in your configuration
- **Invalid JSON**: Verify your JSON syntax
- **Missing required parameters**: Ensure all required fields are present
- **Workflow execution errors**: Check data formats and file contents

## Tips

1. Use absolute paths in configuration files
2. Ensure forcing files are in NetCDF format
3. Check that variable names in `forcing_vars` match your data files
4. Verify catchment file has proper coordinate reference system
5. Use verbose mode (`-v`) for detailed output during troubleshooting
