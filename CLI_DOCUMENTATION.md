# MARRMOTFlow CLI Documentation

## Overview

The MARRMOTFlow CLI provides a command-line interface for running MARRMOT hydrological model workflows from JSON configuration files. This tool allows you to easily execute workflows without writing Python code.

## Installation

After installing the package, the CLI is available as the `marrmotflow` command:

```bash
pip install marrmotflow
```

## Usage

### Basic Usage

```bash
marrmotflow --json config.json --output-path ./results
```

### Command Options

- `--json PATH`: (Required) Path to JSON configuration file containing workflow parameters
- `--output-path PATH`: Output directory path for saving results (default: `./marrmot_output`)
- `-v, --verbose`: Enable verbose output
- `--version`: Show version information
- `--help`: Show help message and exit

### Examples

#### Basic workflow execution:
```bash
marrmotflow --json my_config.json
```

#### Workflow with custom output directory:
```bash
marrmotflow --json my_config.json --output-path /path/to/results
```

#### Workflow with verbose output:
```bash
marrmotflow --json my_config.json --output-path ./results --verbose
```

## JSON Configuration File Format

The JSON configuration file must contain the parameters needed to initialize a MARRMOTWorkflow instance. Here's an example:

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

### Required Parameters

- `name`: String identifier for the workflow
- `cat`: Path to catchment shapefile (GeoDataFrame or PathLike)
- `forcing_vars`: Dictionary mapping variable types to their names in the forcing files
  - Must include `"precip"` and `"temp"` keys
- `forcing_files`: Path to directory containing forcing files or list of file paths

### Optional Parameters

- `forcing_units`: Dictionary specifying units for forcing variables (default: auto-detected)
- `pet_method`: Method for calculating potential evapotranspiration (default: "hamon")
- `model_number`: Model number(s) to use (default: [7, 37] for HBV-96 and GR4J)
- `forcing_time_zone`: Time zone of forcing data (default: "UTC")
- `model_time_zone`: Time zone for model execution (default: auto-detected from catchment)

## Output Files

The CLI saves two main files to the output directory:

1. `marrmot_data.mat`: MATLAB format file containing:
   - Workflow name
   - Gauge ID
   - Dates as MATLAB datenum
   - Precipitation data
   - Temperature data
   - Potential evapotranspiration
   - Time step (delta_t)

2. `marrmot_model.m`: MATLAB script file containing the model code

## Error Handling

The CLI provides clear error messages for common issues:

- **Configuration file not found**: Verify the path to your JSON file
- **Invalid configuration**: Check that all required parameters are present and valid
- **Workflow execution failed**: Review the error message for specific issues with data processing

## Environment Variables

The JSON configuration supports environment variable substitution using the format `$ENV_VAR_NAME/`. For example:

```json
{
    "cat": "$HOME/data/catchment.shp",
    "forcing_files": "$PROJECT_ROOT/forcing/"
}
```

## Tips

1. **Use absolute paths** in your configuration files to avoid path resolution issues
2. **Check time zones** carefully, especially when working with data from different regions
3. **Enable verbose mode** (`-v`) when troubleshooting to see detailed execution information
4. **Ensure forcing files are in NetCDF format** and contain the variables specified in `forcing_vars`
5. **Verify catchment file has proper CRS** (coordinate reference system) information

## Command Reference

### Main Command

The main `marrmotflow` command runs a MARRMOTWorkflow from a JSON configuration file.

**Usage:**
```bash
marrmotflow [OPTIONS]
```

**Options:**
- `--json PATH`: Path to JSON configuration file [required]
- `--output-path PATH`: Output directory path [default: ./marrmot_output]
- `-v, --verbose`: Enable verbose output
- `--version`: Show version information
- `--help`: Show command help
