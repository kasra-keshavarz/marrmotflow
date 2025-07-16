Output Handling
===============

MarrmotFlow provides various methods for handling and analyzing model outputs. This guide covers output formats, data structures, and post-processing techniques.

Output Types
------------

Model Outputs
~~~~~~~~~~~~~

MARRMOT models generate several types of outputs:

* **Discharge/Runoff**: Primary model output (m³/s or mm/day)
* **State Variables**: Internal model states (soil moisture, snow storage, etc.)
* **Fluxes**: Water balance components (evapotranspiration, recharge, etc.)
* **Parameters**: Calibrated or default parameter values

Output Formats
~~~~~~~~~~~~~~

MarrmotFlow supports multiple output formats:

* **MATLAB .mat files**: Native MARRMOT format
* **NetCDF files**: Self-describing, CF-compliant format
* **CSV files**: Tabular data for spreadsheet applications
* **Pandas DataFrames**: In-memory Python data structures
* **Xarray Datasets**: Multidimensional labeled arrays

Data Structures
---------------

Model Results Structure
~~~~~~~~~~~~~~~~~~~~~~~

Typical output structure from MarrmotFlow:

.. code-block:: python

   # Example output structure
   model_results = {
       'model_7': {  # HBV-96
           'discharge': xr.DataArray,     # Time series of discharge
           'states': xr.Dataset,         # Model state variables
           'fluxes': xr.Dataset,         # Water balance fluxes
           'parameters': dict,           # Model parameters
           'metadata': dict              # Run information
       },
       'model_37': {  # GR4J
           'discharge': xr.DataArray,
           'states': xr.Dataset,
           'fluxes': xr.Dataset,
           'parameters': dict,
           'metadata': dict
       }
   }

Time Series Outputs
~~~~~~~~~~~~~~~~~~~

Discharge time series example:

.. code-block:: python

   import xarray as xr
   import pandas as pd

   # Example discharge output
   discharge = xr.DataArray(
       data=[1.2, 1.5, 2.1, 1.8, 1.4],  # Discharge values
       dims=['time'],
       coords={
           'time': pd.date_range('2020-01-01', periods=5, freq='D')
       },
       attrs={
           'units': 'mm/day',
           'long_name': 'Daily discharge',
           'model': 'HBV-96',
           'catchment_id': 'basin_001'
       }
   )

State Variables
~~~~~~~~~~~~~~~

Model state variables output:

.. code-block:: python

   # Example state variables
   states = xr.Dataset({
       'soil_moisture': (['time'], [45, 48, 52, 49, 46], {
           'units': 'mm',
           'long_name': 'Soil moisture storage'
       }),
       'snow_storage': (['time'], [0, 0, 5, 12, 8], {
           'units': 'mm',
           'long_name': 'Snow water equivalent'
       }),
       'groundwater': (['time'], [120, 118, 125, 130, 127], {
           'units': 'mm', 
           'long_name': 'Groundwater storage'
       })
   })

Accessing Results
-----------------

Basic Data Access
~~~~~~~~~~~~~~~~~

Access model results after workflow execution:

.. code-block:: python

   # Assuming workflow has been executed
   # results = workflow.get_results()  # Hypothetical method

   # Access discharge for specific model
   discharge_hbv = results['model_7']['discharge']
   discharge_gr4j = results['model_37']['discharge']

   # Access state variables
   soil_moisture = results['model_7']['states']['soil_moisture']

   # Access parameters
   parameters_hbv = results['model_7']['parameters']
   print(f"HBV TT parameter: {parameters_hbv['TT']}")

Time Series Analysis
~~~~~~~~~~~~~~~~~~~

Analyze time series outputs:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   # Plot discharge time series
   fig, ax = plt.subplots(figsize=(12, 6))
   
   discharge_hbv.plot(ax=ax, label='HBV-96', color='blue')
   discharge_gr4j.plot(ax=ax, label='GR4J', color='red')
   
   ax.set_xlabel('Date')
   ax.set_ylabel('Discharge (mm/day)')
   ax.set_title('Model Comparison')
   ax.legend()
   plt.show()

   # Calculate statistics
   stats = {
       'mean': discharge_hbv.mean().item(),
       'std': discharge_hbv.std().item(),
       'max': discharge_hbv.max().item(),
       'min': discharge_hbv.min().item()
   }
   print(f"HBV-96 discharge statistics: {stats}")

Spatial Analysis
~~~~~~~~~~~~~~~~

For multiple catchments:

.. code-block:: python

   # Access results for multiple catchments
   # results_spatial = workflow.get_spatial_results()  # Hypothetical

   # Calculate catchment-averaged discharge
   catchment_discharge = {}
   for catchment_id in results_spatial.keys():
       catchment_discharge[catchment_id] = {
           'mean_discharge': results_spatial[catchment_id]['discharge'].mean(),
           'peak_discharge': results_spatial[catchment_id]['discharge'].max()
       }

Export Options
--------------

Save to NetCDF
~~~~~~~~~~~~~~

Export results to NetCDF format:

.. code-block:: python

   # Save discharge time series
   discharge_hbv.to_netcdf('hbv_discharge.nc')

   # Save complete state variables
   states.to_netcdf('hbv_states.nc')

   # Save with metadata
   discharge_hbv.to_netcdf(
       'discharge_with_metadata.nc',
       encoding={
           'time': {'units': 'days since 1900-01-01'},
           'discharge': {'zlib': True, 'complevel': 9}
       }
   )

Save to CSV
~~~~~~~~~~~

Export to CSV for spreadsheet applications:

.. code-block:: python

   # Convert to pandas DataFrame
   df = discharge_hbv.to_dataframe(name='discharge')
   df.to_csv('hbv_discharge.csv')

   # Multiple variables
   combined_df = pd.DataFrame({
       'discharge_hbv': discharge_hbv.values,
       'discharge_gr4j': discharge_gr4j.values,
       'soil_moisture': soil_moisture.values
   }, index=discharge_hbv.time.values)
   
   combined_df.to_csv('model_comparison.csv')

Save to MATLAB
~~~~~~~~~~~~~~

Export to MATLAB .mat format:

.. code-block:: python

   from scipy.io import savemat

   # Prepare data for MATLAB
   matlab_data = {
       'discharge': discharge_hbv.values,
       'time': [t.timestamp() for t in discharge_hbv.time.values],
       'parameters': parameters_hbv,
       'metadata': {
           'model': 'HBV-96',
           'units': 'mm/day',
           'catchment': 'basin_001'
       }
   }

   # Save to .mat file
   savemat('hbv_results.mat', matlab_data)

Post-Processing
---------------

Model Performance Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate performance metrics:

.. code-block:: python

   def calculate_performance_metrics(observed, simulated):
       """Calculate common hydrological performance metrics."""
       
       # Nash-Sutcliffe Efficiency
       nse = 1 - np.sum((observed - simulated)**2) / np.sum((observed - np.mean(observed))**2)
       
       # Root Mean Square Error
       rmse = np.sqrt(np.mean((observed - simulated)**2))
       
       # Percent Bias
       pbias = 100 * np.sum(simulated - observed) / np.sum(observed)
       
       # Pearson correlation coefficient
       correlation = np.corrcoef(observed, simulated)[0, 1]
       
       return {
           'NSE': nse,
           'RMSE': rmse,
           'PBIAS': pbias,
           'R': correlation
       }

   # Example usage (requires observed data)
   # observed_discharge = load_observed_data()
   # metrics = calculate_performance_metrics(observed_discharge, discharge_hbv.values)
   # print(f"Model performance: {metrics}")

Water Balance Analysis
~~~~~~~~~~~~~~~~~~~~~

Analyze water balance components:

.. code-block:: python

   def analyze_water_balance(precipitation, evapotranspiration, discharge, storage_change):
       """Analyze water balance closure."""
       
       # Water balance equation: P = ET + Q + ΔS
       balance_error = precipitation - evapotranspiration - discharge - storage_change
       
       # Calculate relative error
       relative_error = balance_error / precipitation * 100
       
       # Annual totals
       annual_totals = {
           'precipitation': np.sum(precipitation),
           'evapotranspiration': np.sum(evapotranspiration),
           'discharge': np.sum(discharge),
           'storage_change': np.sum(storage_change),
           'balance_error': np.sum(balance_error)
       }
       
       return {
           'daily_error': balance_error,
           'relative_error': relative_error,
           'annual_totals': annual_totals
       }

Seasonal Analysis
~~~~~~~~~~~~~~~~~

Analyze seasonal patterns:

.. code-block:: python

   def seasonal_analysis(discharge_ts):
       """Analyze seasonal discharge patterns."""
       
       # Convert to DataFrame for easier grouping
       df = discharge_ts.to_dataframe(name='discharge')
       df['month'] = df.index.month
       df['season'] = df['month'].map({
           12: 'Winter', 1: 'Winter', 2: 'Winter',
           3: 'Spring', 4: 'Spring', 5: 'Spring',
           6: 'Summer', 7: 'Summer', 8: 'Summer',
           9: 'Fall', 10: 'Fall', 11: 'Fall'
       })
       
       # Calculate seasonal statistics
       seasonal_stats = df.groupby('season')['discharge'].agg([
           'mean', 'std', 'min', 'max', 'count'
       ])
       
       return seasonal_stats

   # Usage
   seasonal_patterns = seasonal_analysis(discharge_hbv)
   print(seasonal_patterns)

Visualization
-------------

Time Series Plots
~~~~~~~~~~~~~~~~~

Create comprehensive time series visualizations:

.. code-block:: python

   import matplotlib.pyplot as plt
   import matplotlib.dates as mdates

   def plot_model_comparison(results, start_date=None, end_date=None):
       """Plot comparison of multiple models."""
       
       fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
       
       # Discharge comparison
       for model_name, model_results in results.items():
           discharge = model_results['discharge']
           if start_date and end_date:
               discharge = discharge.sel(time=slice(start_date, end_date))
           discharge.plot(ax=axes[0], label=model_name, alpha=0.8)
       
       axes[0].set_title('Discharge Comparison')
       axes[0].set_ylabel('Discharge (mm/day)')
       axes[0].legend()
       axes[0].grid(True, alpha=0.3)
       
       # State variables (example: soil moisture)
       for model_name, model_results in results.items():
           if 'soil_moisture' in model_results['states']:
               soil_moisture = model_results['states']['soil_moisture']
               if start_date and end_date:
                   soil_moisture = soil_moisture.sel(time=slice(start_date, end_date))
               soil_moisture.plot(ax=axes[1], label=model_name, alpha=0.8)
       
       axes[1].set_title('Soil Moisture Storage')
       axes[1].set_ylabel('Storage (mm)')
       axes[1].legend()
       axes[1].grid(True, alpha=0.3)
       
       # Cumulative discharge
       for model_name, model_results in results.items():
           discharge = model_results['discharge']
           if start_date and end_date:
               discharge = discharge.sel(time=slice(start_date, end_date))
           cumulative = discharge.cumsum()
           cumulative.plot(ax=axes[2], label=model_name, alpha=0.8)
       
       axes[2].set_title('Cumulative Discharge')
       axes[2].set_ylabel('Cumulative (mm)')
       axes[2].set_xlabel('Date')
       axes[2].legend()
       axes[2].grid(True, alpha=0.3)
       
       # Format x-axis
       axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
       axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
       plt.xticks(rotation=45)
       
       plt.tight_layout()
       plt.show()

Error Handling
--------------

Output Validation
~~~~~~~~~~~~~~~~~

Validate model outputs:

.. code-block:: python

   def validate_outputs(results):
       """Validate model output data quality."""
       
       issues = []
       
       for model_name, model_data in results.items():
           discharge = model_data['discharge']
           
           # Check for missing values
           if discharge.isnull().any():
               issues.append(f"{model_name}: Contains missing values")
           
           # Check for negative discharge
           if (discharge < 0).any():
               issues.append(f"{model_name}: Contains negative discharge values")
           
           # Check for unrealistic values
           if (discharge > 1000).any():  # > 1000 mm/day
               issues.append(f"{model_name}: Contains unrealistically high discharge")
           
           # Check time continuity
           time_diff = discharge.time.diff('time')
           if not (time_diff == time_diff[0]).all():
               issues.append(f"{model_name}: Irregular time spacing")
       
       return issues

   # Usage
   # validation_issues = validate_outputs(results)
   # if validation_issues:
   #     print("Output validation issues:")
   #     for issue in validation_issues:
   #         print(f"  - {issue}")

Best Practices
--------------

1. **Always validate outputs** before analysis
2. **Use appropriate file formats** for your use case
3. **Include comprehensive metadata** in saved files
4. **Document units and conventions** clearly
5. **Save intermediate results** for reproducibility
6. **Use version control** for output processing scripts
7. **Create standardized visualization** templates
8. **Archive important results** with clear naming conventions
