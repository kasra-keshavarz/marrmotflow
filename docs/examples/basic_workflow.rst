Basic Workflow Example
======================

This example demonstrates how to set up and run a basic MarrmotFlow workflow with a single catchment and simple forcing data.

Overview
--------

In this example, we will:

1. Load catchment boundary data
2. Prepare meteorological forcing data
3. Create a MarrmotFlow workflow
4. Configure model parameters
5. Analyze results

Prerequisites
-------------

Make sure you have the following data files:

* Catchment boundary shapefile (``basin.shp``)
* Climate data in NetCDF format (``climate_data.nc``)

.. code-block:: bash

   # Example data structure
   data/
   ├── basin.shp
   ├── basin.shx
   ├── basin.dbf
   ├── basin.prj
   └── climate_data.nc

Step 1: Import Libraries
------------------------

.. code-block:: python

   import geopandas as gpd
   import xarray as xr
   import pandas as pd
   import matplotlib.pyplot as plt
   from marrmotflow import MARRMOTWorkflow

Step 2: Load and Inspect Data
-----------------------------

Load Catchment Data
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load catchment boundary
   catchment = gpd.read_file("data/basin.shp")
   
   # Inspect the data
   print("Catchment Information:")
   print(f"Number of catchments: {len(catchment)}")
   print(f"CRS: {catchment.crs}")
   print(f"Columns: {list(catchment.columns)}")
   print(f"Bounds: {catchment.total_bounds}")
   
   # Plot catchment
   fig, ax = plt.subplots(figsize=(8, 6))
   catchment.plot(ax=ax, facecolor='lightblue', edgecolor='black')
   ax.set_title('Study Catchment')
   plt.show()

Load Climate Data
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load climate data
   climate = xr.open_dataset("data/climate_data.nc")
   
   # Inspect the data
   print("Climate Data Information:")
   print(climate)
   print(f"Time range: {climate.time.min().item()} to {climate.time.max().item()}")
   print(f"Variables: {list(climate.data_vars)}")
   
   # Check if data covers catchment
   lon_min, lat_min, lon_max, lat_max = catchment.total_bounds
   print(f"Catchment bounds: [{lon_min:.2f}, {lat_min:.2f}] to [{lon_max:.2f}, {lat_max:.2f}]")
   print(f"Climate lon range: [{climate.lon.min().item():.2f}, {climate.lon.max().item():.2f}]")
   print(f"Climate lat range: [{climate.lat.min().item():.2f}, {climate.lat.max().item():.2f}]")

Visualize Climate Data
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot sample climate data
   fig, axes = plt.subplots(2, 1, figsize=(12, 8))
   
   # Plot precipitation time series for a sample point
   sample_precip = climate.precipitation.isel(lon=0, lat=0)
   sample_precip.plot(ax=axes[0])
   axes[0].set_title('Sample Precipitation Time Series')
   axes[0].set_ylabel('Precipitation (mm/day)')
   
   # Plot temperature time series
   sample_temp = climate.temperature.isel(lon=0, lat=0)
   sample_temp.plot(ax=axes[1])
   axes[1].set_title('Sample Temperature Time Series')
   axes[1].set_ylabel('Temperature (°C)')
   
   plt.tight_layout()
   plt.show()

Step 3: Create MarrmotFlow Workflow
-----------------------------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define forcing variable mapping
   forcing_vars = {
       "precip": "precipitation",  # Map to variable name in NetCDF
       "temp": "temperature"
   }
   
   # Define units (if different from defaults)
   forcing_units = {
       "precip": "mm/day",
       "temp": "celsius"
   }
   
   # Create workflow
   workflow = MARRMOTWorkflow(
       name="BasicExample",
       cat=catchment,
       forcing_files="data/climate_data.nc",
       forcing_vars=forcing_vars,
       forcing_units=forcing_units,
       pet_method="penman_monteith",
       model_number=7,  # HBV-96 model
       forcing_time_zone="UTC",
       model_time_zone="America/Vancouver"
   )
   
   print("Workflow created successfully!")
   print(f"Workflow name: {workflow.name}")

Step 4: Run Workflow (Conceptual)
---------------------------------

.. note::
   The actual execution methods depend on the current implementation of MarrmotFlow.
   This shows the conceptual approach.

.. code-block:: python

   # Execute workflow (implementation dependent)
   # results = workflow.run()
   
   # For this example, we'll demonstrate expected workflow
   print("Workflow configuration:")
   print(f"- Model: HBV-96 (model {workflow.model_number})")
   print(f"- PET method: {workflow.pet_method}")
   print(f"- Catchments: {len(workflow.cat)}")
   print(f"- Forcing variables: {workflow.forcing_vars}")

Step 5: Analyze Configuration
-----------------------------

Model Parameters
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Access default model configuration
   from marrmotflow._default_dicts import default_model_dict
   
   hbv_config = default_model_dict[7]
   print(f"Model name: {hbv_config['name']}")
   print(f"Description: {hbv_config['description']}")
   
   print("\\nModel parameters:")
   for param_name, param_info in hbv_config["parameters"].items():
       print(f"  {param_name}: {param_info['default']} "
             f"(range: {param_info['range']}) - {param_info['description']}")

Data Summary
~~~~~~~~~~~~

.. code-block:: python

   # Analyze forcing data
   precip_stats = climate.precipitation.mean(dim=['lon', 'lat'])
   temp_stats = climate.temperature.mean(dim=['lon', 'lat'])
   
   print("Climate Data Summary:")
   print(f"Mean annual precipitation: {precip_stats.sum().item():.1f} mm")
   print(f"Mean temperature: {temp_stats.mean().item():.1f} °C")
   print(f"Min temperature: {temp_stats.min().item():.1f} °C")
   print(f"Max temperature: {temp_stats.max().item():.1f} °C")

Step 6: Visualization and Validation
------------------------------------

Plot Forcing Data
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create comprehensive plots
   fig, axes = plt.subplots(3, 2, figsize=(15, 12))
   
   # Precipitation time series
   precip_ts = climate.precipitation.mean(dim=['lon', 'lat'])
   precip_ts.plot(ax=axes[0, 0])
   axes[0, 0].set_title('Area-averaged Precipitation')
   axes[0, 0].set_ylabel('Precipitation (mm/day)')
   
   # Temperature time series  
   temp_ts = climate.temperature.mean(dim=['lon', 'lat'])
   temp_ts.plot(ax=axes[0, 1])
   axes[0, 1].set_title('Area-averaged Temperature')
   axes[0, 1].set_ylabel('Temperature (°C)')
   
   # Monthly precipitation climatology
   monthly_precip = precip_ts.groupby('time.month').mean()
   monthly_precip.plot(ax=axes[1, 0], marker='o')
   axes[1, 0].set_title('Monthly Precipitation Climatology')
   axes[1, 0].set_xlabel('Month')
   axes[1, 0].set_ylabel('Precipitation (mm/day)')
   
   # Monthly temperature climatology
   monthly_temp = temp_ts.groupby('time.month').mean()
   monthly_temp.plot(ax=axes[1, 1], marker='o', color='red')
   axes[1, 1].set_title('Monthly Temperature Climatology')
   axes[1, 1].set_xlabel('Month')
   axes[1, 1].set_ylabel('Temperature (°C)')
   
   # Precipitation spatial pattern (annual mean)
   annual_precip = climate.precipitation.mean(dim='time')
   im1 = annual_precip.plot(ax=axes[2, 0], cmap='Blues')
   axes[2, 0].set_title('Annual Mean Precipitation')
   
   # Temperature spatial pattern (annual mean)
   annual_temp = climate.temperature.mean(dim='time')
   im2 = annual_temp.plot(ax=axes[2, 1], cmap='RdYlBu_r')
   axes[2, 1].set_title('Annual Mean Temperature')
   
   plt.tight_layout()
   plt.show()

Data Quality Checks
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check for missing data
   precip_missing = climate.precipitation.isnull().sum()
   temp_missing = climate.temperature.isnull().sum()
   
   print("Data Quality Assessment:")
   print(f"Missing precipitation values: {precip_missing.item()}")
   print(f"Missing temperature values: {temp_missing.item()}")
   
   # Check for unrealistic values
   negative_precip = (climate.precipitation < 0).sum()
   extreme_temp = ((climate.temperature < -50) | (climate.temperature > 50)).sum()
   
   print(f"Negative precipitation values: {negative_precip.item()}")
   print(f"Extreme temperature values: {extreme_temp.item()}")
   
   # Data coverage check
   time_coverage = len(climate.time)
   expected_days = (climate.time[-1] - climate.time[0]).dt.days.item() + 1
   coverage_percent = (time_coverage / expected_days) * 100
   
   print(f"Temporal coverage: {coverage_percent:.1f}% ({time_coverage}/{expected_days} days)")

Step 7: Expected Results Structure
----------------------------------

.. code-block:: python

   # Example of expected results structure after workflow execution
   expected_results = {
       'model_7': {
           'discharge': 'Daily discharge time series (mm/day)',
           'states': {
               'snow_storage': 'Snow water equivalent (mm)',
               'soil_moisture': 'Soil moisture storage (mm)',
               'groundwater': 'Groundwater storage (mm)'
           },
           'fluxes': {
               'evapotranspiration': 'Actual ET (mm/day)',
               'snowmelt': 'Snow melt rate (mm/day)',
               'recharge': 'Groundwater recharge (mm/day)'
           },
           'parameters': {
               'TT': 'Temperature threshold (°C)',
               'C0': 'Degree-day factor (mm/°C/day)',
               # ... other parameters
           },
           'performance': {
               'water_balance_error': 'Daily water balance closure (%)',
               'annual_totals': {
                   'precipitation': 'mm/year',
                   'evapotranspiration': 'mm/year', 
                   'discharge': 'mm/year'
               }
           }
       }
   }
   
   print("Expected results structure:")
   for key, value in expected_results['model_7'].items():
       if isinstance(value, dict):
           print(f"{key}:")
           for subkey, subvalue in value.items():
               print(f"  {subkey}: {subvalue}")
       else:
           print(f"{key}: {value}")

Complete Example Script
-----------------------

.. code-block:: python

   #!/usr/bin/env python3
   """
   Complete basic MarrmotFlow workflow example
   """
   
   import geopandas as gpd
   import xarray as xr
   import matplotlib.pyplot as plt
   from marrmotflow import MARRMOTWorkflow
   
   def main():
       # Load data
       print("Loading data...")
       catchment = gpd.read_file("data/basin.shp")
       climate = xr.open_dataset("data/climate_data.nc")
       
       # Validate data
       print("Validating data...")
       assert len(catchment) > 0, "No catchments found"
       assert 'precipitation' in climate.data_vars, "Precipitation not found"
       assert 'temperature' in climate.data_vars, "Temperature not found"
       
       # Create workflow
       print("Creating workflow...")
       workflow = MARRMOTWorkflow(
           name="BasicExample",
           cat=catchment,
           forcing_files="data/climate_data.nc",
           forcing_vars={
               "precip": "precipitation",
               "temp": "temperature"
           },
           forcing_units={
               "precip": "mm/day",
               "temp": "celsius"
           },
           pet_method="penman_monteith",
           model_number=7
       )
       
       print(f"Workflow '{workflow.name}' created successfully!")
       print(f"Ready to run HBV-96 model for {len(catchment)} catchment(s)")
       
       # Additional analysis could be added here
       
   if __name__ == "__main__":
       main()

Next Steps
----------

After completing this basic example, you can:

1. **Explore multi-model comparisons**: See :doc:`multi_model_comparison`
2. **Add observed data**: Compare model results with observations
3. **Calibrate parameters**: Optimize model parameters for your catchment
4. **Extend spatial analysis**: Work with multiple catchments
5. **Climate change studies**: Use future climate projections

Troubleshooting
---------------

Common issues and solutions:

**File not found errors:**

.. code-block:: python

   import os
   print(f"Current directory: {os.getcwd()}")
   print(f"Files in data/: {os.listdir('data') if os.path.exists('data') else 'data/ not found'}")

**CRS mismatch:**

.. code-block:: python

   # Ensure consistent coordinate systems
   if catchment.crs != 'EPSG:4326':
       catchment = catchment.to_crs('EPSG:4326')

**Data type issues:**

.. code-block:: python

   # Ensure time is properly formatted
   climate['time'] = pd.to_datetime(climate.time)

**Memory issues with large datasets:**

.. code-block:: python

   # Use dask for large datasets
   climate = xr.open_dataset("data/climate_data.nc", chunks={'time': 365})
