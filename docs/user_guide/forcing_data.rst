Forcing Data
============

Forcing data provides the meteorological inputs required to drive MARRMOT models. This guide covers how to work with different types of forcing data in MarrmotFlow.

Required Variables
------------------

Precipitation
~~~~~~~~~~~~~

Precipitation is a mandatory input for all MARRMOT models.

**Supported units:**
- mm/day (default)
- mm/hour
- m/day
- kg m-2 s-1 (CMIP6 standard)
- inches/day

.. code-block:: python

   forcing_vars = {
       "precip": "precipitation"  # Variable name in your dataset
   }
   
   forcing_units = {
       "precip": "mm/day"  # Units of precipitation data
   }

Temperature
~~~~~~~~~~~

Temperature is required for evapotranspiration calculations and snow processes.

**Supported units:**
- celsius (default)
- kelvin
- fahrenheit

.. code-block:: python

   forcing_vars = {
       "temp": "temperature"  # Variable name in your dataset
   }
   
   forcing_units = {
       "temp": "celsius"  # Units of temperature data
   }

Data Sources
------------

Climate Reanalysis
~~~~~~~~~~~~~~~~~~

Popular reanalysis datasets supported:

**ERA5 (ECMWF)**:

.. code-block:: python

   # ERA5 configuration
   forcing_vars = {
       "precip": "total_precipitation",
       "temp": "2m_temperature"
   }
   
   forcing_units = {
       "precip": "m/day",  # ERA5 uses meters
       "temp": "kelvin"    # ERA5 uses Kelvin
   }

**NCEP/NCAR Reanalysis**:

.. code-block:: python

   # NCEP configuration
   forcing_vars = {
       "precip": "prate",
       "temp": "air"
   }
   
   forcing_units = {
       "precip": "kg m-2 s-1",
       "temp": "kelvin"
   }

Climate Models
~~~~~~~~~~~~~~

**CMIP6 Data**:

.. code-block:: python

   # CMIP6 standard names
   forcing_vars = {
       "precip": "pr",   # precipitation_flux
       "temp": "tas"     # air_temperature
   }
   
   forcing_units = {
       "precip": "kg m-2 s-1",
       "temp": "K"
   }

Observational Data
~~~~~~~~~~~~~~~~~~

**Station Data**:

.. code-block:: python

   # Station observations
   forcing_vars = {
       "precip": "daily_precip",
       "temp": "mean_temp"
   }
   
   forcing_units = {
       "precip": "mm/day",
       "temp": "celsius"
   }

**Gridded Products** (e.g., Daymet, PRISM):

.. code-block:: python

   # Daymet configuration
   forcing_vars = {
       "precip": "prcp",
       "temp": "tmax"  # or "tmin", "tmean"
   }
   
   forcing_units = {
       "precip": "mm/day",
       "temp": "celsius"
   }

File Handling
-------------

Single File
~~~~~~~~~~~

Load data from a single NetCDF file:

.. code-block:: python

   workflow = MARRMOTWorkflow(
       forcing_files="climate_data.nc",
       forcing_vars={"precip": "precipitation", "temp": "temperature"},
       # ... other parameters
   )

Multiple Files
~~~~~~~~~~~~~~

Handle multiple forcing files:

.. code-block:: python

   # Multiple files with same structure
   forcing_files = [
       "climate_2020.nc",
       "climate_2021.nc",
       "climate_2022.nc"
   ]
   
   workflow = MARRMOTWorkflow(
       forcing_files=forcing_files,
       forcing_vars={"precip": "precipitation", "temp": "temperature"},
       # ... other parameters
   )

Different Variables in Different Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When variables are in separate files:

.. code-block:: python

   # Separate files for different variables
   forcing_files = [
       "precipitation_data.nc",
       "temperature_data.nc"
   ]
   
   # Make sure variable names match across files
   forcing_vars = {
       "precip": "precipitation",
       "temp": "temperature"
   }

Time Handling
-------------

Time Zones
~~~~~~~~~~

Specify time zones for proper temporal alignment:

.. code-block:: python

   workflow = MARRMOTWorkflow(
       forcing_time_zone="UTC",        # Time zone of forcing data
       model_time_zone="America/Vancouver",  # Local time zone for analysis
       # ... other parameters
   )

Time Resolution
~~~~~~~~~~~~~~~

MarrmotFlow expects daily time resolution. Higher frequency data will be aggregated:

.. code-block:: python

   # Hourly data will be automatically aggregated to daily
   # Sub-daily aggregation is handled internally
   pass

Temporal Coverage
~~~~~~~~~~~~~~~~~

Ensure your forcing data covers the analysis period:

.. code-block:: python

   import xarray as xr
   
   # Check temporal coverage
   forcing = xr.open_dataset("climate_data.nc")
   print(f"Data period: {forcing.time.min().item()} to {forcing.time.max().item()}")
   print(f"Time steps: {len(forcing.time)}")

Spatial Considerations
----------------------

Spatial Resolution
~~~~~~~~~~~~~~~~~~

Consider the resolution of your forcing data relative to catchment size:

.. code-block:: python

   # Check spatial resolution
   import numpy as np
   
   forcing = xr.open_dataset("climate_data.nc")
   lon_res = np.diff(forcing.lon).mean()
   lat_res = np.diff(forcing.lat).mean()
   
   print(f"Spatial resolution: {lon_res:.3f}° × {lat_res:.3f}°")
   print(f"Approximate resolution: {lon_res*111:.1f} km × {lat_res*111:.1f} km")

Spatial Coverage
~~~~~~~~~~~~~~~~

Verify that forcing data covers all catchments:

.. code-block:: python

   import geopandas as gpd
   import xarray as xr
   
   # Load data
   catchments = gpd.read_file("catchments.shp")
   forcing = xr.open_dataset("climate_data.nc")
   
   # Check coverage
   cat_bounds = catchments.total_bounds  # [minx, miny, maxx, maxy]
   forcing_bounds = [
       forcing.lon.min().item(), forcing.lat.min().item(),
       forcing.lon.max().item(), forcing.lat.max().item()
   ]
   
   print(f"Catchment bounds: {cat_bounds}")
   print(f"Forcing bounds: {forcing_bounds}")

Data Quality Assessment
-----------------------

Missing Values
~~~~~~~~~~~~~~

Check for and handle missing data:

.. code-block:: python

   import xarray as xr
   
   forcing = xr.open_dataset("climate_data.nc")
   
   # Check for missing values
   precip_missing = forcing.precipitation.isnull().sum()
   temp_missing = forcing.temperature.isnull().sum()
   
   print(f"Missing precipitation: {precip_missing.item()} values")
   print(f"Missing temperature: {temp_missing.item()} values")
   
   # Handle missing data
   # Option 1: Drop time steps with missing data
   forcing_clean = forcing.dropna(dim='time')
   
   # Option 2: Interpolate missing values
   forcing_interp = forcing.interpolate_na(dim='time')

Outlier Detection
~~~~~~~~~~~~~~~~~

Identify potential data quality issues:

.. code-block:: python

   # Check for unrealistic values
   
   # Precipitation outliers
   negative_precip = (forcing.precipitation < 0).sum()
   extreme_precip = (forcing.precipitation > 500).sum()  # > 500 mm/day
   
   print(f"Negative precipitation: {negative_precip.item()}")
   print(f"Extreme precipitation (>500mm/day): {extreme_precip.item()}")
   
   # Temperature outliers
   extreme_cold = (forcing.temperature < -60).sum()  # < -60°C
   extreme_hot = (forcing.temperature > 60).sum()    # > 60°C
   
   print(f"Extreme cold (<-60°C): {extreme_cold.item()}")
   print(f"Extreme hot (>60°C): {extreme_hot.item()}")

Advanced Forcing Data Configuration
-----------------------------------

Custom Variable Mapping
~~~~~~~~~~~~~~~~~~~~~~~~

Handle non-standard variable names:

.. code-block:: python

   # Custom mapping for specific datasets
   dataset_configs = {
       "era5": {
           "vars": {"precip": "total_precipitation", "temp": "2m_temperature"},
           "units": {"precip": "m/day", "temp": "kelvin"}
       },
       "cmip6": {
           "vars": {"precip": "pr", "temp": "tas"},
           "units": {"precip": "kg m-2 s-1", "temp": "K"}
       },
       "station": {
           "vars": {"precip": "daily_precip", "temp": "mean_temp"},
           "units": {"precip": "mm/day", "temp": "celsius"}
       }
   }
   
   # Use configuration
   config = dataset_configs["era5"]
   workflow = MARRMOTWorkflow(
       forcing_vars=config["vars"],
       forcing_units=config["units"],
       # ... other parameters
   )

Multiple Data Sources
~~~~~~~~~~~~~~~~~~~~~

Combine different data sources:

.. code-block:: python

   # Example: Use high-quality station data where available,
   # fill gaps with reanalysis data
   forcing_files = [
       "station_data.nc",      # Higher priority
       "reanalysis_data.nc"    # Gap-filling
   ]
   
   # MarrmotFlow will handle data merging internally

Preprocessing Workflow
----------------------

Complete preprocessing example:

.. code-block:: python

   import xarray as xr
   import pandas as pd
   
   def preprocess_forcing_data(input_file, output_file):
       """Comprehensive forcing data preprocessing."""
       
       # Load data
       ds = xr.open_dataset(input_file)
       
       # Standardize time
       ds['time'] = pd.to_datetime(ds.time)
       
       # Handle missing values
       ds = ds.interpolate_na(dim='time', method='linear')
       
       # Quality control
       # Remove negative precipitation
       ds['precipitation'] = ds.precipitation.where(ds.precipitation >= 0, 0)
       
       # Flag extreme values
       temp_range = (-50, 50)  # Reasonable temperature range in Celsius
       ds['temperature'] = ds.temperature.where(
           (ds.temperature >= temp_range[0]) & (ds.temperature <= temp_range[1])
       )
       
       # Save processed data
       ds.to_netcdf(output_file)
       
       print(f"Processed data saved to {output_file}")
       return ds
   
   # Use preprocessing
   processed_data = preprocess_forcing_data("raw_data.nc", "processed_data.nc")

Best Practices
--------------

1. **Validate data quality** before using in workflows
2. **Use consistent time zones** across all datasets
3. **Check spatial and temporal coverage** matches your study domain
4. **Document data sources** and preprocessing steps
5. **Handle missing data appropriately** for your analysis
6. **Consider multiple data sources** for robustness
7. **Test with small subsets** before processing large datasets
8. **Keep original data** and track all preprocessing steps
