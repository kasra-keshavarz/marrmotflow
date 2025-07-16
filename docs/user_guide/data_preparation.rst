Data Preparation
================

Proper data preparation is crucial for successful MarrmotFlow workflows. This guide covers how to prepare catchment data, forcing data, and other inputs.

Catchment Data
--------------

Format Requirements
~~~~~~~~~~~~~~~~~~~

MarrmotFlow accepts catchment data in various spatial formats:

* Shapefiles (.shp)
* GeoJSON (.geojson)
* GeoPackage (.gpkg)
* Any format supported by GeoPandas/GDAL

Loading Catchment Data
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import geopandas as gpd

   # From shapefile
   catchments = gpd.read_file("catchments.shp")

   # From GeoJSON
   catchments = gpd.read_file("catchments.geojson")

   # From GeoPackage
   catchments = gpd.read_file("catchments.gpkg")

Required Attributes
~~~~~~~~~~~~~~~~~~~

Your catchment data should include:

* **Unique identifiers**: Each catchment should have a unique ID
* **Geometry**: Polygon geometries defining catchment boundaries
* **Optional metadata**: Name, area, elevation, etc.

.. code-block:: python

   # Example catchment structure
   print(catchments.columns)
   # Index(['id', 'name', 'area_km2', 'mean_elev', 'geometry'], dtype='object')

Coordinate Reference Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure your catchment data has a proper CRS:

.. code-block:: python

   # Check CRS
   print(catchments.crs)

   # Set CRS if missing
   catchments = catchments.set_crs('EPSG:4326')

   # Reproject if needed
   catchments = catchments.to_crs('EPSG:3857')

Data Validation
~~~~~~~~~~~~~~~

Validate your catchment data before using it:

.. code-block:: python

   # Check for valid geometries
   invalid_geoms = catchments[~catchments.is_valid]
   if not invalid_geoms.empty:
       print(f"Warning: {len(invalid_geoms)} invalid geometries found")

   # Check for missing data
   missing_data = catchments.isnull().sum()
   print("Missing data per column:")
   print(missing_data[missing_data > 0])

   # Basic statistics
   print(f"Total catchments: {len(catchments)}")
   print(f"Total area: {catchments.area.sum():.2f} square units")

Forcing Data
------------

Supported Formats
~~~~~~~~~~~~~~~~~

MarrmotFlow supports various forcing data formats:

* NetCDF (.nc, .nc4)
* HDF5 (.h5, .hdf5)
* Zarr stores
* Any format supported by xarray

Data Structure
~~~~~~~~~~~~~~

Your forcing data should be structured as multidimensional arrays with:

* **Time dimension**: Temporal coordinate
* **Spatial dimensions**: Latitude/longitude or other spatial coordinates
* **Variables**: Precipitation, temperature, and other meteorological variables

.. code-block:: python

   import xarray as xr

   # Load forcing data
   forcing = xr.open_dataset("climate_data.nc")
   
   # Check structure
   print(forcing)
   print(forcing.coords)
   print(forcing.data_vars)

Required Variables
~~~~~~~~~~~~~~~~~~

At minimum, you need:

* **Precipitation**: Any units that can be converted to mm/day
* **Temperature**: Any units that can be converted to Celsius

.. code-block:: python

   # Example forcing data structure
   forcing_vars = {
       "precip": "precipitation",  # Variable name in your data
       "temp": "temperature"       # Variable name in your data
   }

Time Handling
~~~~~~~~~~~~~

Ensure proper time coordinates:

.. code-block:: python

   # Check time coordinate
   print(forcing.time)
   
   # Convert time if needed
   forcing['time'] = pd.to_datetime(forcing.time)
   
   # Set time zone if needed
   forcing = forcing.assign_coords(
       time=forcing.time.dt.tz_localize('UTC')
   )

Spatial Alignment
~~~~~~~~~~~~~~~~~

Your forcing data should cover your catchment areas:

.. code-block:: python

   # Check spatial bounds
   lon_min, lat_min, lon_max, lat_max = catchments.total_bounds
   
   forcing_lon_range = [forcing.lon.min().item(), forcing.lon.max().item()]
   forcing_lat_range = [forcing.lat.min().item(), forcing.lat.max().item()]
   
   print(f"Catchment bounds: {lon_min:.2f}, {lat_min:.2f}, {lon_max:.2f}, {lat_max:.2f}")
   print(f"Forcing lon range: {forcing_lon_range}")
   print(f"Forcing lat range: {forcing_lat_range}")

Unit Conversion
---------------

Precipitation Units
~~~~~~~~~~~~~~~~~~~

Common precipitation unit conversions:

.. code-block:: python

   forcing_units = {
       "precip": "mm/day",      # Direct use
       "precip": "mm/hour",     # Will be converted
       "precip": "m/day",       # Will be converted
       "precip": "kg m-2 s-1"   # CMIP6 standard, will be converted
   }

Temperature Units
~~~~~~~~~~~~~~~~~

Common temperature unit conversions:

.. code-block:: python

   forcing_units = {
       "temp": "celsius",       # Direct use
       "temp": "kelvin",        # Will be converted
       "temp": "fahrenheit"     # Will be converted
   }

Data Quality Checks
-------------------

Missing Data
~~~~~~~~~~~~

Check for and handle missing data:

.. code-block:: python

   # Check for NaN values
   precip_missing = forcing.precipitation.isnull().sum()
   temp_missing = forcing.temperature.isnull().sum()
   
   print(f"Missing precipitation values: {precip_missing.item()}")
   print(f"Missing temperature values: {temp_missing.item()}")

Outliers
~~~~~~~~

Identify potential outliers:

.. code-block:: python

   # Precipitation outliers (negative values or extremely high)
   negative_precip = (forcing.precipitation < 0).sum()
   extreme_precip = (forcing.precipitation > 1000).sum()  # > 1000 mm/day
   
   print(f"Negative precipitation values: {negative_precip.item()}")
   print(f"Extreme precipitation values: {extreme_precip.item()}")
   
   # Temperature outliers
   extreme_cold = (forcing.temperature < -50).sum()  # < -50°C
   extreme_hot = (forcing.temperature > 60).sum()    # > 60°C
   
   print(f"Extremely cold values: {extreme_cold.item()}")
   print(f"Extremely hot values: {extreme_hot.item()}")

Data Preprocessing Workflow
----------------------------

Complete preprocessing example:

.. code-block:: python

   import geopandas as gpd
   import xarray as xr
   import pandas as pd

   def prepare_data(catchment_file, forcing_file):
       """Complete data preparation workflow."""
       
       # Load catchment data
       catchments = gpd.read_file(catchment_file)
       
       # Validate catchments
       if catchments.crs is None:
           catchments = catchments.set_crs('EPSG:4326')
       
       # Load forcing data
       forcing = xr.open_dataset(forcing_file)
       
       # Standardize time
       forcing['time'] = pd.to_datetime(forcing.time)
       
       # Check spatial coverage
       lon_min, lat_min, lon_max, lat_max = catchments.total_bounds
       
       forcing_subset = forcing.sel(
           lon=slice(lon_min, lon_max),
           lat=slice(lat_min, lat_max)
       )
       
       # Quality checks
       print("Data quality summary:")
       print(f"Catchments: {len(catchments)} features")
       print(f"Forcing time range: {forcing.time.min().item()} to {forcing.time.max().item()}")
       print(f"Forcing spatial extent: {forcing_subset.dims}")
       
       return catchments, forcing_subset

   # Use the function
   catchments, forcing = prepare_data("catchments.shp", "climate_data.nc")

Best Practices
--------------

1. **Always validate your data** before creating workflows
2. **Use consistent coordinate systems** across all spatial data
3. **Document your data sources** and preprocessing steps
4. **Check temporal alignment** between different datasets
5. **Handle missing data appropriately** for your use case
6. **Use meaningful variable names** in your forcing data mapping
7. **Test with small datasets** before processing large volumes
