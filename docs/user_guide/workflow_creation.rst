Workflow Creation
=================

The ``MARRMOTWorkflow`` class is the central component of MarrmotFlow. This guide explains how to create and configure workflows for different use cases.

Basic Workflow Setup
---------------------

Creating a basic workflow requires minimal configuration:

.. code-block:: python

   from marrmotflow import MARRMOTWorkflow
   import geopandas as gpd

   # Load catchment data
   catchments = gpd.read_file("catchments.shp")

   # Create workflow
   workflow = MARRMOTWorkflow(
       name="MyWorkflow",
       cat=catchments,
       forcing_files=["climate_data.nc"],
       forcing_vars={"precip": "precipitation", "temp": "temperature"}
   )

Required Parameters
-------------------

name
~~~~

A string identifier for your workflow:

.. code-block:: python

   workflow = MARRMOTWorkflow(name="WatershedAnalysis2024")

cat
~~~

Catchment data as a GeoDataFrame or file path:

.. code-block:: python

   # From GeoDataFrame
   catchments = gpd.read_file("catchments.shp")
   workflow = MARRMOTWorkflow(cat=catchments)

   # From file path
   workflow = MARRMOTWorkflow(cat="catchments.shp")

forcing_vars
~~~~~~~~~~~~

Dictionary mapping standard variable names to your data variable names:

.. code-block:: python

   forcing_vars = {
       "precip": "precipitation",  # Required
       "temp": "temperature"       # Required
   }

Optional Parameters
-------------------

forcing_files
~~~~~~~~~~~~~

Paths to your forcing data files:

.. code-block:: python

   # Single file
   forcing_files = "climate_data.nc"

   # Multiple files
   forcing_files = [
       "precip_2020.nc",
       "temp_2020.nc",
       "climate_2021.nc"
   ]

forcing_units
~~~~~~~~~~~~~

Units for your forcing variables:

.. code-block:: python

   forcing_units = {
       "precip": "mm/day",
       "temp": "celsius"
   }

pet_method
~~~~~~~~~~

Method for calculating potential evapotranspiration:

.. code-block:: python

   # Available methods
   pet_method = "penman_monteith"  # Default
   pet_method = "hamon"

model_number
~~~~~~~~~~~~

MARRMOT model(s) to use:

.. code-block:: python

   # Single model
   model_number = 7  # HBV-96

   # Multiple models
   model_number = [7, 37, 1]  # HBV-96, GR4J, and Collie River Basin 1

Time Zone Configuration
-----------------------

forcing_time_zone
~~~~~~~~~~~~~~~~~

Time zone of your forcing data:

.. code-block:: python

   forcing_time_zone = "UTC"
   forcing_time_zone = "America/Edmonton"
   forcing_time_zone = "Europe/London"

model_time_zone
~~~~~~~~~~~~~~~

Time zone for model execution:

.. code-block:: python

   model_time_zone = "America/Vancouver"

Advanced Configuration Examples
-------------------------------

Multi-Model Watershed Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   workflow = MARRMOTWorkflow(
       name="MultiModelComparison",
       cat="large_watershed.shp",
       forcing_files=[
           "era5_precip_2010_2020.nc",
           "era5_temp_2010_2020.nc"
       ],
       forcing_vars={
           "precip": "total_precipitation",
           "temp": "2m_temperature"
       },
       forcing_units={
           "precip": "m/day",  # ERA5 uses meters
           "temp": "kelvin"    # ERA5 uses Kelvin
       },
       pet_method="penman_monteith",
       model_number=[7, 37, 1, 2],  # Multiple models for comparison
       forcing_time_zone="UTC",
       model_time_zone="America/Edmonton"
   )

Regional Climate Study
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   workflow = MARRMOTWorkflow(
       name="ClimateChangeImpact",
       cat=gpd.read_file("regional_catchments.geojson"),
       forcing_files="gcm_downscaled_data.nc",
       forcing_vars={
           "precip": "pr",      # CMIP6 standard names
           "temp": "tas"
       },
       forcing_units={
           "precip": "kg m-2 s-1",  # CMIP6 standard units
           "temp": "K"
       },
       pet_method="penman_monteith",
       model_number=37,  # GR4J for this study
       forcing_time_zone="UTC",
       model_time_zone="local"
   )

Error Handling
--------------

Common errors and solutions:

**Missing Required Parameters**:

.. code-block:: python

   try:
       workflow = MARRMOTWorkflow(name="Test")
   except ValueError as e:
       print(f"Error: {e}")
       # Catchment (cat) must be provided

**Invalid File Paths**:

.. code-block:: python

   import os
   
   forcing_file = "nonexistent_file.nc"
   if not os.path.exists(forcing_file):
       print(f"Warning: {forcing_file} does not exist")

**Incompatible Units**:

.. code-block:: python

   import pint
   
   ureg = pint.UnitRegistry()
   try:
       ureg.parse_expression("invalid_unit")
   except pint.UndefinedUnitError:
       print("Invalid unit specified")

Best Practices
--------------

1. **Use descriptive names**: Choose meaningful workflow names for easier identification
2. **Validate inputs**: Check that files exist and units are valid before creating workflows
3. **Document your choices**: Keep track of why you chose specific models and methods
4. **Start simple**: Begin with single models and basic configurations before advancing
5. **Test with subsets**: Use small catchments or short time periods for initial testing
