Quick Start Guide
=================

This guide will help you get started with MarrmotFlow quickly. We'll walk through a basic example of setting up and running a MARRMOT workflow.

Basic Workflow Example
-----------------------

Here's a minimal example to get you started:

.. code-block:: python

   import geopandas as gpd
   from marrmotflow import MARRMOTWorkflow

   # Load catchment data
   catchments = gpd.read_file("path/to/your/catchments.shp")

   # Define forcing variables mapping
   forcing_vars = {
       "precip": "precipitation",  # Variable name in your forcing files
       "temp": "temperature"       # Variable name in your forcing files
   }

   # Define units for forcing variables
   forcing_units = {
       "precip": "mm/day",
       "temp": "celsius"
   }

   # Create a workflow
   workflow = MARRMOTWorkflow(
       name="BasicWorkflow",
       cat=catchments,
       forcing_files=["forcing_data.nc"],
       forcing_vars=forcing_vars,
       forcing_units=forcing_units,
       pet_method="penman_monteith",
       model_number=[7, 37]  # HBV-96 and GR4J models
   )

Understanding the Components
----------------------------

Catchment Data
~~~~~~~~~~~~~~

MarrmotFlow expects catchment data as a GeoPandas GeoDataFrame or a path to a spatial file (shapefile, GeoJSON, etc.):

.. code-block:: python

   # From file
   catchments = gpd.read_file("catchments.shp")
   
   # Or pass the file path directly
   workflow = MARRMOTWorkflow(
       cat="catchments.shp",
       # ... other parameters
   )

Forcing Variables
~~~~~~~~~~~~~~~~~

The ``forcing_vars`` dictionary maps the standardized variable names used by MarrmotFlow to the actual variable names in your forcing data files:

.. code-block:: python

   forcing_vars = {
       "precip": "your_precipitation_variable_name",
       "temp": "your_temperature_variable_name"
   }

Forcing Units
~~~~~~~~~~~~~

Specify the units of your forcing variables using Pint-compatible unit strings:

.. code-block:: python

   forcing_units = {
       "precip": "mm/day",      # or "mm/d", "millimeter/day"
       "temp": "celsius"        # or "degC", "degree_Celsius"
   }

Model Selection
~~~~~~~~~~~~~~~

MarrmotFlow supports various MARRMOT model structures. You can specify one or multiple models:

.. code-block:: python

   # Single model
   model_number = 7  # HBV-96
   
   # Multiple models
   model_number = [7, 37]  # HBV-96 and GR4J

Common model numbers include:

* 7: HBV-96
* 37: GR4J
* 1: Collie River Basin 1
* 2: Wetland model

PET Methods
~~~~~~~~~~~

MarrmotFlow supports different methods for calculating potential evapotranspiration:

.. code-block:: python

   # Available methods
   pet_method = "penman_monteith"  # Default and recommended
   pet_method = "hamon"           # Alternative method

Time Zones
~~~~~~~~~~

You can specify time zones for forcing data and model execution:

.. code-block:: python

   workflow = MARRMOTWorkflow(
       # ... other parameters
       forcing_time_zone="UTC",
       model_time_zone="America/Edmonton"
   )

Next Steps
----------

Now that you have a basic understanding of MarrmotFlow, you can:

1. Explore the :doc:`user_guide/index` for more detailed information
2. Check out the :doc:`examples/index` for more complex scenarios
3. Review the :doc:`api_reference/index` for complete API documentation

Common Issues
-------------

**Import Error**: If you encounter import errors, make sure all dependencies are installed:

.. code-block:: bash

   pip install -e ".[dev]"

**File Not Found**: Ensure your file paths are correct and files exist:

.. code-block:: python

   import os
   print(os.path.exists("your_file_path"))

**Unit Errors**: Make sure your unit strings are valid Pint units:

.. code-block:: python

   import pint
   ureg = pint.UnitRegistry()
   print(ureg.parse_expression("mm/day"))  # Should not raise an error
