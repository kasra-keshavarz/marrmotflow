Core Module
===========

The core module contains the main MarrmotFlow classes and functions.

.. automodule:: marrmotflow.core
   :members:
   :undoc-members:
   :show-inheritance:

MARRMOTWorkflow Class
--------------------

.. autoclass:: marrmotflow.core.MARRMOTWorkflow
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/

      ~MARRMOTWorkflow.__init__

Class Methods
~~~~~~~~~~~~~

The following methods are available on the MARRMOTWorkflow class:

.. automethod:: marrmotflow.core.MARRMOTWorkflow.__init__

Parameters
~~~~~~~~~~

The MARRMOTWorkflow class accepts the following parameters:

.. list-table:: MARRMOTWorkflow Parameters
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - name
     - str
     - Yes
     - Name identifier for the workflow
   * - cat
     - GeoDataFrame or PathLike
     - Yes
     - Catchment data as GeoDataFrame or path to spatial file
   * - forcing_vars
     - Dict[str, str]
     - Yes
     - Mapping of standard variable names to data variable names
   * - forcing_files
     - Sequence[PathLike] or PathLike
     - No
     - Path(s) to forcing data files
   * - forcing_units
     - Dict[str, str]
     - No
     - Units for forcing variables (using Pint notation)
   * - pet_method
     - str
     - No
     - PET calculation method ('penman_monteith' or 'hamon')
   * - model_number
     - Sequence[int] or int
     - No
     - MARRMOT model number(s) to use (default: [7, 37])
   * - forcing_time_zone
     - str
     - No
     - Time zone of forcing data
   * - model_time_zone
     - str
     - No
     - Time zone for model execution

Examples
~~~~~~~~

Basic workflow creation:

.. code-block:: python

   from marrmotflow import MARRMOTWorkflow
   import geopandas as gpd

   # Load catchment data
   catchments = gpd.read_file("catchments.shp")

   # Create workflow
   workflow = MARRMOTWorkflow(
       name="BasicWorkflow",
       cat=catchments,
       forcing_files=["climate_data.nc"],
       forcing_vars={"precip": "precipitation", "temp": "temperature"},
       forcing_units={"precip": "mm/day", "temp": "celsius"},
       model_number=[7, 37]
   )

Advanced workflow with multiple models:

.. code-block:: python

   workflow = MARRMOTWorkflow(
       name="MultiModelAnalysis",
       cat="large_watershed.shp",
       forcing_files=[
           "precip_2010_2020.nc",
           "temp_2010_2020.nc"
       ],
       forcing_vars={
           "precip": "total_precipitation",
           "temp": "2m_temperature"
       },
       forcing_units={
           "precip": "m/day",
           "temp": "kelvin"
       },
       pet_method="penman_monteith",
       model_number=[7, 37, 1, 2],
       forcing_time_zone="UTC",
       model_time_zone="America/Edmonton"
   )

Error Handling
~~~~~~~~~~~~~~

The MARRMOTWorkflow class raises the following exceptions:

.. py:exception:: ValueError

   Raised when required parameters are missing or invalid:
   
   - Missing catchment data
   - Invalid forcing variable mapping
   - Unsupported PET method
   - Invalid model numbers

.. py:exception:: TypeError

   Raised when parameter types are incorrect:
   
   - Catchment data is not a GeoDataFrame or PathLike
   - Forcing files are not in expected format

.. py:exception:: FileNotFoundError

   Raised when specified files cannot be found:
   
   - Catchment file does not exist
   - Forcing data files cannot be accessed

Type Definitions
~~~~~~~~~~~~~~~~

The core module defines the following types:

.. autodata:: marrmotflow.core.PathLike
   :annotation:

   Type alias for file paths, supporting both string and os.PathLike objects.
