Default Dictionaries Module
============================

The _default_dicts module contains default configurations and mappings used throughout MarrmotFlow.

.. automodule:: marrmotflow._default_dicts
   :members:
   :undoc-members:
   :show-inheritance:

Default Forcing Units
---------------------

.. autodata:: marrmotflow._default_dicts.default_forcing_units

   Default units for forcing variables used when units are not explicitly specified.

   :type: Dict[str, str]

   **Example values:**

   .. code-block:: python

      default_forcing_units = {
          "precip": "mm/day",
          "temp": "celsius",
          "pet": "mm/day",
          "rad": "W/m2",
          "wind": "m/s",
          "rh": "percent"
      }

   These units follow Pint conventions and are used for automatic unit conversion
   when processing forcing data.

Default Forcing Variables
-------------------------

.. autodata:: marrmotflow._default_dicts.default_forcing_vars

   Default variable name mappings for common forcing data formats.

   :type: Dict[str, str]

   **Example values:**

   .. code-block:: python

      default_forcing_vars = {
          "precip": "precipitation",
          "temp": "temperature",
          "pet": "potential_evapotranspiration"
      }

   These mappings are used when ``forcing_vars`` is not explicitly provided
   to the MARRMOTWorkflow constructor.

Default Model Dictionary
------------------------

.. autodata:: marrmotflow._default_dicts.default_model_dict

   Default configuration dictionary for MARRMOT models.

   :type: Dict[int, Dict[str, Any]]

   **Structure:**

   .. code-block:: python

      default_model_dict = {
          7: {  # HBV-96
              "name": "HBV-96",
              "description": "HBV-96 conceptual model",
              "parameters": {
                  "TT": {"default": 0.0, "range": [-3, 3], "description": "Temperature threshold"},
                  "C0": {"default": 3.0, "range": [1, 6], "description": "Degree-day factor"},
                  # ... more parameters
              },
              "states": ["snow_storage", "soil_moisture", "groundwater"],
              "fluxes": ["snowmelt", "evapotranspiration", "recharge", "discharge"]
          },
          37: {  # GR4J
              "name": "GR4J",
              "description": "GR4J daily model",
              "parameters": {
                  "X1": {"default": 100, "range": [1, 3000], "description": "Production store capacity"},
                  "X2": {"default": 0, "range": [-10, 10], "description": "Intercatchment exchange"},
                  # ... more parameters
              },
              "states": ["production_store", "routing_store"],
              "fluxes": ["evapotranspiration", "percolation", "exchange", "discharge"]
          }
      }

Model Configuration Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each model entry in the default_model_dict contains:

.. list-table:: Model Dictionary Structure
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - name
     - str
     - Human-readable model name
   * - description
     - str
     - Brief description of the model
   * - parameters
     - Dict[str, Dict]
     - Model parameters with defaults and ranges
   * - states
     - List[str]
     - List of model state variable names
   * - fluxes
     - List[str]
     - List of model flux variable names

Parameter Configuration
~~~~~~~~~~~~~~~~~~~~~~

Each parameter in the model configuration has the following structure:

.. list-table:: Parameter Structure
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - default
     - float
     - Default parameter value
   * - range
     - List[float, float]
     - Valid parameter range [min, max]
   * - description
     - str
     - Physical meaning of the parameter
   * - units
     - str (optional)
     - Parameter units

Examples
--------

Accessing Default Values
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from marrmotflow._default_dicts import (
       default_forcing_units,
       default_forcing_vars,
       default_model_dict
   )

   # Get default units for precipitation
   precip_unit = default_forcing_units["precip"]
   print(f"Default precipitation unit: {precip_unit}")

   # Get default variable mapping
   precip_var = default_forcing_vars["precip"]
   print(f"Default precipitation variable name: {precip_var}")

   # Get HBV-96 model information
   hbv_config = default_model_dict[7]
   print(f"Model name: {hbv_config['name']}")
   print(f"Parameters: {list(hbv_config['parameters'].keys())}")

Working with Model Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Access parameter information for HBV-96
   hbv_params = default_model_dict[7]["parameters"]

   # Get temperature threshold parameter
   tt_param = hbv_params["TT"]
   print(f"TT default: {tt_param['default']}")
   print(f"TT range: {tt_param['range']}")
   print(f"TT description: {tt_param['description']}")

   # Get all parameter defaults for a model
   param_defaults = {
       name: config["default"]
       for name, config in hbv_params.items()
   }
   print(f"HBV-96 defaults: {param_defaults}")

Custom Configurations
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create custom configuration based on defaults
   custom_forcing_units = default_forcing_units.copy()
   custom_forcing_units["precip"] = "mm/hour"  # Override default

   # Extend default model configuration
   custom_model_dict = default_model_dict.copy()
   custom_model_dict[999] = {  # Custom model
       "name": "CustomModel",
       "description": "Custom hydrological model",
       "parameters": {
           "K": {"default": 1.0, "range": [0.1, 10], "description": "Flow coefficient"}
       },
       "states": ["storage"],
       "fluxes": ["outflow"]
   }

Usage in Workflows
------------------

Default Configuration
~~~~~~~~~~~~~~~~~~~~

These defaults are used automatically when not specified:

.. code-block:: python

   from marrmotflow import MARRMOTWorkflow

   # This workflow will use default units and variable mappings
   workflow = MARRMOTWorkflow(
       name="DefaultConfig",
       cat="catchments.shp",
       forcing_files="climate_data.nc"
       # forcing_vars and forcing_units will use defaults
   )

Override Defaults
~~~~~~~~~~~~~~~~

You can override defaults by providing explicit values:

.. code-block:: python

   # Override default units
   custom_units = {
       "precip": "inches/day",  # Override default mm/day
       "temp": "fahrenheit"     # Override default celsius
   }

   workflow = MARRMOTWorkflow(
       name="CustomUnits",
       cat="catchments.shp",
       forcing_files="climate_data.nc",
       forcing_vars={"precip": "rainfall", "temp": "air_temp"},
       forcing_units=custom_units
   )

Model Selection
~~~~~~~~~~~~~~

.. code-block:: python

   # Check available models
   available_models = list(default_model_dict.keys())
   print(f"Available models: {available_models}")

   # Get model names
   model_names = {
       num: config["name"] 
       for num, config in default_model_dict.items()
   }
   print(f"Model mapping: {model_names}")

   # Select models based on characteristics
   conceptual_models = [
       num for num, config in default_model_dict.items()
       if "conceptual" in config["description"].lower()
   ]

Validation Functions
--------------------

Unit Validation
~~~~~~~~~~~~~~~

.. code-block:: python

   def validate_forcing_units(units_dict):
       """Validate that forcing units are recognized."""
       import pint
       
       ureg = pint.UnitRegistry()
       valid_units = {}
       
       for var, unit in units_dict.items():
           try:
               ureg.parse_expression(unit)
               valid_units[var] = unit
           except pint.UndefinedUnitError:
               print(f"Warning: Unrecognized unit '{unit}' for variable '{var}'")
               # Use default
               valid_units[var] = default_forcing_units.get(var, "dimensionless")
       
       return valid_units

Model Validation
~~~~~~~~~~~~~~~

.. code-block:: python

   def validate_model_number(model_number):
       """Validate that model number is supported."""
       if isinstance(model_number, int):
           model_numbers = [model_number]
       else:
           model_numbers = list(model_number)
       
       unsupported = [
           num for num in model_numbers
           if num not in default_model_dict
       ]
       
       if unsupported:
           available = list(default_model_dict.keys())
           raise ValueError(
               f"Unsupported model numbers: {unsupported}. "
               f"Available models: {available}"
           )
       
       return model_numbers

Best Practices
--------------

1. **Use defaults when possible** - they are tested and validated
2. **Override only when necessary** - maintain consistency
3. **Validate custom configurations** - ensure units and ranges are appropriate
4. **Document changes** - explain why defaults were overridden
5. **Test custom configurations** - verify they work with your data
6. **Consider backward compatibility** - when modifying defaults
7. **Use standard units** - prefer SI or commonly accepted units

Constants
---------

The module may also define various constants used throughout the package:

.. code-block:: python

   # Example constants (check actual implementation)
   DEFAULT_PET_METHOD = "penman_monteith"
   DEFAULT_MODEL_NUMBERS = [7, 37]
   SUPPORTED_FILE_FORMATS = [".nc", ".nc4", ".h5", ".hdf5"]
