Model Configuration
===================

MarrmotFlow supports various MARRMOT model structures. This guide explains how to configure and use different models effectively.

Available Models
----------------

MarrmotFlow provides access to numerous MARRMOT model structures. The default models are:

* **Model 7**: HBV-96 (Hydrologiska Byråns Vattenbalansavdelning)
* **Model 37**: GR4J (Génie Rural à 4 paramètres Journalier)

Model Selection
---------------

Single Model
~~~~~~~~~~~~

Configure a workflow with a single model:

.. code-block:: python

   from marrmotflow import MARRMOTWorkflow

   # HBV-96 model only
   workflow = MARRMOTWorkflow(
       name="HBV_Analysis",
       cat="catchments.shp",
       forcing_files="climate_data.nc",
       forcing_vars={"precip": "precipitation", "temp": "temperature"},
       model_number=7  # HBV-96
   )

Multiple Models
~~~~~~~~~~~~~~~

Compare multiple models in the same workflow:

.. code-block:: python

   # Multiple models for comparison
   workflow = MARRMOTWorkflow(
       name="ModelComparison",
       cat="catchments.shp",
       forcing_files="climate_data.nc",
       forcing_vars={"precip": "precipitation", "temp": "temperature"},
       model_number=[7, 37]  # HBV-96 and GR4J
   )

Model Descriptions
------------------

HBV-96 (Model 7)
~~~~~~~~~~~~~~~~

The HBV-96 model is a conceptual hydrological model developed by the Swedish Meteorological and Hydrological Institute (SMHI).

**Characteristics:**
- Snow accumulation and melt routines
- Soil moisture accounting
- Simple groundwater representation
- Suitable for Nordic and mountainous catchments

**Parameters:**
- TT: Temperature threshold for snowfall/snowmelt
- C0: Degree-day factor
- ETF: Evapotranspiration correction factor
- LP: Limit for potential evapotranspiration
- FC: Maximum soil moisture storage
- BETA: Shape parameter for recharge function
- CFLUX: Maximum capillary flow from upper to lower zone
- K0, K1, K2: Recession coefficients
- ALPHA: Measure of non-linearity of quick flow

.. code-block:: python

   # HBV-96 specific configuration
   workflow = MARRMOTWorkflow(
       model_number=7,
       # ... other parameters
   )

GR4J (Model 37)
~~~~~~~~~~~~~~~

GR4J is a daily lumped rainfall-runoff model developed by Cemagref (now INRAE).

**Characteristics:**
- Simple 4-parameter structure
- Production and routing functions
- Intercatchment groundwater exchange
- Robust performance across various climates

**Parameters:**
- X1: Maximum capacity of production store (mm)
- X2: Intercatchment exchange coefficient (mm/day)
- X3: Maximum capacity of routing store (mm)
- X4: Time base of unit hydrograph (days)

.. code-block:: python

   # GR4J specific configuration
   workflow = MARRMOTWorkflow(
       model_number=37,
       # ... other parameters
   )

Model Configuration Options
---------------------------

Time Step Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Models operate at daily time steps by default:

.. code-block:: python

   # Daily time step (default)
   workflow = MARRMOTWorkflow(
       # ... parameters
       # Daily time step is implicit
   )

Initial Conditions
~~~~~~~~~~~~~~~~~~

Models use default initial conditions, but these can be customized through the workflow:

.. code-block:: python

   # Default initial conditions are used automatically
   # Custom initial conditions would be set during model execution
   pass

Model Comparison Workflows
--------------------------

Ensemble Modeling
~~~~~~~~~~~~~~~~~

Run multiple models to create ensemble predictions:

.. code-block:: python

   # Ensemble of multiple models
   ensemble_models = [7, 37, 1, 2]  # Multiple model structures
   
   workflow = MARRMOTWorkflow(
       name="EnsembleModeling",
       cat="catchments.shp",
       forcing_files="climate_data.nc",
       forcing_vars={"precip": "precipitation", "temp": "temperature"},
       model_number=ensemble_models
   )

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

Set up workflows for systematic model comparison:

.. code-block:: python

   # Define models to compare
   models_to_compare = {
       "HBV": 7,
       "GR4J": 37,
       "Collie": 1
   }
   
   workflows = {}
   for name, model_num in models_to_compare.items():
       workflows[name] = MARRMOTWorkflow(
           name=f"Comparison_{name}",
           cat="catchments.shp",
           forcing_files="climate_data.nc",
           forcing_vars={"precip": "precipitation", "temp": "temperature"},
           model_number=model_num
       )

Advanced Model Configuration
----------------------------

Climate-Specific Model Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose models based on climate characteristics:

.. code-block:: python

   def select_models_by_climate(climate_type):
       """Select appropriate models based on climate."""
       if climate_type == "snow_dominated":
           return [7]  # HBV-96 with snow routines
       elif climate_type == "arid":
           return [37]  # GR4J for water-limited conditions
       elif climate_type == "temperate":
           return [7, 37]  # Both models
       else:
           return [7, 37]  # Default to both

   # Usage
   models = select_models_by_climate("snow_dominated")
   workflow = MARRMOTWorkflow(
       model_number=models,
       # ... other parameters
   )

Catchment-Specific Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adapt model selection to catchment characteristics:

.. code-block:: python

   import geopandas as gpd

   def configure_by_catchment(catchment_file):
       """Configure models based on catchment properties."""
       catchments = gpd.read_file(catchment_file)
       
       # Example: Use HBV for high-elevation catchments
       if 'mean_elev' in catchments.columns:
           high_elev = catchments['mean_elev'].mean() > 1000  # meters
           if high_elev:
               return [7]  # HBV-96
           else:
               return [37]  # GR4J
       
       return [7, 37]  # Default

   # Usage
   models = configure_by_catchment("catchments.shp")
   workflow = MARRMOTWorkflow(
       model_number=models,
       # ... other parameters
   )

Model Validation Considerations
-------------------------------

Data Requirements
~~~~~~~~~~~~~~~~~

Different models may have different data requirements:

.. code-block:: python

   # Check if snow models need additional data
   def check_model_requirements(model_number, forcing_vars):
       """Check if model requirements are met."""
       if model_number == 7:  # HBV-96
           # Could benefit from snow data if available
           if 'snow' not in forcing_vars:
               print("Note: HBV-96 can use snow data if available")
       
       # All models require precip and temp
       required = ['precip', 'temp']
       missing = [var for var in required if var not in forcing_vars]
       if missing:
           raise ValueError(f"Missing required variables: {missing}")

   # Usage
   check_model_requirements(7, {"precip": "precipitation", "temp": "temperature"})

Performance Metrics
~~~~~~~~~~~~~~~~~~~

Consider model-specific performance characteristics:

.. code-block:: python

   # Model performance characteristics
   model_characteristics = {
       7: {  # HBV-96
           "strengths": ["snow processes", "Nordic climates", "conceptual clarity"],
           "limitations": ["parameter equifinality", "snow-free regions"]
       },
       37: {  # GR4J
           "strengths": ["robustness", "few parameters", "various climates"],
           "limitations": ["no explicit snow", "lumped approach"]
       }
   }

Best Practices
--------------

1. **Start with default models** (7 and 37) for initial analysis
2. **Consider climate characteristics** when selecting models
3. **Use multiple models** for uncertainty assessment
4. **Match model complexity** to data availability and study objectives
5. **Document model selection rationale** for reproducibility
6. **Validate model assumptions** against catchment characteristics
7. **Consider computational resources** when using multiple models
