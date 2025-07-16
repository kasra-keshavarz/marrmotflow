PET Calculation
===============

Potential Evapotranspiration (PET) is a crucial component of hydrological modeling. MarrmotFlow provides multiple methods for calculating PET from meteorological data.

Overview
--------

PET represents the amount of water that would be evaporated and transpired if there were sufficient water available. MarrmotFlow calculates PET automatically based on your forcing data and the specified method.

Available Methods
-----------------

Penman-Monteith (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~

The Penman-Monteith method is the recommended approach and is used by default:

.. code-block:: python

   workflow = MARRMOTWorkflow(
       pet_method="penman_monteith",  # Default method
       # ... other parameters
   )

**Advantages:**
- Physically-based approach
- Accounts for energy balance and aerodynamic components
- Recommended by FAO for reference evapotranspiration

**Data requirements:**
- Temperature (required)
- Additional meteorological variables (when available):
  - Solar radiation
  - Wind speed
  - Relative humidity

Hamon Method
~~~~~~~~~~~~

A temperature-based empirical method:

.. code-block:: python

   workflow = MARRMOTWorkflow(
       pet_method="hamon",
       # ... other parameters
   )

**Advantages:**
- Simple implementation
- Requires only temperature data
- Suitable when limited meteorological data is available

**Limitations:**
- Less physically-based
- May be less accurate in certain climates

Method Configuration
--------------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

Specify the PET method when creating a workflow:

.. code-block:: python

   from marrmotflow import MARRMOTWorkflow

   # Penman-Monteith (recommended)
   workflow_pm = MARRMOTWorkflow(
       name="PenmanMonteith_Analysis",
       pet_method="penman_monteith",
       # ... other parameters
   )

   # Hamon method
   workflow_hamon = MARRMOTWorkflow(
       name="Hamon_Analysis", 
       pet_method="hamon",
       # ... other parameters
   )

Data Requirements by Method
---------------------------

Penman-Monteith Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Minimum requirements:**
- Temperature (mandatory)

**Additional data (when available):**

.. code-block:: python

   # Extended forcing variables for Penman-Monteith
   forcing_vars = {
       "precip": "precipitation",
       "temp": "temperature",        # Required
       "rad": "solar_radiation",     # Optional but recommended
       "wind": "wind_speed",         # Optional
       "rh": "relative_humidity"     # Optional
   }

   forcing_units = {
       "precip": "mm/day",
       "temp": "celsius",
       "rad": "W/m2",               # or "MJ/m2/day"
       "wind": "m/s",
       "rh": "percent"              # or fraction (0-1)
   }

Hamon Requirements
~~~~~~~~~~~~~~~~~~

**Minimum requirements:**
- Temperature (mandatory)

.. code-block:: python

   # Minimal configuration for Hamon
   forcing_vars = {
       "precip": "precipitation",
       "temp": "temperature"        # Only temperature required
   }

Climate Considerations
----------------------

Method Selection by Climate
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose PET methods based on climate characteristics:

.. code-block:: python

   def select_pet_method(climate_type, data_availability):
       """Select appropriate PET method based on climate and data."""
       
       if data_availability == "full":
           return "penman_monteith"
       elif data_availability == "temperature_only":
           if climate_type in ["arid", "semi_arid"]:
               # Hamon may overestimate in arid regions
               return "penman_monteith"  # Use with temperature only
           else:
               return "hamon"
       else:
           return "penman_monteith"  # Default

   # Usage
   pet_method = select_pet_method("temperate", "full")
   workflow = MARRMOTWorkflow(
       pet_method=pet_method,
       # ... other parameters
   )

Regional Calibration
~~~~~~~~~~~~~~~~~~~~

PET methods may need regional calibration:

.. code-block:: python

   # Regional considerations for PET calculation
   regional_configs = {
       "mediterranean": {
           "method": "penman_monteith",
           "note": "High radiation, consider crop coefficients"
       },
       "tropical": {
           "method": "penman_monteith", 
           "note": "High humidity affects calculations"
       },
       "arctic": {
           "method": "hamon",
           "note": "Limited radiation, temperature-based methods suitable"
       },
       "continental": {
           "method": "penman_monteith",
           "note": "Large temperature variations, full method recommended"
       }
   }

PET Calculation Workflow
------------------------

Internal Processing
~~~~~~~~~~~~~~~~~~~

MarrmotFlow handles PET calculation internally:

.. code-block:: python

   # PET is calculated automatically during workflow execution
   workflow = MARRMOTWorkflow(
       name="AutoPET",
       cat="catchments.shp",
       forcing_files="climate_data.nc",
       forcing_vars={"precip": "precipitation", "temp": "temperature"},
       pet_method="penman_monteith"
   )

   # PET values will be computed and used in model runs
   # No explicit PET calculation call needed

Location-Specific Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PET calculations account for geographic location:

.. code-block:: python

   # Geographic factors automatically considered:
   # - Latitude (for solar radiation estimation)
   # - Elevation (for atmospheric pressure)
   # - Time zone (for solar time calculations)
   
   workflow = MARRMOTWorkflow(
       # Geographic information from catchment data
       cat="catchments_with_elevation.shp",  # Should include elevation
       forcing_time_zone="UTC",
       model_time_zone="America/Vancouver",  # Local time zone
       pet_method="penman_monteith",
       # ... other parameters
   )

Advanced PET Configuration
--------------------------

Custom PET Data
~~~~~~~~~~~~~~~

If you have pre-calculated PET data:

.. code-block:: python

   # Include PET in forcing variables (future feature)
   forcing_vars = {
       "precip": "precipitation",
       "temp": "temperature",
       "pet": "potential_evapotranspiration"  # Pre-calculated PET
   }

   # When PET is provided, it may override method calculation
   # (Check current implementation for this feature)

Quality Assessment
------------------

PET Validation
~~~~~~~~~~~~~~

Validate calculated PET values:

.. code-block:: python

   import numpy as np
   
   def validate_pet_estimates(temperature, pet_values):
       """Basic validation of PET estimates."""
       
       # PET should be positive
       negative_pet = np.sum(pet_values < 0)
       if negative_pet > 0:
           print(f"Warning: {negative_pet} negative PET values")
       
       # PET should generally correlate with temperature
       correlation = np.corrcoef(temperature, pet_values)[0, 1]
       print(f"Temperature-PET correlation: {correlation:.3f}")
       
       # Reasonable annual totals (varies by climate)
       annual_pet = np.sum(pet_values)  # Assuming daily values
       print(f"Annual PET: {annual_pet:.1f} mm")
       
       return {
           "negative_values": negative_pet,
           "temp_correlation": correlation,
           "annual_total": annual_pet
       }

Comparison Between Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare different PET methods:

.. code-block:: python

   # Create workflows with different methods
   methods = ["penman_monteith", "hamon"]
   workflows = {}

   for method in methods:
       workflows[method] = MARRMOTWorkflow(
           name=f"PET_Comparison_{method}",
           cat="catchments.shp",
           forcing_files="climate_data.nc", 
           forcing_vars={"precip": "precipitation", "temp": "temperature"},
           pet_method=method
       )

   # Compare results after running workflows
   # (Implementation depends on output handling)

Best Practices
--------------

1. **Use Penman-Monteith when possible** - more physically-based and accurate
2. **Consider data availability** when selecting methods
3. **Account for climate characteristics** in method selection
4. **Validate PET estimates** against known values or other methods
5. **Document method choice** and rationale
6. **Consider regional calibration** for improved accuracy
7. **Test sensitivity** to different PET methods in your analysis

Common Issues and Solutions
---------------------------

High PET Values
~~~~~~~~~~~~~~~

If PET values seem unreasonably high:

.. code-block:: python

   # Check input data quality
   # - Temperature units (should be Celsius)
   # - Solar radiation units and values
   # - Geographic coordinates accuracy

Low PET Values
~~~~~~~~~~~~~~

If PET values seem unreasonably low:

.. code-block:: python

   # Check for:
   # - Missing or incorrect meteorological data
   # - Wrong latitude/longitude in catchment data
   # - Incorrect time zone specifications

Method Comparison
~~~~~~~~~~~~~~~~~

When methods give very different results:

.. code-block:: python

   # Expected differences:
   # - Penman-Monteith typically higher in arid regions
   # - Hamon may be lower in high-radiation environments
   # - Both should show similar seasonal patterns

Troubleshooting
---------------

1. **Check input data units** - especially temperature (must be Celsius for internal calculations)
2. **Verify geographic coordinates** - latitude affects solar radiation calculations
3. **Confirm time zones** - important for solar time calculations
4. **Validate forcing data quality** - missing or extreme values affect PET
5. **Compare with literature values** - check if results are reasonable for your region
