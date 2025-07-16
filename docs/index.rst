MarrmotFlow Documentation
==========================

Welcome to MarrmotFlow, a Python package designed for creating and managing MARRMOT hydrological model workflows.

MarrmotFlow provides tools for configuring, running, and analyzing MARRMOT (Modular Assessment of Rainfall-Runoff Models Toolbox) models with Python, enabling seamless integration with modern data science workflows.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide/index
   api_reference/index
   examples/index
   contributing
   changelog

Features
--------

* **Easy Model Configuration**: Simple Python interface for configuring MARRMOT models
* **Multiple Models Support**: Support for various MARRMOT model structures
* **Data Integration**: Seamless integration with pandas, xarray, and geopandas
* **PET Calculation**: Built-in potential evapotranspiration calculation methods
* **Template System**: Flexible Jinja2-based templating for model generation
* **Forcing Data Management**: Comprehensive tools for handling meteorological forcing data

Quick Start
-----------

Install MarrmotFlow:

.. code-block:: bash

   pip install marrmotflow

Basic usage:

.. code-block:: python

   from marrmotflow import MARRMOTWorkflow
   
   # Example usage
   config = {
       'cat': 'path/to/catchment.shp', 
       'forcing_files': 'path/to/forcing/files',
       'forcing_vars': {
           "temperature": "temperature_variable_name",
           "precipitation": "precipitation_variable_name",
       },
       'forcing_units': {
           'temperature': 'celsius',
           'precipitation': 'meter / hour',
       },
       'forcing_time_zone': 'UTC',
   }
   
   # Build the MARRMOT workflow
   marrmot_experiment = MARRMOTWorkflow(**config)
   
   # Run the workflow
   marrmot_experiment.run()
   
   # Save the results
   marrmot_experiment.save_results('path/to/save/results/directory')


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
