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
   import geopandas as gpd

   # Load your catchment data
   catchments = gpd.read_file("catchments.shp")
   
   # Create a workflow
   workflow = MARRMOTWorkflow(
       name="MyWorkflow",
       cat=catchments,
       forcing_files=["forcing_data.nc"],
       forcing_vars={"precip": "precipitation", "temp": "temperature"},
       model_number=[7, 37]  # HBV-96 and GR4J models
   )

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
