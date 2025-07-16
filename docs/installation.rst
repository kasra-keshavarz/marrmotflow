Installation
============

Requirements
------------

MarrmotFlow requires Python 3.8 or later and depends on several scientific Python packages:

* numpy >= 2.0.0
* pandas >= 2.0.0
* xarray >= 0.11
* geopandas >= 0.13.2
* scipy >= 1.15.0
* pint >= 0.20.0
* pint-pandas >= 0.7.1
* pint-xarray >= 0.5.0
* pyet >= 1.3.0
* netCDF4 >= 1.6.0
* timezonefinder >= 6.5.9
* click >= 8.2.1
* distributed >= 2023.1.0

Installation from PyPI
-----------------------

The easiest way to install MarrmotFlow is from PyPI using pip:

.. code-block:: bash

   pip install marrmotflow

Installation from Source
-------------------------

To install the latest development version from the GitHub repository:

.. code-block:: bash

   git clone https://github.com/kasra-keshavarz/marrmotflow.git
   cd marrmotflow
   pip install -e .

Development Installation
------------------------

If you plan to contribute to MarrmotFlow or want to use the latest features, install in development mode:

.. code-block:: bash

   git clone https://github.com/kasra-keshavarz/marrmotflow.git
   cd marrmotflow
   pip install -e ".[dev]"

This will install additional development dependencies including:

* pytest >= 6.0
* pytest-cov
* black
* flake8
* mypy
* pre-commit

Documentation Dependencies
---------------------------

To build the documentation locally, install the documentation dependencies:

.. code-block:: bash

   pip install -e ".[docs]"

This includes:

* sphinx
* sphinx-rtd-theme

Virtual Environment
-------------------

It's recommended to install MarrmotFlow in a virtual environment to avoid conflicts with other packages:

.. code-block:: bash

   python -m venv marrmotflow-env
   source marrmotflow-env/bin/activate  # On Windows: marrmotflow-env\Scripts\activate
   pip install marrmotflow

Verification
------------

To verify that MarrmotFlow is installed correctly, you can run:

.. code-block:: python

   import marrmotflow
   print(marrmotflow.__version__)

This should print the version number without any errors.
