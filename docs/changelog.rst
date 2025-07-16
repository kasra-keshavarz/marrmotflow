Changelog
=========

All notable changes to MarrmotFlow will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~
* Comprehensive documentation with user guides and API reference
* Examples for basic workflows and multi-model comparisons
* Type hints throughout the codebase
* Enhanced error handling and validation

Changed
~~~~~~~
* Improved docstring format to NumPy style
* Updated configuration for better development workflow

[0.1.0] - 2024-07-16
--------------------

Added
~~~~~
* Initial release of MarrmotFlow
* Core ``MARRMOTWorkflow`` class for workflow management
* Support for multiple MARRMOT model structures (HBV-96, GR4J, etc.)
* Jinja2-based templating system for model generation
* PET calculation methods (Penman-Monteith, Hamon)
* Integration with pandas, xarray, and geopandas
* Default configurations for common use cases
* Basic test suite
* Project structure and packaging

Features
~~~~~~~~
* **Workflow Management**: Create and configure MARRMOT modeling workflows
* **Multi-Model Support**: Run multiple model structures in parallel
* **Flexible Data Input**: Support for various forcing data formats (NetCDF, HDF5)
* **Automatic Unit Conversion**: Handle different unit systems automatically
* **Template System**: Generate MATLAB code for MARRMOT models
* **Geospatial Integration**: Work with catchment boundaries and spatial data
* **Time Zone Support**: Handle data from different time zones
* **Quality Assurance**: Built-in data validation and error checking

Dependencies
~~~~~~~~~~~~
* Python >= 3.8
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

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~
* pytest >= 6.0
* pytest-cov
* black
* flake8
* mypy
* pre-commit
* sphinx
* sphinx-rtd-theme

Known Issues
~~~~~~~~~~~~
* Template generation is currently limited to default MARRMOT models
* PET calculation methods are limited to Penman-Monteith and Hamon
* Model execution interface is still under development

Future Releases
---------------

[0.2.0] - Planned
~~~~~~~~~~~~~~~~~

Planned Additions
^^^^^^^^^^^^^^^^^
* Model execution and result handling
* Additional PET calculation methods
* Enhanced error handling and logging
* Performance optimization for large datasets
* Additional MARRMOT model structures
* Calibration utilities
* Visualization tools

[0.3.0] - Planned  
~~~~~~~~~~~~~~~~~

Planned Additions
^^^^^^^^^^^^^^^^^
* Distributed computing support
* Real-time data integration
* Advanced uncertainty quantification
* Model performance metrics
* Automated parameter optimization
* Enhanced spatial analysis tools

[1.0.0] - Planned
~~~~~~~~~~~~~~~~~

Planned Additions
^^^^^^^^^^^^^^^^^
* Complete API stabilization
* Comprehensive model validation
* Production-ready deployment tools
* Full documentation coverage
* Complete test coverage (>95%)
* Performance benchmarking
* Integration with other hydrological frameworks

Migration Guide
---------------

From 0.1.0 to Future Versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When upgrading to future versions, please note:

* **API Stability**: The core API is considered stable, but minor changes may occur before 1.0.0
* **Configuration Changes**: Default configurations may be updated for better performance
* **Dependency Updates**: Keep dependencies up to date for security and compatibility
* **Breaking Changes**: Will be clearly documented and migration paths provided

Deprecation Policy
------------------

* **Minor Versions** (0.x.0): May include deprecation warnings for features to be removed
* **Major Versions** (x.0.0): May remove deprecated features with at least one minor version warning
* **Patch Versions** (0.0.x): Will not include breaking changes or deprecations

Contributing
------------

See :doc:`contributing` for information on how to contribute to MarrmotFlow development.

Support
-------

* **Documentation**: https://marrmotflow.readthedocs.io/
* **Issues**: https://github.com/kasra-keshavarz/marrmotflow/issues
* **Discussions**: https://github.com/kasra-keshavarz/marrmotflow/discussions

License
-------

MarrmotFlow is distributed under the terms of the license specified in the LICENSE file.

Authors
-------

* **Kasra Keshavarz** - *Initial development* - University of Calgary

Acknowledgments
---------------

* MARRMOT development team for the underlying hydrological models
* Contributors to the scientific Python ecosystem
* University of Calgary for supporting this research

Release Process
---------------

Our release process follows these steps:

1. **Development**: Features developed on feature branches
2. **Testing**: Comprehensive testing on development branch
3. **Documentation**: Update documentation and changelog
4. **Review**: Code review and quality assurance
5. **Release**: Tag version and publish to PyPI
6. **Announcement**: Update documentation and notify users

Version Numbering
-----------------

We follow Semantic Versioning (SemVer):

* **MAJOR** version for incompatible API changes
* **MINOR** version for backwards-compatible functionality additions  
* **PATCH** version for backwards-compatible bug fixes

Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.
