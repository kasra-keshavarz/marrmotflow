Contributing Guide
=================

We welcome contributions to MarrmotFlow! This guide will help you get started with contributing to the project.

Types of Contributions
----------------------

We welcome several types of contributions:

* **Bug reports**: Help us identify and fix issues
* **Feature requests**: Suggest new functionality
* **Code contributions**: Implement bug fixes or new features
* **Documentation improvements**: Enhance or expand documentation
* **Examples**: Add new usage examples
* **Testing**: Improve test coverage

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

.. code-block:: bash

   git clone https://github.com/your-username/marrmotflow.git
   cd marrmotflow

3. **Create a virtual environment**:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

4. **Install in development mode**:

.. code-block:: bash

   pip install -e ".[dev]"

5. **Install pre-commit hooks**:

.. code-block:: bash

   pre-commit install

Development Workflow
~~~~~~~~~~~~~~~~~~~

1. **Create a new branch** for your changes:

.. code-block:: bash

   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix

2. **Make your changes** following the coding standards below

3. **Run tests** to ensure everything works:

.. code-block:: bash

   pytest

4. **Run code quality checks**:

.. code-block:: bash

   black src/ tests/
   flake8 src/ tests/
   mypy src/

5. **Commit your changes**:

.. code-block:: bash

   git add .
   git commit -m "Add feature: description of your changes"

6. **Push to your fork**:

.. code-block:: bash

   git push origin feature/your-feature-name

7. **Create a Pull Request** on GitHub

Coding Standards
----------------

Code Style
~~~~~~~~~~

We use several tools to maintain code quality:

* **Black**: Code formatting
* **Flake8**: Linting
* **MyPy**: Type checking
* **Pre-commit**: Automated checks

Configuration files are included in the repository:

* ``.pre-commit-config.yaml``
* ``pyproject.toml`` (Black and pytest configuration)
* ``setup.cfg`` (Flake8 and MyPy configuration)

Python Style Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

Follow these guidelines when writing code:

.. code-block:: python

   # Use descriptive variable names
   catchment_boundaries = gpd.read_file("catchments.shp")  # Good
   gdf = gpd.read_file("catchments.shp")  # Avoid

   # Add type hints
   def process_forcing_data(
       forcing_file: PathLike,
       variables: Dict[str, str]
   ) -> xr.Dataset:
       """Process forcing data from file."""
       pass

   # Use docstrings for all public functions
   def calculate_pet(
       temperature: np.ndarray,
       method: str = "penman_monteith"
   ) -> np.ndarray:
       """
       Calculate potential evapotranspiration.
       
       Parameters
       ----------
       temperature : np.ndarray
           Daily temperature values in Celsius
       method : str, optional
           PET calculation method, by default "penman_monteith"
           
       Returns
       -------
       np.ndarray
           Daily PET values in mm/day
       """
       pass

Documentation Style
~~~~~~~~~~~~~~~~~~~

Use NumPy-style docstrings:

.. code-block:: python

   def example_function(param1: str, param2: int = 10) -> bool:
       """
       Brief description of the function.
       
       Longer description if needed, explaining the purpose
       and usage of the function.
       
       Parameters
       ----------
       param1 : str
           Description of param1
       param2 : int, optional
           Description of param2, by default 10
           
       Returns
       -------
       bool
           Description of return value
           
       Raises
       ------
       ValueError
           Description of when this exception is raised
           
       Examples
       --------
       >>> result = example_function("test", 5)
       >>> print(result)
       True
       """
       return True

Testing Guidelines
------------------

Test Structure
~~~~~~~~~~~~~~

Tests are located in the ``tests/`` directory and follow this structure:

.. code-block:: text

   tests/
   ├── __init__.py
   ├── test_core.py
   ├── test_templating.py
   ├── test_default_dicts.py
   └── data/
       ├── test_catchments.shp
       └── test_climate.nc

Writing Tests
~~~~~~~~~~~~~

Use pytest for testing:

.. code-block:: python

   import pytest
   import numpy as np
   from marrmotflow import MARRMOTWorkflow

   def test_workflow_creation():
       """Test basic workflow creation."""
       workflow = MARRMOTWorkflow(
           name="TestWorkflow",
           cat="tests/data/test_catchments.shp",
           forcing_files="tests/data/test_climate.nc",
           forcing_vars={"precip": "precipitation", "temp": "temperature"}
       )
       
       assert workflow.name == "TestWorkflow"
       assert workflow.model_number == [7, 37]  # Default models

   def test_invalid_model_number():
       """Test that invalid model numbers raise appropriate errors."""
       with pytest.raises(ValueError, match="Unsupported model number"):
           MARRMOTWorkflow(
               name="TestWorkflow",
               cat="tests/data/test_catchments.shp",
               forcing_files="tests/data/test_climate.nc",
               forcing_vars={"precip": "precipitation", "temp": "temperature"},
               model_number=999  # Invalid model number
           )

   @pytest.fixture
   def sample_workflow():
       """Fixture providing a sample workflow for testing."""
       return MARRMOTWorkflow(
           name="SampleWorkflow",
           cat="tests/data/test_catchments.shp",
           forcing_files="tests/data/test_climate.nc",
           forcing_vars={"precip": "precipitation", "temp": "temperature"}
       )

   def test_workflow_with_fixture(sample_workflow):
       """Test using a pytest fixture."""
       assert sample_workflow.name == "SampleWorkflow"

Running Tests
~~~~~~~~~~~~~

Run tests with different options:

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=marrmotflow

   # Run specific test file
   pytest tests/test_core.py

   # Run specific test
   pytest tests/test_core.py::test_workflow_creation

   # Run with verbose output
   pytest -v

Documentation Contributions
---------------------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

The documentation is built using Sphinx:

.. code-block:: bash

   cd docs/
   make html

The built documentation will be in ``docs/_build/html/``.

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~

When writing documentation:

1. **Use clear, concise language**
2. **Include code examples** for complex concepts
3. **Test all code examples** to ensure they work
4. **Use proper reStructuredText formatting**
5. **Link to related sections** when appropriate

Adding Examples
~~~~~~~~~~~~~~~

When adding new examples:

1. Create a new ``.rst`` file in ``docs/examples/``
2. Follow the existing example structure
3. Include complete, runnable code
4. Add the example to ``docs/examples/index.rst``
5. Test the example thoroughly

Pull Request Process
--------------------

Before Submitting
~~~~~~~~~~~~~~~~~

Ensure your pull request:

* **Passes all tests**: ``pytest``
* **Follows code style**: ``black``, ``flake8``, ``mypy``
* **Includes appropriate tests** for new functionality
* **Updates documentation** if needed
* **Has a clear commit message**

Pull Request Template
~~~~~~~~~~~~~~~~~~~~

Use this template for your pull request description:

.. code-block:: text

   ## Description
   Brief description of the changes made.

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Code refactoring
   - [ ] Other (please describe)

   ## Testing
   - [ ] I have added tests that prove my fix is effective or that my feature works
   - [ ] New and existing unit tests pass locally with my changes
   - [ ] I have run the code quality checks (black, flake8, mypy)

   ## Documentation
   - [ ] I have updated the documentation accordingly
   - [ ] I have added docstrings to new functions/classes

   ## Additional Notes
   Any additional information about the implementation or considerations.

Review Process
~~~~~~~~~~~~~~

After submitting your pull request:

1. **Automated checks** will run (tests, code quality)
2. **Maintainers will review** your code
3. **Address feedback** by making additional commits
4. **Final approval** and merge by maintainers

Issue Reporting
---------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, please include:

.. code-block:: text

   **Describe the bug**
   A clear and concise description of what the bug is.

   **To Reproduce**
   Steps to reproduce the behavior:
   1. Go to '...'
   2. Click on '....'
   3. Scroll down to '....'
   4. See error

   **Expected behavior**
   A clear and concise description of what you expected to happen.

   **Environment:**
   - OS: [e.g. macOS 12.0]
   - Python version: [e.g. 3.9.7]
   - MarrmotFlow version: [e.g. 0.1.0]
   - Relevant package versions: [e.g. pandas 1.3.0]

   **Additional context**
   Add any other context about the problem here.

Feature Requests
~~~~~~~~~~~~~~~

When requesting features:

.. code-block:: text

   **Is your feature request related to a problem? Please describe.**
   A clear and concise description of what the problem is.

   **Describe the solution you'd like**
   A clear and concise description of what you want to happen.

   **Describe alternatives you've considered**
   A clear and concise description of any alternative solutions or features you've considered.

   **Additional context**
   Add any other context or screenshots about the feature request here.

Communication
-------------

* **GitHub Issues**: For bug reports and feature requests
* **GitHub Discussions**: For questions and general discussion
* **Pull Requests**: For code contributions

Code of Conduct
---------------

By participating in this project, you agree to abide by our Code of Conduct:

1. **Be respectful** and inclusive
2. **Be collaborative**
3. **Be patient** with newcomers
4. **Give constructive feedback**
5. **Focus on what is best** for the community

Recognition
-----------

All contributors will be recognized in:

* **CONTRIBUTORS.md** file
* **Release notes** for significant contributions
* **Documentation** for major feature additions

Thank you for contributing to MarrmotFlow!
