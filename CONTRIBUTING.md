# Contributing to MarrmotFlow

Thank you for your interest in contributing to MarrmotFlow! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/marrmotflow.git
   cd marrmotflow
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Making Changes

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and add tests for new functionality

3. Run the test suite:
   ```bash
   pytest
   ```

4. Format your code:
   ```bash
   black src/ tests/
   ```

5. Check types:
   ```bash
   mypy src/
   ```

6. Commit your changes:
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

7. Push to your fork and submit a pull request

## Code Style

- Use [Black](https://black.readthedocs.io/) for code formatting
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Add type hints for all function parameters and return values
- Write docstrings for all public functions and classes

## Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Add tests for new functionality
3. Ensure all checks pass (tests, linting, type checking)
4. Request review from maintainers
