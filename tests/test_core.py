"""Tests for the core module."""

import pytest
from marrmotflow.core import example_function, add_numbers


def test_example_function():
    """Test the example function."""
    result = example_function()
    assert result == "Hello from MarrmotFlow!"
    assert isinstance(result, str)


def test_add_numbers():
    """Test the add_numbers function."""
    assert add_numbers(2, 3) == 5
    assert add_numbers(0, 0) == 0
    assert add_numbers(-1, 1) == 0
    assert add_numbers(2.5, 3.5) == 6.0


def test_add_numbers_types():
    """Test that add_numbers works with different numeric types."""
    assert add_numbers(2, 3.0) == 5.0
    assert add_numbers(2.0, 3) == 5.0
