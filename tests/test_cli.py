"""Tests for the CLI module."""

import pytest
import json
import tempfile
import os
from click.testing import CliRunner
from pathlib import Path

from marrmotflow.cli import cli


def test_cli_help():
    """Test that the CLI help command works."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert "MarrmotFlow" in result.output
    assert "hydrological modeling workflows" in result.output
    assert "--json" in result.output
    assert "--output-path" in result.output


def test_cli_missing_json():
    """Test that CLI fails without JSON file."""
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_cli_nonexistent_json():
    """Test that CLI fails with nonexistent JSON file."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--json', 'nonexistent.json'])
    assert result.exit_code == 2  # Click returns 2 for file not found
    assert "does not exist" in result.output or "Error" in result.output


def test_cli_invalid_json():
    """Test that CLI fails with invalid JSON."""
    runner = CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"invalid": "json"')  # Invalid JSON
        temp_json = f.name
    
    try:
        result = runner.invoke(cli, ['--json', temp_json])
        assert result.exit_code == 1
    finally:
        os.unlink(temp_json)


def test_cli_empty_json():
    """Test that CLI fails with empty JSON configuration."""
    runner = CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({}, f)
        temp_json = f.name
    
    try:
        result = runner.invoke(cli, ['--json', temp_json])
        assert result.exit_code == 1
        assert "cannot be empty" in result.output
    finally:
        os.unlink(temp_json)


def test_cli_version():
    """Test that the CLI version command works."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--version'])
    assert result.exit_code == 0
    # The version should be displayed
    assert "version" in result.output.lower() or "0.1.0" in result.output


def test_verbose_flag():
    """Test that verbose flag is accepted."""
    runner = CliRunner()
    
    # Create a minimal JSON config that will fail but test verbose flag
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"name": "test", "cat": "nonexistent.shp"}, f)
        temp_json = f.name
    
    try:
        result = runner.invoke(cli, ['--json', temp_json, '--verbose'])
        # Should fail due to missing catchment file, but verbose flag should be processed
        assert result.exit_code == 1
        # We can't easily test verbose output without a valid config, but at least
        # we know the flag is accepted
    finally:
        os.unlink(temp_json)


def test_custom_output_directory():
    """Test that custom output directory is accepted."""
    runner = CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"name": "test", "cat": "nonexistent.shp"}, f)
        temp_json = f.name
    
    try:
        result = runner.invoke(cli, ['--json', temp_json, '--output-path', '/tmp/test_output'])
        # Should fail due to missing catchment file, but output-path flag should be processed
        assert result.exit_code == 1
    finally:
        os.unlink(temp_json)


if __name__ == '__main__':
    pytest.main([__file__])
