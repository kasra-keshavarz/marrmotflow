"""Command-line interface for MarrmotFlow."""

import os
import sys
import click
from pathlib import Path
from typing import Optional

from .core import MARRMOTWorkflow


@click.command()
@click.option(
    '--json',
    'json_file',
    type=click.Path(exists=True, readable=True, path_type=Path),
    required=True,
    help='Path to JSON configuration file containing workflow parameters'
)
@click.option(
    '--output-path',
    'output_path',
    type=click.Path(path_type=Path),
    default='./marrmot_output',
    help='Output directory path for saving results (default: ./marrmot_output)'
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='Enable verbose output'
)
def run_workflow(json_file: Path, output_path: Path, verbose: bool):
    """
    Run MARRMOTWorkflow from a JSON configuration file.
    
    This command reads a JSON configuration file, creates a MARRMOTWorkflow instance,
    runs the workflow, and saves the results to the specified output directory.
    
    Example:
        marrmotflow --json config.json --output ./results
    """
    try:
        if verbose:
            click.echo(f"Reading configuration from: {json_file}")
        
        # Create MARRMOTWorkflow instance from JSON file
        workflow = MARRMOTWorkflow.from_json_file(str(json_file))
        
        if verbose:
            click.echo(f"Created workflow: {workflow.name}")
            click.echo(f"Model numbers: {workflow.model_number}")
            click.echo(f"PET method: {workflow.pet_method}")
        
        # Run the workflow
        click.echo("Running workflow...")
        result = workflow.run()
        
        if verbose:
            click.echo(result)
        
        # Save the results
        click.echo(f"Saving results to: {output_path}")
        save_result = workflow.save(str(output_path))
        
        if verbose:
            click.echo(save_result)
        
        click.echo("✓ Workflow completed successfully!")
        
    except FileNotFoundError as e:
        click.echo(f"Error: Configuration file not found: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: Invalid configuration: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: Workflow execution failed: {e}", err=True)
        sys.exit(1)


@click.group()
@click.version_option()
def cli():
    """
    MarrmotFlow - A Python package for hydrological modeling workflows.
    
    This CLI provides tools for running MARRMOT model workflows from configuration files.
    """
    pass


# Add the run_workflow command to the CLI group
cli.add_command(run_workflow, name="run")


# Main entry point
def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
