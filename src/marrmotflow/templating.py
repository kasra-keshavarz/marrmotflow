"""
This module provides functions to generate textual configuration files
using Jinja2 templating engine for MARRMOT model instantiations.
"""
# third-party libraries
from jinja2 import (
    Environment,
    PackageLoader,
)

from typing import (
    Dict,
    Sequence,
    Union
)

# define types
try:
    from os import PathLike
except ImportError:
    PathLike = str
else:
    PathLike = Union[str, PathLike]


# constants
TEMPLATE_MODEL = "marrmot_models.m.jinja"

# global variables and helper functions
def raise_helper(msg):
    """Jinja2 helper function to raise exceptions."""
    raise Exception(msg)
# Jinja2 environment setup
environment = Environment(
    loader=PackageLoader("marrmotflow", "templates"),
    trim_blocks=True,
    lstrip_blocks=True,
    line_comment_prefix='##',
)
environment.globals['raise'] = raise_helper


def render_models(
    model_files: Sequence[str],
    template_jinja_path: PathLike = TEMPLATE_MODEL, # type: ignore
) -> Dict[str, str]:
    """
    Render model files from a sequence of model names.

    Parameters
    ----------
    model_files : Sequence[str]
        A sequence of model names.
    template_jinja_path : PathLike, optional
        Path to the Jinja2 template file. Default is TEMPLATE_MODEL.

    Returns
    -------
    Dict[str, str]
        A dictionary mapping model names to their rendered content.
    """
    # create the template environment
    template = environment.get_template(template_jinja_path)

    # create content
    content = template.render(models=model_files)

    return content