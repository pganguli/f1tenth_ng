"""Sphinx configuration file for the F1TENTH-NG documentation."""
# pylint: disable=invalid-name,redefined-builtin
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../packages/f110_gym/src"))
sys.path.insert(0, os.path.abspath("../../packages/f110_planning/src"))
sys.path.insert(0, os.path.abspath("../../packages/f110_scripts/src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "F1TENTH-NG"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_markdown_builder",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

templates_path = ["_templates"]
exclude_patterns = []

# Mock heavy/graphics dependencies that are unavailable in headless CI.
# autodoc imports every module it documents; pyglet tries to load EGL/OpenGL
# at import time which fails on headless runners.
autodoc_mock_imports = ["pyglet"]

# Suppress warnings that arise from intentional re-export patterns:
# - ref.python: "more than one target" when a symbol lives in both its
#   defining sub-module and a re-exporting parent.
suppress_warnings = ["ref.python", "ref.duplicate", "app.add_directive_to_domain:py"]


# ---------------------------------------------------------------------------
# Prevent duplicate object descriptions for base types
# ---------------------------------------------------------------------------
# Action, BasePlanner and CloudScheduler are defined in f110_planning.base
# and imported at module level by every single planner module.  Those
# planner modules have no __all__, so autodoc would document the types from
# all of them, triggering "duplicate object description" warnings.
# This hook ensures the types are only documented from their defining module.
_PLANNING_BASE_NAMES = frozenset({"Action", "BasePlanner", "CloudScheduler"})


def _skip_planning_base_imports(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    app, what, name, obj, skip, options  # pylint: disable=unused-argument
):
    if name in _PLANNING_BASE_NAMES:
        if getattr(obj, "__module__", "") != "f110_planning.base":
            return True  # skip this re-imported copy
    return skip


def setup(app):
    """Register Sphinx event hooks."""
    app.connect("autodoc-skip-member", _skip_planning_base_imports)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
