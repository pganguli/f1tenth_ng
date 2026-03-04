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
copyright = "2026, Prateek Ganguli"
author = "Prateek Ganguli"

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

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
