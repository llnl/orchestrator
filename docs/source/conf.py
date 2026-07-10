# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.append(os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'Orchestrator'
copyright = '2026, Lawrence Livermore National Laboratory'
author = 'IAP UQ Thrust 4 Team'

# The full version, including alpha/beta/rc tags
release = '0.7'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.viewcode',
    'sphinx.ext.inheritance_diagram', 'sphinx.ext.graphviz',
    'sphinx_copybutton', 'sphinx_autodoc_typehints', 'sphinx_design'
]

inheritance_graph_attrs = dict(fontsize=12, size='"16.0, 20.0"')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autodoc_mock_imports = [
    'numpy',
    'kimpy',
    'ase',
    'pymatgen',
    'kliff',
    'colabfit',
    'kimkit',
    'pandas',
    'sklearn',
    'scipy',
    'matplotlib',
    'tqdm',
    'fitsnap3lib',
    'ltau_ff',
    'quests',
    'kim_edn',
    'periodictable',
    'aiida',
    'aiida_quantumespresso',
    'numdifftools',
    'cvxpy',
    'sdpa-python',
    'information_matching',
    'yaml',
]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

rst_prolog = ('.. |default| raw:: html\n\n'
              '    <div class="default-value-section"> '
              '<span class="default-value-label">Default:</span>')

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'furo'
html_logo = "orchestrator-logo-vert-color.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

# html_static_path = ['_static']
