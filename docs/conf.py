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
# import sys
# sys.path.insert(0, os.path.abspath('.'))
# autodoc_mock_imports = ['ctlearn']

# -- Project information -----------------------------------------------------
source_suffix = '.rst'
master_doc = 'index'
project = 'ctlearn_optimizer'
author = 'Juan Alfonso Redondo Pizarro'
year = '2019'
copyright = '{0}, {1}'.format(year, author)
# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autosummary',
              'sphinx.ext.autosectionlabel',
              'sphinx.ext.doctest',
              'sphinx.ext.extlinks',
              'sphinx.ext.ifconfig',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'nbsphinx',
              'autoapi.extension',
              ]

autoapi_dirs = ['../src']
autoapi_add_toctree_entry = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only set the theme if we're building docs locally
    html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {'navigation_depth': -1}

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
