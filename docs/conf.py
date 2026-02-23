import os
import sys

# Add project root to sys.path so autodoc can import the package
sys.path.insert(0, os.path.abspath('..'))

project = 'CaGenNet'
author = 'Jordi Abante Llenas and Berta Ros Blanco'
release = '0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx.ext.viewcode',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Mock heavy dependencies on ReadTheDocs to avoid import errors during build
autodoc_mock_imports = [
    'torch', 'pyro', 'sklearn', 'numpy', 'pandas', 'scipy'
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
