# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DiaBayes"
copyright = "%Y, Martijn van den Ende"
author = "Martijn van den Ende"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_togglebutton",
    "hoverxref.extension",
]

pygments_style = "sphinx"
napoleon_numpy_docstring = True

bibtex_default_style = "unsrt"  # or plain, alpha, unsrt, abbrv
bibtex_bibfiles = ["references.bib"]
hoverxref_role_types = {
    "cite": "tooltip",  # Show tooltip for :cite: role
    "numref": "tooltip",  # Show tooltip for :numref: role
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_show_sourcelink = False

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

html_sidebars = {
    "getting-started": [],  # Prevent an empty navigation bar from appearing
    "community": [],
}
