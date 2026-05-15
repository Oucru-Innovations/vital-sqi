# Configuration file for the Sphinx documentation builder.
#
# Reference: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the package importable so autodoc can pick it up
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../vital_sqi/"))
)

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------

project = "vital_sqi"
copyright = "2026, Oucru"
author = "Oucru"

# ---------------------------------------------------------------------------
# Sphinx extensions
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "myst_nb",                 # executable notebooks + native Plotly support
    "sphinxcontrib.jquery",    # ensures jQuery for any extension that expects it
]

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
master_doc = "index"
rst_epilog = ".. |project_name| replace:: %s" % project

# ---------------------------------------------------------------------------
# myst-nb / MyST configuration
# ---------------------------------------------------------------------------

# Execute notebooks at build time and embed outputs. Cached re-runs are skipped.
# "auto" runs notebooks that don't have stored outputs OR whose source changed.
nb_execution_mode = "auto"
nb_execution_timeout = 300        # seconds per notebook
nb_execution_allow_errors = False
nb_render_plugin = "default"

# Don't fail RTD when a notebook can't find its sample data — render anyway.
nb_execution_excludepatterns = []

# Keep Plotly figures interactive in HTML output. require.js is provided by
# the notebook via the default mime bundle; myst-nb renders it automatically.
nb_mime_priority_overrides = [
    ("html", "application/vnd.plotly.v1+json", 10),
    ("html", "text/html", 20),
    ("html", "image/svg+xml", 30),
    ("html", "image/png", 40),
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "html_admonition",
    "html_image",
]

# ---------------------------------------------------------------------------
# autodoc / napoleon
# ---------------------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "exclude-members": "__weakref__,__dict__",
}
autodoc_typehints = "description"
autosummary_generate = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# Mock only dependencies that aren't installable on RTD. Do NOT mock plotly,
# pandas, or numpy — they ship in our requirements.txt and mocking them
# breaks notebook execution.
autodoc_mock_imports = [
    "dash",
    "dash_bootstrap_components",
    "dash_core_components",
    "dash_html_components",
    "pyflowchart",
    "pmdarima",
    "vitalDSP",
    "soundfile",
    "hrvanalysis",
]

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python":     ("https://docs.python.org/3/", None),
    "numpy":      ("https://numpy.org/doc/stable/", None),
    "scipy":      ("https://docs.scipy.org/doc/scipy/", None),
    "pandas":     ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_baseurl = "https://vital-sqi.readthedocs.io/"

html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    # Collapse sub-trees by default; user expands by clicking.
    "collapse_navigation": True,
    "sticky_navigation": True,
    # Depth 4 is enough for "Package > Module > Class > Method" without
    # producing a wall-of-text sidebar.
    "navigation_depth": 4,
    "includehidden": True,
    # Only show page-title entries (not every heading inside every page) so
    # the sidebar stays scannable.
    "titles_only": True,
    "style_nav_header_background": "#2980B9",
}

html_sidebars = {
    "**": [
        "globaltoc.html",
        "relations.html",
        "sourcelink.html",
        "searchbox.html",
    ]
}

# ---------------------------------------------------------------------------
# Warning suppression
# ---------------------------------------------------------------------------

suppress_warnings = [
    "myst_nb",
    "myst.header",
    "autosummary",
    "ref.duplicate_label",
    # Dataclass fields show up twice (class body + member list) — harmless.
    "ref.python",
    "autodoc.duplicate_object_description",
    "app.add_directive",
]
