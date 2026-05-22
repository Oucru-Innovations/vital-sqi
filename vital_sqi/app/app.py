"""Dash app object used across all views.

A single ``dash.Dash`` instance is created here and re-imported by every
view module.  Background callbacks (introduced in Phase 1 for the Compute
view) need a callback manager; we wire a ``DiskcacheManager`` to a local
``.cache/`` directory if ``diskcache`` is installed.  Without it the app
still imports fine — background callbacks just won't be available.
"""

import logging
import os
import pathlib

import dash
import dash_bootstrap_components as dbc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Background-callback manager
# ---------------------------------------------------------------------------

_CACHE_DIR = pathlib.Path(
    os.environ.get("VITAL_SQI_APP_CACHE_DIR", ".cache/vital_sqi_app")
).expanduser()
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

try:
    import diskcache  # noqa: F401  - imported for the side effect of checking availability

    background_callback_manager = dash.DiskcacheManager(
        diskcache.Cache(str(_CACHE_DIR))
    )
    logger.info("Background callbacks enabled; cache at %s", _CACHE_DIR)
except ImportError:  # pragma: no cover - exercised when diskcache is absent
    background_callback_manager = None
    logger.warning(
        "diskcache is not installed; background callbacks will be disabled. "
        "Run `pip install diskcache` to enable long-running computations "
        "in the Compute view."
    )
except Exception:  # pragma: no cover - e.g. cache dir not writable on a host
    # Any non-ImportError failure (unwritable cache dir, locked Cache, …)
    # should degrade to foreground callbacks rather than crash app import.
    background_callback_manager = None
    logger.warning(
        "Background-callback manager init failed; running callbacks in the "
        "foreground. Compute progress updates will be disabled.",
        exc_info=True,
    )

# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    background_callback_manager=background_callback_manager,
)
server = app.server
