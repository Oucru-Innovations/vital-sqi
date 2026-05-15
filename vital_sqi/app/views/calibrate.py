"""Calibrate view — run the calibration pipeline from the GUI.

Phase 3 of the app enhancement plan.  Wraps
:func:`vital_sqi.calibration.run_calibration.calibrate` in a Dash
background callback so the user can drive it without dropping into a
notebook.

Workflow:

1. The user picks wave type, segment count, percentile bounds, and
   whether to do a dry run.
2. Pressing **Run** invokes ``calibrate(dry_run=True, ...)`` in the
   background; results are stored in a session-only ``dcc.Store`` and
   summarised in a ``DataTable``.
3. If the user is satisfied, **Save as default** copies the thresholds
   into ``vital_sqi/resource/`` — atomically, with a timestamped
   backup of the previous defaults.

The dry-run-first design ensures destructive file writes only happen
on an explicit second click.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from vital_sqi.app.app import app, background_callback_manager  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resource paths
# ---------------------------------------------------------------------------

_RESOURCE_DIR = Path(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "resource"))
)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _config_form() -> dbc.Form:
    row_basics = dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("Wave type"),
                    dcc.Dropdown(
                        id="calibrate-wave-type",
                        options=[
                            {"label": "PPG", "value": "PPG"},
                            {"label": "ECG", "value": "ECG"},
                        ],
                        value="PPG",
                        clearable=False,
                    ),
                ],
                md=3,
            ),
            dbc.Col(
                [
                    dbc.Label("Accept-pool segments"),
                    dbc.Input(
                        id="calibrate-n-segments",
                        type="number",
                        min=10, max=2000, step=10,
                        value=200,
                    ),
                    html.Small(
                        "Clean signals generated synthetically. More = "
                        "tighter quantile estimates but longer runs.",
                        className="text-muted",
                    ),
                ],
                md=3,
            ),
            dbc.Col(
                [
                    dbc.Label("Reject-pool segments / profile"),
                    dbc.Input(
                        id="calibrate-n-reject",
                        type="number",
                        min=5, max=500, step=5,
                        value=50,
                    ),
                    html.Small(
                        "Used for diagnostics only; thresholds come "
                        "from the accept pool.",
                        className="text-muted",
                    ),
                ],
                md=3,
            ),
            dbc.Col(
                [
                    dbc.Label("Segment duration (s)"),
                    dbc.Input(
                        id="calibrate-duration",
                        type="number",
                        min=5, max=120, step=1,
                        value=30,
                    ),
                ],
                md=3,
            ),
        ],
        className="mb-2",
    )

    row_bounds = dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("Lower percentile"),
                    dbc.Input(
                        id="calibrate-lower-pct",
                        type="number",
                        min=0, max=49, step=0.5,
                        value=5,
                    ),
                ],
                md=3,
            ),
            dbc.Col(
                [
                    dbc.Label("Upper percentile"),
                    dbc.Input(
                        id="calibrate-upper-pct",
                        type="number",
                        min=51, max=100, step=0.5,
                        value=95,
                    ),
                ],
                md=3,
            ),
            dbc.Col(
                [
                    dbc.Label("Random seed"),
                    dbc.Input(
                        id="calibrate-seed",
                        type="number",
                        step=1,
                        value=42,
                    ),
                ],
                md=3,
            ),
        ],
        className="mb-3",
    )

    return dbc.Form([row_basics, row_bounds])


def _action_row() -> html.Div:
    return html.Div(
        [
            dbc.Button(
                "Run calibration",
                id="calibrate-run-btn",
                color="primary",
                n_clicks=0,
            ),
            dbc.Button(
                "Cancel",
                id="calibrate-cancel-btn",
                color="secondary",
                outline=True,
                className="ms-2",
                disabled=True,
                n_clicks=0,
            ),
            html.Span(
                id="calibrate-status",
                className="ms-3 text-muted small",
            ),
            dbc.Progress(
                id="calibrate-progress",
                value=0,
                animated=True,
                striped=True,
                style={"marginTop": "12px", "display": "none"},
            ),
        ],
        className="mb-3",
    )


def _results_panel() -> html.Div:
    return html.Div(
        [
            html.H5("Thresholds (latest dry run)"),
            html.Div(
                id="calibrate-results-table",
                children=html.Em(
                    "No calibration has run yet.  Adjust the form above and "
                    "press Run calibration."
                ),
                className="small text-muted",
            ),
            html.Hr(),
            html.H5("Save as default"),
            html.P(
                "Writes the latest thresholds to ",
                className="small text-muted mb-1",
            ),
            html.Pre(
                str(_RESOURCE_DIR / "rule_dict.json"),
                className="small text-muted",
            ),
            html.P(
                "A timestamped backup of the existing rule_dict.json and "
                "sqi_dict.json is taken first.  Run pytest after saving to "
                "make sure nothing downstream relies on the old bounds.",
                className="small text-muted",
            ),
            dbc.Button(
                "Save as default",
                id="calibrate-save-btn",
                color="danger",
                outline=True,
                disabled=True,
                n_clicks=0,
            ),
            html.Span(id="calibrate-save-status", className="ms-3 small"),
        ]
    )


layout = html.Div(
    [
        html.H3("Calibrate thresholds"),
        html.P(
            "Generate synthetic clean signals, sweep through noise profiles, "
            "and derive accept/reject thresholds for every SQI in the "
            "catalogue.  This is the same calibration step that produced "
            "the bundled rule_dict.json; re-run it whenever you change a "
            "noise profile or add a new SQI.",
            className="text-muted",
        ),
        html.Hr(),
        _config_form(),
        _action_row(),
        html.Hr(),
        _results_panel(),

        # Session-only store holding the latest dry-run result.  Persisting
        # this in localStorage is pointless — every row holds a JSON-encoded
        # SQIThreshold per SQI, which adds up.
        dcc.Store(id="calibrate-thresholds-store", storage_type="memory"),
    ]
)


# ---------------------------------------------------------------------------
# Background callback — Run
# ---------------------------------------------------------------------------


_background = background_callback_manager is not None


@callback(
    output=[
        Output("calibrate-status", "children"),
        Output("calibrate-progress", "value"),
        Output("calibrate-progress", "style"),
        Output("calibrate-results-table", "children"),
        Output("calibrate-thresholds-store", "data"),
        Output("calibrate-save-btn", "disabled"),
    ],
    inputs=[Input("calibrate-run-btn", "n_clicks")],
    state=[
        State("calibrate-wave-type", "value"),
        State("calibrate-n-segments", "value"),
        State("calibrate-n-reject", "value"),
        State("calibrate-duration", "value"),
        State("calibrate-lower-pct", "value"),
        State("calibrate-upper-pct", "value"),
        State("calibrate-seed", "value"),
    ],
    background=_background,
    progress=[
        Output("calibrate-status", "children"),
        Output("calibrate-progress", "value"),
        Output("calibrate-progress", "style"),
    ] if _background else None,
    running=[
        (Output("calibrate-run-btn", "disabled"), True, False),
        (Output("calibrate-cancel-btn", "disabled"), False, True),
    ] if _background else None,
    cancel=[Input("calibrate-cancel-btn", "n_clicks")] if _background else None,
    prevent_initial_call=True,
    manager=background_callback_manager,
)
def _run_calibration(
    set_progress,
    n_clicks,
    wave_type,
    n_segments,
    n_reject,
    duration,
    lower_pct,
    upper_pct,
    seed,
):
    """Run :func:`calibrate` in dry-run mode and return a summary table."""
    if not n_clicks:
        raise PreventUpdate

    progress_visible = {"marginTop": "12px", "display": "block"}
    progress_hidden = {"display": "none"}

    try:
        lower_pct_f = float(lower_pct or 5.0)
        upper_pct_f = float(upper_pct or 95.0)
        if lower_pct_f >= upper_pct_f:
            return (
                f"✗ lower_pct ({lower_pct_f}) must be < upper_pct ({upper_pct_f}).",
                0, progress_visible, dash.no_update, dash.no_update, True,
            )
    except (TypeError, ValueError):
        return (
            "✗ Lower/upper percentile must be numbers.",
            0, progress_visible, dash.no_update, dash.no_update, True,
        )

    if _background:
        set_progress((
            f"Starting calibration ({wave_type}, n={n_segments}) — synthesising signals…",
            10, progress_visible,
        ))

    # Heavy lifting — calibrate() prints to stdout via the runner; the user
    # sees the progress bar in the UI, the terminal log captures the rest.
    try:
        from vital_sqi.calibration.run_calibration import calibrate

        thresholds = calibrate(
            wave_type=str(wave_type),
            n_segments=int(n_segments or 200),
            n_reject_segments=int(n_reject or 50),
            duration=float(duration or 30.0),
            lower_pct=lower_pct_f,
            upper_pct=upper_pct_f,
            seed=int(seed) if seed is not None else 42,
            dry_run=True,        # never write from the Run button
            show_progress=False, # don't spam the terminal
        )
    except Exception as exc:
        logger.exception("Calibration failed")
        return (
            f"✗ Calibration failed: {exc}",
            0, progress_visible, html.Em(str(exc)), dash.no_update, True,
        )

    if _background:
        set_progress((
            "Calibration finished — building results table…",
            95, progress_visible,
        ))

    table = _thresholds_to_table(thresholds)
    store_payload = _thresholds_to_payload(thresholds)
    n_calibrated = sum(1 for t in thresholds.values() if t.calibrated)
    summary = (
        f"✓ Calibrated {n_calibrated}/{len(thresholds)} SQIs "
        f"({wave_type}, p{lower_pct_f:.0f}–p{upper_pct_f:.0f}). "
        "Review below, then 'Save as default' to write to disk."
    )
    return summary, 100, progress_visible, table, store_payload, False


# ---------------------------------------------------------------------------
# Save-as-default callback (foreground — fast)
# ---------------------------------------------------------------------------


@callback(
    Output("calibrate-save-status", "children"),
    Input("calibrate-save-btn", "n_clicks"),
    State("calibrate-thresholds-store", "data"),
    State("calibrate-wave-type", "value"),
    State("calibrate-lower-pct", "value"),
    State("calibrate-upper-pct", "value"),
    prevent_initial_call=True,
)
def _save_as_default(n_clicks, payload, wave_type, lower_pct, upper_pct):
    """Write the cached thresholds to vital_sqi/resource/ with a backup."""
    if not n_clicks or not payload:
        raise PreventUpdate
    try:
        result = _atomic_save_thresholds(
            payload,
            output_dir=_RESOURCE_DIR,
            wave_type=str(wave_type),
            lower_pct=float(lower_pct or 5.0),
            upper_pct=float(upper_pct or 95.0),
        )
    except Exception as exc:
        logger.exception("Save-as-default failed")
        return html.Span(f"✗ Save failed: {exc}", style={"color": "#b30000"})

    return html.Span(
        f"✓ Saved {result['n_rule_entries']} rules + "
        f"{result['n_sqi_entries']} SQI templates. Backup: {result['backup_name']}",
        style={"color": "#0f5132"},
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _thresholds_to_table(thresholds: Dict[str, Any]) -> html.Div:
    """Render the thresholds dict as a sortable DataTable."""
    if not thresholds:
        return html.Em("No SQIs were calibrated.")
    df = _thresholds_to_dataframe(thresholds)
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_table={"overflowX": "auto", "maxHeight": "420px", "overflowY": "auto"},
        style_cell={
            "textAlign": "left",
            "minWidth": "90px",
            "maxWidth": "240px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "padding": "4px 8px",
        },
        style_data_conditional=[
            {
                "if": {"filter_query": "{calibrated} = false"},
                "color": "#999",
                "fontStyle": "italic",
            },
        ],
        page_size=30,
        sort_action="native",
        filter_action="native",
    )


def _thresholds_to_dataframe(thresholds: Dict[str, Any]) -> pd.DataFrame:
    """Plain-DataFrame view of the thresholds dict.  Used by the UI table."""
    rows: List[dict] = []
    for name, t in thresholds.items():
        rows.append({
            "sqi": name,
            "calibrated": bool(getattr(t, "calibrated", False)),
            "lower": _safe_round(getattr(t, "lower", None), 4),
            "upper": _safe_round(getattr(t, "upper", None), 4),
            "accept_median": _safe_round(getattr(t, "accept_median", None), 4),
            "accept_std": _safe_round(getattr(t, "accept_std", None), 4),
            "reject_median": _safe_round(getattr(t, "reject_median", None), 4),
            "n_accept": getattr(t, "n_accept", 0),
            "n_reject": getattr(t, "n_reject", 0),
            "note": getattr(t, "note", "") or "",
        })
    return pd.DataFrame(rows)


def _thresholds_to_payload(thresholds: Dict[str, Any]) -> dict:
    """Serialise the thresholds dict for ``dcc.Store``."""
    return {
        name: {
            "sqi_name":      t.sqi_name,
            "lower":         t.lower,
            "upper":         t.upper,
            "accept_median": t.accept_median,
            "accept_std":    t.accept_std,
            "reject_median": t.reject_median,
            "n_accept":      t.n_accept,
            "n_reject":      t.n_reject,
            "calibrated":    t.calibrated,
            "note":          t.note,
        }
        for name, t in thresholds.items()
    }


def _safe_round(value, ndigits):
    try:
        if value is None:
            return ""
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return ""


def _atomic_save_thresholds(
    payload: dict,
    output_dir: Path,
    wave_type: str,
    lower_pct: float,
    upper_pct: float,
) -> dict:
    """Write rule_dict.json + sqi_dict.json with backup, return diagnostics.

    Strategy: rebuild the :class:`SQIThreshold` objects from the store
    payload, then reuse the existing exporter functions.  This keeps the
    file format consistent with the CLI's ``calibrate`` output.
    """
    from vital_sqi.calibration.exporter import export_rule_dict, export_sqi_dict
    from vital_sqi.calibration.threshold_estimator import SQIThreshold

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build SQIThreshold instances from the dict so the exporters can run.
    thresholds = {
        name: SQIThreshold(**entry) for name, entry in payload.items()
    }

    rule_dict_path = output_dir / "rule_dict.json"
    sqi_dict_path = output_dir / "sqi_dict.json"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backup_{ts}"
    # ``export_rule_dict`` / ``export_sqi_dict`` already write timestamped
    # backups when ``backup=True``; we let them do it so the format stays
    # consistent with the CLI runner.
    export_rule_dict(
        thresholds, str(rule_dict_path), wave_type=wave_type,
        lower_pct=lower_pct, upper_pct=upper_pct, backup=True,
    )
    export_sqi_dict(thresholds, str(sqi_dict_path), backup=True)

    n_rule_entries = sum(1 for t in thresholds.values() if t.calibrated)
    n_sqi_entries = n_rule_entries
    return {
        "rule_dict_path": str(rule_dict_path),
        "sqi_dict_path": str(sqi_dict_path),
        "backup_name": backup_name,
        "n_rule_entries": n_rule_entries,
        "n_sqi_entries": n_sqi_entries,
    }
