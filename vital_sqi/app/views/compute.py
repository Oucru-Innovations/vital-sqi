"""Compute view — turn a raw waveform into an SQI DataFrame.

This is the entry point introduced by Phase 1 of the app enhancement plan.
The user uploads a recording (CSV / EDF / WFDB), chooses pipeline
parameters in a form, and presses **Run**.  The actual ``extract_sqi``
call runs as a Dash background callback so the UI stays responsive.

After a successful run the resulting DataFrame is written to the shared
``dcc.Store(id="dataframe")`` so the Inspect and Export views can see it.

Layout overview
---------------
    +-------------------------------------------------------------+
    |  [Upload raw waveform]   filename indicator                 |
    |  -- form ------------------------------------------------   |
    |   wave_type  | sampling_rate | duration | overlap           |
    |   sqi_dict   | n_jobs                                       |
    |  -- actions --                                              |
    |   [ Run ]     idle / running / done / error  +  progress    |
    |  -- result snapshot --                                      |
    |   DataTable preview of the SQI DataFrame                    |
    +-------------------------------------------------------------+
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from typing import Optional, Tuple

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from vital_sqi.app.app import app, background_callback_manager
from vital_sqi.app.util.waveform_loader import (
    LoadedWaveform,
    WaveformLoaderError,
    introspect_columns,
    load_from_upload,
)
from vital_sqi.common.utils import generate_timestamp
from vital_sqi.common.utils import format_milestone
from vital_sqi.pipeline.pipeline_functions import extract_sqi
from vital_sqi.preprocess.segment_split import split_segment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths to bundled assets
# ---------------------------------------------------------------------------

_BUNDLED_SQI_DICT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "resource", "sqi_dict.json"
)
_BUNDLED_SQI_DICT_PATH = os.path.normpath(_BUNDLED_SQI_DICT_PATH)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _upload_zone() -> html.Div:
    return dcc.Upload(
        id="compute-upload-waveform",
        children=html.Div(
            [
                html.Strong("Drag & drop"),
                html.Span(" or "),
                html.A("select a recording"),
                html.Br(),
                html.Small(
                    "CSV (flat or Oucru row-per-second), EDF/EDF+, or WFDB (.hea + .dat)",
                    style={"color": "#666"},
                ),
            ]
        ),
        style={
            "width": "100%",
            "minHeight": "90px",
            "padding": "16px",
            "lineHeight": "1.4",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "8px",
            "textAlign": "center",
            "margin": "2px 0 16px 0",
            "background": "#fafafa",
        },
        multiple=False,
    )


def _config_form() -> dbc.Form:
    row_1 = dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("Wave type"),
                    dcc.Dropdown(
                        id="compute-wave-type",
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
                    dbc.Label("Sampling rate (Hz)"),
                    dbc.Input(
                        id="compute-sampling-rate",
                        type="number",
                        min=1,
                        step=1,
                        placeholder="auto (from timestamps) or e.g. 100",
                    ),
                ],
                md=3,
            ),
            dbc.Col(
                [
                    dbc.Label("Segment duration (s)"),
                    dbc.Input(
                        id="compute-duration",
                        type="number",
                        min=1,
                        step=1,
                        value=30,
                    ),
                ],
                md=3,
            ),
            dbc.Col(
                [
                    dbc.Label("Overlap (s)"),
                    dbc.Input(
                        id="compute-overlap",
                        type="number",
                        min=0,
                        step=1,
                        value=0,
                    ),
                ],
                md=3,
            ),
        ],
        className="mb-2",
    )

    # Signal-column row.  Hidden until a file is staged; once the user
    # uploads a recording, introspect_columns() populates the dropdown
    # with every viable column (array-valued for Oucru-style CSV, plain
    # numeric for flat CSV).
    row_signal_col = dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("Signal column"),
                    dcc.Dropdown(
                        id="compute-signal-column",
                        placeholder="Auto (drop a file first)",
                        clearable=False,
                    ),
                    html.Small(
                        "If your file has multiple candidate columns (e.g. "
                        "pleth/red/ir in an Oucru SmartCare export), pick "
                        "the one to use.",
                        className="text-muted",
                    ),
                ],
                md=12,
            ),
        ],
        id="compute-signal-column-row",
        style={"display": "none"},
        className="mb-2",
    )

    row_2 = dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("SQI dictionary"),
                    dcc.RadioItems(
                        id="compute-sqi-dict-source",
                        options=[
                            {"label": "  Bundled calibrated default", "value": "bundled"},
                            {"label": "  Upload custom JSON", "value": "custom"},
                        ],
                        value="bundled",
                        labelStyle={"display": "block"},
                    ),
                    dcc.Upload(
                        id="compute-sqi-dict-upload",
                        children=html.Div(["Drag & drop or ", html.A("select")]),
                        style={
                            "width": "100%",
                            "padding": "8px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "4px",
                            "textAlign": "center",
                            "marginTop": "4px",
                            "display": "none",
                        },
                        multiple=False,
                    ),
                ],
                md=6,
            ),
            dbc.Col(
                [
                    dbc.Label("Parallel workers"),
                    dbc.Input(
                        id="compute-n-jobs",
                        type="number",
                        min=1,
                        max=16,
                        step=1,
                        value=1,
                    ),
                ],
                md=3,
            ),
        ],
        className="mb-2",
    )

    return dbc.Form([row_1, row_signal_col, row_2])


def _status_panel() -> html.Div:
    return html.Div(
        [
            dbc.Button(
                "Run",
                id="compute-run-btn",
                color="primary",
                n_clicks=0,
                disabled=True,
            ),
            dbc.Button(
                "Cancel",
                id="compute-cancel-btn",
                color="secondary",
                outline=True,
                className="ms-2",
                disabled=True,
            ),
            html.Span(id="compute-status", className="ms-3 text-muted"),
            dbc.Progress(
                id="compute-progress",
                value=0,
                animated=True,
                striped=True,
                style={"marginTop": "12px", "display": "none"},
            ),
        ],
        className="mb-3",
    )


def _result_preview() -> html.Div:
    return html.Div(id="compute-result-preview")


def _sqi_table_upload_card() -> dbc.Card:
    """Optional 'I already have a pre-computed SQI table' shortcut.

    Skips the whole pipeline-run path: the user drops a CSV/JSON of
    per-segment SQI values and we write it straight into the
    ``dataframe`` store so Inspect + Export unlock.  Useful when the
    SQIs came from an external tool or from an earlier Compute run.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6("Already have an SQI table?", className="card-title"),
                html.P(
                    "Drop a CSV / JSON with one row per segment and one "
                    "column per SQI.  Skips the recording-upload + pipeline "
                    "step and goes straight to Inspect / Export.",
                    className="small text-muted mb-2",
                ),
                dcc.Upload(
                    id="compute-sqi-table-upload",
                    children=html.Div(
                        [
                            html.Strong("Drag & drop"),
                            html.Span(" or "),
                            html.A("select an SQI table"),
                        ]
                    ),
                    style={
                        "width": "100%",
                        "minHeight": "60px",
                        "padding": "12px",
                        "lineHeight": "1.4",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "6px",
                        "textAlign": "center",
                        "margin": "4px 0",
                        "background": "#fafafa",
                    },
                    multiple=False,
                ),
                html.Div(
                    id="compute-sqi-table-status",
                    className="small text-muted mt-2",
                ),
            ]
        ),
        className="mt-3",
    )


layout = html.Div(
    [
        html.H3("Compute SQIs from a raw recording"),
        html.P(
            "Upload a waveform, choose pipeline parameters, then press Run. "
            "The resulting SQI table flows directly into the Inspect view; "
            "from there it can be classified and exported.",
            className="text-muted",
        ),
        html.Hr(),
        _upload_zone(),
        # Hidden filename indicator that lights up once an upload is staged.
        html.Div(id="compute-filename", className="mb-2 text-muted"),
        # Stash the staged upload (raw data URL + filename) so the long
        # callback can read it without re-uploading.
        dcc.Store(id="compute-staged-upload"),
        # Stash a parsed snapshot for the preview (kept small).
        dcc.Store(id="compute-loaded-waveform"),
        # Stash the parsed-but-uncomputed SQI dict (when "custom" is chosen).
        dcc.Store(id="compute-sqi-dict-staged"),
        _config_form(),
        _status_panel(),
        html.Hr(),
        html.H5("Preview"),
        _result_preview(),
        html.Hr(),
        _sqi_table_upload_card(),
    ]
)


# ---------------------------------------------------------------------------
# Helpers used by both the staging and the background callback
# ---------------------------------------------------------------------------


def _resolve_sqi_dict_path(
    source: str, custom_payload: Optional[dict]
) -> Tuple[str, Optional[str]]:
    """Return (path_to_sqi_dict, temp_dir_to_clean_up_or_None)."""
    if source == "bundled" or not custom_payload:
        return _BUNDLED_SQI_DICT_PATH, None
    tmp_dir = tempfile.mkdtemp(prefix="vital_sqi_app_sqi_dict_")
    path = os.path.join(tmp_dir, "sqi_dict.json")
    with open(path, "w") as fh:
        json.dump(custom_payload, fh)
    return path, tmp_dir


def _segments_dataframe(
    signal: np.ndarray, sampling_rate: float
) -> pd.DataFrame:
    """Wrap a 1-D signal into the (timestamps, signal) DataFrame split_segment expects."""
    timestamps = generate_timestamp(None, sampling_rate, len(signal))
    return pd.DataFrame({"timestamps": timestamps, "signal": signal})


def _df_preview(df: pd.DataFrame, max_rows: int = 10) -> html.Div:
    """Render a small DataTable preview of the SQI DataFrame."""
    if df is None or df.empty:
        return html.Em("No data to preview.")
    preview = df.head(max_rows).round(4)
    return dash_table.DataTable(
        data=preview.to_dict("records"),
        columns=[{"name": c, "id": c} for c in preview.columns],
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "right",
            "minWidth": "80px",
            "maxWidth": "180px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        page_size=max_rows,
    )


# ---------------------------------------------------------------------------
# Callbacks — lightweight (foreground)
# ---------------------------------------------------------------------------


@callback(
    Output("compute-sqi-dict-upload", "style"),
    Input("compute-sqi-dict-source", "value"),
    prevent_initial_call=True,
)
def _toggle_custom_sqi_uploader(source: str):
    base_style = {
        "width": "100%",
        "padding": "8px",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "4px",
        "textAlign": "center",
        "marginTop": "4px",
    }
    if source == "custom":
        return {**base_style, "display": "block"}
    return {**base_style, "display": "none"}


@callback(
    Output("compute-staged-upload", "data"),
    Output("compute-filename", "children"),
    Output("compute-loaded-waveform", "data"),
    Output("compute-run-btn", "disabled"),
    Output("compute-status", "children", allow_duplicate=True),
    Output("compute-signal-column", "options"),
    Output("compute-signal-column", "value"),
    Output("compute-signal-column-row", "style"),
    Input("compute-upload-waveform", "contents"),
    Input("compute-wave-type", "value"),
    State("compute-upload-waveform", "filename"),
    State("compute-sampling-rate", "value"),
    State("compute-signal-column", "value"),
    prevent_initial_call=True,
)
def _stage_upload(contents, wave_type, filename, sampling_rate, current_column):
    """When the user drops a recording, introspect columns and unlock Run.

    Re-runs when the user toggles wave_type so the "preferred" column
    changes (ECG → ``ecg``, PPG → ``pleth``).
    """
    if not contents or not filename:
        raise PreventUpdate

    # First introspect the file so we can populate the dropdown even if the
    # auto-pick load fails.  This way the user always sees the candidates.
    candidates = introspect_columns(contents, filename, wave_type)
    column_options = [
        {
            "label": f"{c.name}  —  {c.preview}" + ("  (recommended)" if c.preferred else ""),
            "value": c.name,
        }
        for c in candidates
    ]
    # Preserve the user's current choice when it's still valid; otherwise
    # default to the first (preferred) candidate.
    valid_names = [c.name for c in candidates]
    if current_column in valid_names:
        selected_column = current_column
    elif candidates:
        selected_column = candidates[0].name
    else:
        selected_column = None
    column_row_style = {"display": "block"} if column_options else {"display": "none"}

    try:
        lw = load_from_upload(
            contents=contents,
            filename=filename,
            wave_type=wave_type,
            sampling_rate=sampling_rate,
            signal_column=selected_column,
        )
    except WaveformLoaderError as err:
        logger.warning("Upload rejected: %s", err)
        return (
            None,
            html.Span(f"✗ {err}", style={"color": "#b30000"}),
            None,
            True,
            "",
            column_options,
            selected_column,
            column_row_style,
        )

    staged = {
        "contents": contents,
        "filename": filename,
        "signal_column": selected_column,
    }
    summary = (
        f"✓ {filename} — {len(lw.signal)} samples @ {lw.sampling_rate:.2f} Hz "
        f"({len(lw.signal) / lw.sampling_rate:.1f} s)"
    )
    return (
        staged,
        summary,
        lw.to_store(),
        False,
        "",
        column_options,
        selected_column,
        column_row_style,
    )


@callback(
    Output("compute-staged-upload", "data", allow_duplicate=True),
    Output("compute-loaded-waveform", "data", allow_duplicate=True),
    Output("compute-filename", "children", allow_duplicate=True),
    Output("compute-run-btn", "disabled", allow_duplicate=True),
    Input("compute-signal-column", "value"),
    State("compute-upload-waveform", "contents"),
    State("compute-upload-waveform", "filename"),
    State("compute-wave-type", "value"),
    State("compute-sampling-rate", "value"),
    prevent_initial_call=True,
)
def _restage_with_new_column(column, contents, filename, wave_type, sampling_rate):
    """Re-run the loader when the user picks a different signal column."""
    if not column or not contents or not filename:
        raise PreventUpdate
    try:
        lw = load_from_upload(
            contents=contents,
            filename=filename,
            wave_type=wave_type,
            sampling_rate=sampling_rate,
            signal_column=column,
        )
    except WaveformLoaderError as err:
        logger.warning("Re-stage failed (%s): %s", column, err)
        return (
            None,
            None,
            html.Span(f"✗ {err}", style={"color": "#b30000"}),
            True,
        )
    staged = {
        "contents": contents,
        "filename": filename,
        "signal_column": column,
    }
    summary = (
        f"✓ {filename} ({column}) — {len(lw.signal)} samples "
        f"@ {lw.sampling_rate:.2f} Hz ({len(lw.signal) / lw.sampling_rate:.1f} s)"
    )
    return staged, lw.to_store(), summary, False


@callback(
    Output("compute-sqi-dict-staged", "data"),
    Input("compute-sqi-dict-upload", "contents"),
    State("compute-sqi-dict-upload", "filename"),
    prevent_initial_call=True,
)
def _stage_sqi_dict(contents, filename):
    if not contents or not filename:
        raise PreventUpdate
    try:
        _, b64 = contents.split(",", 1)
        import base64

        payload = json.loads(base64.b64decode(b64).decode("utf-8"))
    except Exception as exc:
        logger.warning("Could not parse SQI-dict upload %s: %s", filename, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("SQI-dict upload is not a JSON object")
        return None
    return payload


# ---------------------------------------------------------------------------
# Shortcut: drop a pre-computed SQI table → write straight to `dataframe`.
# Skips the recording-upload + extract_sqi step; the Inspect view treats
# this exactly like the output of a full Compute run.
# ---------------------------------------------------------------------------


@callback(
    Output("dataframe", "data", allow_duplicate=True),
    Output("compute-sqi-table-status", "children"),
    Output("raw-waveform", "data", allow_duplicate=True),
    Output("segment-milestones", "data", allow_duplicate=True),
    Input("compute-sqi-table-upload", "contents"),
    State("compute-sqi-table-upload", "filename"),
    prevent_initial_call=True,
)
def _load_pre_computed_sqi_table(contents, filename):
    """Parse a CSV or JSON upload as a pre-computed SQI table.

    Writes the result to the canonical ``dataframe`` store and clears
    the ``raw-waveform`` / ``segment-milestones`` stores (we have no
    waveform context for an externally-produced SQI table; the Inspect
    view's per-segment plot gracefully degrades to a placeholder).
    """
    if not contents or not filename:
        raise PreventUpdate
    try:
        df = _parse_sqi_table_upload(contents, filename)
    except _SqiTableUploadError as err:
        msg = html.Span(f"✗ {err}", style={"color": "#b30000"})
        return dash.no_update, msg, dash.no_update, dash.no_update
    if df is None or df.empty:
        return (
            dash.no_update,
            html.Span("✗ Upload contained no rows.", style={"color": "#b30000"}),
            dash.no_update,
            dash.no_update,
        )
    msg = html.Span(
        f"✓ Loaded {len(df)} segments × {df.shape[1]} columns. "
        "Open the Inspect tab to browse.",
        style={"color": "#0f5132"},
    )
    return df.to_dict(), msg, None, None


class _SqiTableUploadError(ValueError):
    """Internal: the SQI-table upload could not be parsed."""


def _parse_sqi_table_upload(contents: str, filename: str) -> pd.DataFrame:
    """Decode the Dash upload payload as a DataFrame.

    Accepts ``.csv`` (header row) and ``.json``.  Raises
    :class:`_SqiTableUploadError` with a user-readable message on
    failure; the caller turns this into a UI banner.
    """
    import base64 as _base64
    import io as _io

    try:
        _, b64 = contents.split(",", 1)
        raw = _base64.b64decode(b64)
    except Exception as exc:
        raise _SqiTableUploadError("Upload payload is malformed.") from exc

    lower = filename.lower()
    try:
        if lower.endswith(".csv") or lower.endswith(".txt"):
            return pd.read_csv(_io.BytesIO(raw))
        if lower.endswith(".json"):
            payload = json.loads(raw.decode("utf-8"))
            if isinstance(payload, dict):
                return pd.DataFrame(payload)
            if isinstance(payload, list):
                return pd.DataFrame(payload)
            raise _SqiTableUploadError(
                "JSON must be either a dict-of-columns or a list-of-rows."
            )
    except _SqiTableUploadError:
        raise
    except Exception as exc:
        raise _SqiTableUploadError(f"Could not parse {filename}: {exc}") from exc
    raise _SqiTableUploadError(
        f"Unsupported file type {filename!r}; use .csv or .json."
    )


# ---------------------------------------------------------------------------
# Callback — heavy lifting (background when diskcache is available)
# ---------------------------------------------------------------------------


_background = background_callback_manager is not None


@callback(
    output=[
        Output("dataframe", "data", allow_duplicate=True),
        Output("raw-waveform", "data", allow_duplicate=True),
        Output("segment-milestones", "data", allow_duplicate=True),
        Output("compute-status", "children"),
        Output("compute-progress", "value"),
        Output("compute-progress", "style"),
        Output("compute-result-preview", "children"),
    ],
    inputs=[Input("compute-run-btn", "n_clicks")],
    state=[
        State("compute-staged-upload", "data"),
        State("compute-wave-type", "value"),
        State("compute-sampling-rate", "value"),
        State("compute-duration", "value"),
        State("compute-overlap", "value"),
        State("compute-sqi-dict-source", "value"),
        State("compute-sqi-dict-staged", "data"),
        State("compute-n-jobs", "value"),
    ],
    background=_background,
    progress=[
        Output("compute-status", "children"),
        Output("compute-progress", "value"),
        Output("compute-progress", "style"),
    ] if _background else None,
    running=[
        (Output("compute-run-btn", "disabled"), True, False),
        (Output("compute-cancel-btn", "disabled"), False, True),
    ] if _background else None,
    cancel=[Input("compute-cancel-btn", "n_clicks")] if _background else None,
    prevent_initial_call=True,
    manager=background_callback_manager,
)
def _run_compute(
    set_progress,
    n_clicks,
    staged_upload,
    wave_type,
    sampling_rate,
    duration,
    overlap,
    sqi_dict_source,
    sqi_dict_payload,
    n_jobs,
):
    """Background callback: load → segment → extract_sqi → preview + store."""
    # Dash's "running" tuples reset disabled state, but the foreground branch
    # (no background manager) needs a guard.
    if not n_clicks or not staged_upload:
        raise PreventUpdate

    progress_visible = {"marginTop": "12px", "display": "block"}

    def _fail(msg: str):
        """Return tuple for an error path: keep stores untouched, surface msg."""
        return (
            dash.no_update,    # dataframe
            dash.no_update,    # raw-waveform
            dash.no_update,    # segment-milestones
            msg,
            0,
            progress_visible,
            html.Em(msg),
        )

    if _background:
        set_progress(("Loading recording…", 5, progress_visible))

    try:
        lw = load_from_upload(
            contents=staged_upload["contents"],
            filename=staged_upload["filename"],
            wave_type=wave_type,
            sampling_rate=sampling_rate,
            signal_column=staged_upload.get("signal_column"),
        )
    except WaveformLoaderError as err:
        msg = f"✗ Upload failed: {err}"
        logger.warning(msg)
        return _fail(msg)

    if _background:
        set_progress(("Segmenting…", 20, progress_visible))

    seg_df = _segments_dataframe(lw.signal, lw.sampling_rate)
    overlap = float(overlap or 0)
    duration = float(duration or 30)

    try:
        segments, milestones = split_segment(
            seg_df,
            sampling_rate=lw.sampling_rate,
            split_type=0,
            duration=duration,
            overlapping=overlap,
            wave_type=lw.wave_type,
        )
    except Exception as exc:
        logger.exception("split_segment failed")
        return _fail(f"✗ Segmentation failed: {exc}")

    if not segments:
        return _fail("✗ No segments produced. Reduce overlap or extend the recording.")

    if _background:
        set_progress(
            (f"Computing SQIs over {len(segments)} segments…", 50, progress_visible)
        )

    sqi_path, tmp_dir = _resolve_sqi_dict_path(sqi_dict_source, sqi_dict_payload)
    try:
        sqi_df = extract_sqi(
            segments,
            milestones,
            sqi_path,
            wave_type=lw.wave_type,
            n_jobs=int(n_jobs or 1),
        )
    except Exception as exc:
        logger.exception("extract_sqi failed")
        return _fail(f"✗ SQI computation failed: {exc}")
    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    if _background:
        set_progress(("Done", 100, progress_visible))

    summary = (
        f"✓ Computed {sqi_df.shape[1]} SQI columns over {len(segments)} segments."
    )
    # Persist the SQI DataFrame in the shared 'dataframe' store as a dict so
    # the existing dashboards consume it unchanged.  Also stash the raw
    # waveform and segment milestones in the session-only stores so the
    # Inspect view can show waveform context per segment.
    waveform_payload = lw.to_store()
    milestones_payload = milestones.to_dict("list")
    return (
        sqi_df.to_dict(),
        waveform_payload,
        milestones_payload,
        summary,
        100,
        progress_visible,
        _df_preview(sqi_df),
    )
