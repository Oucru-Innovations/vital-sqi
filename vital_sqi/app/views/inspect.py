"""Inspect view — visualise SQIs and waveforms segment by segment.

Phase 2 of the app enhancement plan.  Layout::

    +-----------------------------------------------------------+
    | Recording summary: <filename>  fs=<hz>  duration=<s>      |
    |                                                           |
    | Timeline strip [██ ██ ░░ ██ ██ ██ ░░ ░░ ██ ██ ██ ██ ...]  |
    |                                                           |
    | Segment dropdown: [ 023 | 14:02 – 14:32 | reject ▾ ]      |
    |                                                           |
    | +-------------- Waveform ----------+ +--- SQI panel ----+ |
    | |  plotly trace of raw samples     | |  kurtosis  3.4   | |
    | |  with segment-boundary markers   | |  perfusion 12.1  | |
    | |                                  | |  ...             | |
    | +----------------------------------+ +------------------+ |
    |                                                           |
    | Rule trace                                                |
    |   1. kurtosis_sqi   3.4    accept                         |
    |   2. perfusion_sqi  12.1   accept                         |
    |   3. zero_cross     0.84   REJECT ← decisive              |
    +-----------------------------------------------------------+

Classification is performed on-demand whenever any control changes,
so the timeline always reflects the active threshold mode + rule
selection.  No data is mutated; the shared SQI table in
``dcc.Store(id="dataframe")`` is left untouched.

Phase 5 dropped the user-uploaded ``rule-set-store``: the rule dict
now always comes from the bundled ``vital_sqi/resource/rule_dict.json``.
To change the defaults globally, run the Calibrate view and click
*Save as default*.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from vital_sqi.app.app import app  # noqa: F401  - registers callbacks below
from vital_sqi.app.util.waveform_loader import LoadedWaveform
from vital_sqi.common.utils import create_rule_def, sanitize_sqi
from vital_sqi.rule import Rule, classify_segments_robust
from vital_sqi.rule.auto_threshold import (
    quantile_band,
    strictest_columns,
    tuned_bands,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ACCEPT_COLOR = "#198754"   # bootstrap "success" green
_REJECT_COLOR = "#dc3545"   # bootstrap "danger" red
_UNKNOWN_COLOR = "#adb5bd"  # bootstrap "secondary" grey

_BUNDLED_RULE_DICT_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resource", "rule_dict.json"
    )
)

#: A small, validated set of SQIs that are pre-checked in the Inspect view's
#: rule picker.  These are the classical SQIs with well-known accept ranges
#: in the literature; the rest of the catalogue is more diagnostic and
#: tends to over-reject when auto-thresholded on a single recording.
#:
#: Order matters: cheap, discriminative SQIs go first so RuleSet.execute's
#: early-exit short-circuit triggers fastest on bad segments.
DEFAULT_RULE_COLUMNS = (
    "kurtosis_sqi",
    "perfusion_sqi",
    "correlogram_sqi",
    "msq_sqi",
    "dtw_sqi",
)



# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _summary_panel() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Recording", className="card-title"),
                html.Div(id="inspect-summary", className="small text-muted"),
            ]
        ),
        className="mb-3",
    )


def _timeline_panel() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Timeline", className="card-title"),
                html.P(
                    "Each cell is one segment. Click to inspect.",
                    className="small text-muted mb-2",
                ),
                dcc.Graph(
                    id="inspect-timeline",
                    config={
                        "displayModeBar": False,
                        "staticPlot": False,
                    },
                    style={"height": "120px"},
                ),
            ]
        ),
        className="mb-3",
    )


def _rules_panel() -> dbc.Card:
    """Card with rule selection, threshold mode, and an auto-tune control.

    Three modes (mirroring ``classify_segments``):

    * **Quantile** — slider picks the symmetric percentile trim
      (p5/p95 is the default; loosen to p1/p99 for a more permissive band).
    * **Auto-tune** — slider picks the joint accept target; per-rule
      quantile is computed so the product of independent accept rates
      hits the target.
    * **Manual** — use the bounds shipped in ``rule_dict.json`` verbatim,
      no adaptation.

    A second row offers a "Drop strictest rule" button that removes
    whichever rule is rejecting far more segments than its peers.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H5("Rules", className="card-title d-inline"),
                        html.Span(id="inspect-rules-summary", className="ms-2 small text-muted"),
                        html.Span(id="inspect-regime-banner", className="ms-2"),
                    ]
                ),

                # ── Mode selector ──────────────────────────────────────
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Threshold mode", className="small"),
                                dcc.RadioItems(
                                    id="inspect-mode",
                                    options=[
                                        {"label": " Quantile", "value": "quantile"},
                                        {"label": " Auto-tune", "value": "tune"},
                                        {"label": " Manual", "value": "manual"},
                                        {"label": " Robust", "value": "robust"},
                                    ],
                                    value="quantile",
                                    inline=True,
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                # Slider for quantile mode (p5/p95 by default).
                                html.Div(
                                    [
                                        dbc.Label(
                                            "Accept band (per-rule percentile)",
                                            className="small",
                                        ),
                                        dcc.RangeSlider(
                                            id="inspect-quantile-slider",
                                            min=0.0, max=0.5, step=0.005,
                                            value=[0.05, 0.05],
                                            marks={
                                                0.0: "p0",
                                                0.01: "p1",
                                                0.025: "p2.5",
                                                0.05: "p5",
                                                0.10: "p10",
                                                0.25: "p25",
                                            },
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                    ],
                                    id="inspect-quantile-slider-row",
                                ),
                            ],
                            md=8,
                        ),
                    ],
                    className="mb-2",
                ),

                # ── Auto-tune slider (separate row so it can be hidden) ─
                html.Div(
                    [
                        dbc.Label(
                            "Target joint accept rate",
                            className="small",
                        ),
                        dcc.Slider(
                            id="inspect-tune-slider",
                            min=0.5, max=0.99, step=0.01,
                            value=0.85,
                            marks={0.5: "50%", 0.7: "70%", 0.85: "85%", 0.95: "95%"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    id="inspect-tune-slider-row",
                    style={"display": "none"},
                    className="mb-2",
                ),

                html.Hr(),
                html.P(
                    "Toggle which SQIs participate in the accept/reject decision. "
                    "SQIs whose distribution is constant across this recording are "
                    "marked unusable.",
                    className="small text-muted mb-2",
                ),
                dbc.Checklist(
                    id="inspect-rules-checklist",
                    options=[],
                    value=[],
                    inline=True,
                    switch=False,
                    style={"maxHeight": "180px", "overflowY": "auto"},
                ),
                html.Div(id="inspect-rules-skipped", className="small text-muted mt-2"),

                # ── Drop strictest rule ────────────────────────────────
                html.Div(
                    [
                        dbc.Button(
                            "Drop strictest rule",
                            id="inspect-drop-strictest-btn",
                            color="warning",
                            outline=True,
                            size="sm",
                            n_clicks=0,
                            disabled=True,
                            className="mt-2",
                        ),
                        html.Span(
                            id="inspect-strictest-hint",
                            className="ms-2 small text-muted",
                        ),
                    ]
                ),
            ]
        ),
        className="mb-3",
    )


def _detail_panel() -> dbc.Card:
    waveform_col = dbc.Col(
        dcc.Graph(
            id="inspect-waveform",
            config={"displayModeBar": True},
            style={"height": "320px"},
        ),
        md=8,
    )
    sqi_col = dbc.Col(
        [
            html.H6("SQI values"),
            html.Div(id="inspect-sqi-table"),
        ],
        md=4,
    )
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H5("Segment", className="card-title d-inline"),
                        html.Span(id="inspect-segment-label", className="ms-2 small text-muted"),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Pick segment"),
                                dcc.Dropdown(
                                    id="inspect-segment-picker",
                                    placeholder="Loading…",
                                    clearable=False,
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Filter"),
                                dcc.RadioItems(
                                    id="inspect-filter",
                                    options=[
                                        {"label": " all", "value": "all"},
                                        {"label": " accept", "value": "accept"},
                                        {"label": " reject", "value": "reject"},
                                    ],
                                    value="all",
                                    inline=True,
                                ),
                            ],
                            md=6,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row([waveform_col, sqi_col]),
                html.Hr(),
                html.H6("Rule trace"),
                html.Div(id="inspect-rule-trace"),
            ]
        )
    )


layout = html.Div(
    [
        html.H3("Inspect"),
        html.P(
            "Click a segment in the timeline (or pick one in the dropdown) to "
            "see its waveform, SQI values, and the rule trace that decided "
            "whether it was accepted.",
            className="text-muted",
        ),
        html.Hr(),
        _summary_panel(),
        _rules_panel(),
        _timeline_panel(),
        _detail_panel(),
        # ``inspect-decisions`` lives at the app root (Phase 5) so the
        # Export view can read it.
    ]
)


# ---------------------------------------------------------------------------
# Pure helpers — exercised by unit tests
# ---------------------------------------------------------------------------


def _sqi_df_from_store(payload) -> Optional[pd.DataFrame]:
    """Reconstruct the SQI DataFrame from the shared ``dataframe`` store."""
    if not payload:
        return None
    try:
        df = pd.DataFrame(payload)
    except Exception as exc:
        logger.warning("Could not rebuild SQI DataFrame from store: %s", exc)
        return None
    if df.empty:
        return None
    return df


def _load_rule_dict() -> Dict[str, dict]:
    """Load the bundled ``rule_dict.json``.

    Phase 5 removed the per-session user-uploaded rule dict; custom
    bounds now live in ``vital_sqi/resource/rule_dict.json`` and are
    written there by the Calibrate view's *Save as default* button.
    Returns an empty dict if the resource is missing (the Inspect view
    will then surface "no rules applicable" for every segment).
    """
    try:
        with open(_BUNDLED_RULE_DICT_PATH, "r") as fh:
            return json.load(fh)
    except FileNotFoundError:
        logger.error("Bundled rule_dict.json not found at %s", _BUNDLED_RULE_DICT_PATH)
        return {}


def _build_rules(
    sqi_df: pd.DataFrame,
    rule_dict: Dict[str, dict],
    selected_columns: Optional[List[str]] = None,
    *,
    mode: str = "quantile",
    quantile_lo: float = 0.05,
    quantile_hi: float = 0.95,
    target_accept_rate: float = 0.85,
) -> List[Rule]:
    """Build :class:`Rule` objects from the current SQI distribution.

    Threshold-derivation strategy is governed by ``mode``:

    * ``"quantile"`` — symmetric trim at ``(quantile_lo, quantile_hi)``.
    * ``"tune"`` — each rule's quantile is chosen so the *joint* accept
      rate (under the independence approximation) hits
      ``target_accept_rate``.

    The selection / degenerate-band filtering logic is independent of the
    mode; both branches delegate to :mod:`vital_sqi.rule.auto_threshold`.

    Parameters
    ----------
    sqi_df
        The current SQI table (one row per segment).
    rule_dict
        Loaded rule definitions; only SQIs that appear here are considered.
    selected_columns
        Optional whitelist.  When non-empty only the named SQIs participate.
    mode
        ``"quantile"`` (default) or ``"tune"``.  Manual mode is handled
        in the calling code by short-circuiting before this is reached.
    quantile_lo, quantile_hi
        Used when ``mode == "quantile"``.
    target_accept_rate
        Used when ``mode == "tune"``.
    """
    whitelist = set(selected_columns) if selected_columns else None
    columns = [
        c for c in sqi_df.columns
        if c in rule_dict and (whitelist is None or c in whitelist)
    ]

    rules: List[Rule] = []
    if not columns:
        return rules

    if mode == "tune":
        col_values = {
            c: sanitize_sqi(sqi_df[c].values) for c in columns
        }
        bands = tuned_bands(col_values, target_accept_rate=target_accept_rate)
    else:
        bands = []
        for column in columns:
            band = quantile_band(
                column, sanitize_sqi(sqi_df[column].values),
                lower_pct=quantile_lo, upper_pct=quantile_hi,
            )
            if band is not None:
                bands.append(band)

    for band in bands:
        rule_def = create_rule_def(
            band.column, lower_bound=band.lower, upper_bound=band.upper,
        )
        try:
            rule = Rule(band.column)
            rule.load_def(rule_def)
            rules.append(rule)
        except Exception as exc:
            logger.warning("Could not build Rule for %s: %s", band.column, exc)
    return rules


def _candidate_rule_columns(
    sqi_df: pd.DataFrame, rule_dict: Dict[str, dict]
) -> List[Tuple[str, bool, str]]:
    """List columns that *could* be used in classification, with status.

    "Usable" is determined by whether the column's p5/p95 band is
    non-degenerate — that's the loosest test we apply.  A column that
    fails this test won't survive any tighter trim either, so the
    decision is mode-independent.
    """
    out: List[Tuple[str, bool, str]] = []
    for column in sqi_df.columns:
        if column not in rule_dict:
            continue
        values = sanitize_sqi(sqi_df[column].values)
        band = quantile_band(column, values, lower_pct=0.05, upper_pct=0.95)
        if band is None:
            valid = values[np.isfinite(values)] if values is not None else np.array([])
            if valid.size < 2:
                out.append((column, False, "fewer than 2 finite samples"))
            else:
                lo = float(np.quantile(valid, 0.05))
                hi = float(np.quantile(valid, 0.95))
                out.append(
                    (column, False, f"degenerate band p5={lo:.4g}, p95={hi:.4g}")
                )
            continue
        out.append((column, True, ""))
    return out


def _default_rules_for(usable_columns: List[str]) -> List[str]:
    """Intersection of :data:`DEFAULT_RULE_COLUMNS` and usable columns."""
    return [c for c in DEFAULT_RULE_COLUMNS if c in usable_columns]


def _classify(sqi_df: pd.DataFrame, rules: List[Rule]) -> List[dict]:
    """Run each rule on every row.  Returns ``[{decision, trace: [...]}, ...]``.

    A row's overall decision is ``reject`` if any rule rejects, ``accept``
    otherwise.  ``trace`` records the per-rule outcome and the value seen so
    the UI can highlight which rule was decisive.
    """
    decisions: List[dict] = []
    for _, row in sqi_df.iterrows():
        trace = []
        overall = "accept"
        for rule in rules:
            value = row.get(rule.name)
            try:
                outcome = rule.apply_rule(value)
            except Exception:
                outcome = "reject"
            trace.append({"name": rule.name, "value": value, "outcome": outcome})
            if outcome != "accept" and overall == "accept":
                overall = "reject"
        decisions.append({"decision": overall, "trace": trace})
    return decisions


def _segment_indices_for_filter(
    decisions: List[dict], filter_mode: str
) -> List[int]:
    if filter_mode == "all":
        return list(range(len(decisions)))
    return [i for i, d in enumerate(decisions) if d["decision"] == filter_mode]


def _make_timeline_figure(decisions: List[dict]) -> go.Figure:
    n = len(decisions)
    colors = [
        _ACCEPT_COLOR if d["decision"] == "accept" else _REJECT_COLOR
        for d in decisions
    ]
    fig = go.Figure(
        go.Bar(
            x=list(range(n)),
            y=[1] * n,
            marker={"color": colors},
            hovertemplate=(
                "<b>Segment %{x}</b><br>%{customdata}<extra></extra>"
            ),
            customdata=[d["decision"] for d in decisions],
            width=1.0,
        )
    )
    fig.update_layout(
        showlegend=False,
        margin={"l": 0, "r": 0, "t": 4, "b": 20},
        bargap=0,
        plot_bgcolor="#ffffff",
        xaxis={
            "title": "Segment index",
            "showgrid": False,
            "zeroline": False,
            "fixedrange": True,
        },
        yaxis={
            "showgrid": False,
            "showticklabels": False,
            "zeroline": False,
            "range": [0, 1],
            "fixedrange": True,
        },
    )
    return fig


def _make_waveform_figure(
    raw_signal: np.ndarray,
    sampling_rate: float,
    start_idx: int,
    end_idx: int,
    label: str,
) -> go.Figure:
    """Plot the raw samples for [start_idx, end_idx) as a Plotly line trace."""
    start_idx = max(0, int(start_idx))
    end_idx = min(len(raw_signal), int(end_idx))
    if end_idx <= start_idx:
        return go.Figure(layout={"title": "Empty segment"})
    samples = raw_signal[start_idx:end_idx]
    t_offset = start_idx / sampling_rate
    times = np.arange(samples.size) / sampling_rate + t_offset
    fig = go.Figure(
        go.Scattergl(
            x=times,
            y=samples,
            mode="lines",
            line={"width": 1},
            hovertemplate="t=%{x:.3f}s<br>y=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=label,
        margin={"l": 40, "r": 10, "t": 30, "b": 40},
        xaxis={"title": "Time (s)"},
        yaxis={"title": "Amplitude"},
        plot_bgcolor="#ffffff",
    )
    return fig


def _format_segment_label(
    idx: int, start_idx: int, end_idx: int, sampling_rate: float, decision: str
) -> str:
    duration = (end_idx - start_idx) / sampling_rate if sampling_rate else 0
    start_s = start_idx / sampling_rate if sampling_rate else 0
    return f"#{idx:04d}  •  t={start_s:.1f}s  •  Δ{duration:.1f}s  •  {decision}"


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("inspect-summary", "children"),
    Input("raw-waveform", "data"),
    Input("dataframe", "data"),
    prevent_initial_call=True,
)
def _render_summary(waveform_payload, sqi_payload):
    sqi_df = _sqi_df_from_store(sqi_payload)
    if not waveform_payload and sqi_df is None:
        return html.Em(
            "No data yet — load an SQI table via Compute or upload one on the Home page."
        )
    lines = []
    if waveform_payload:
        try:
            lw = LoadedWaveform.from_store(waveform_payload)
            duration = lw.signal.size / lw.sampling_rate if lw.sampling_rate else 0
            lines.append(
                f"{lw.source_name} — {lw.wave_type} @ {lw.sampling_rate:.2f} Hz, "
                f"{lw.signal.size} samples ({duration:.1f} s)"
            )
        except Exception:
            lines.append("Raw waveform present but could not be summarised.")
    if sqi_df is not None:
        lines.append(f"{len(sqi_df)} segments × {sqi_df.shape[1]} SQI columns")
    return [html.Div(line) for line in lines]


@callback(
    Output("inspect-rules-checklist", "options"),
    Output("inspect-rules-checklist", "value"),
    Output("inspect-rules-skipped", "children"),
    Output("inspect-rules-summary", "children"),
    Input("dataframe", "data"),
    State("inspect-rules-checklist", "value"),
    prevent_initial_call=True,
)
def _render_rule_picker(sqi_payload, current_value):
    """Populate the rule checklist when the SQI table or rule dict changes.

    The checklist offers every column that:

    * has a definition in the active rule dict, AND
    * is in the current SQI table.

    Columns whose auto-mode p5/p95 band is degenerate (e.g. constant
    ``zero_crossings_rate_sqi``) are surfaced as disabled options with an
    explanation, so the user can see *why* they aren't usable rather than
    silently dropping them.
    """
    sqi_df = _sqi_df_from_store(sqi_payload)
    if sqi_df is None:
        return [], [], "", html.Em("Load an SQI table to enable rule selection.")

    rule_dict = _load_rule_dict()
    candidates = _candidate_rule_columns(sqi_df, rule_dict)
    options = [
        {
            "label": (
                f" {name}" if usable else f" {name}  ✗ {reason}"
            ),
            "value": name,
            "disabled": not usable,
        }
        for (name, usable, reason) in candidates
    ]
    usable_names = [name for (name, usable, _) in candidates if usable]

    # Preserve the user's existing selection where still valid; otherwise
    # fall back to the default subset intersected with what's usable.
    if current_value:
        retained = [c for c in current_value if c in usable_names]
        value = retained or _default_rules_for(usable_names)
    else:
        value = _default_rules_for(usable_names)

    skipped = [(n, why) for (n, usable, why) in candidates if not usable]
    if skipped:
        skipped_node = html.Span(
            f"Auto-skipped: {', '.join(n for (n, _) in skipped)}"
        )
    else:
        skipped_node = ""

    summary = (
        f"{len(usable_names)} usable rule"
        + ("s" if len(usable_names) != 1 else "")
        + f", {len(value)} active"
    )
    return options, value, skipped_node, summary


@callback(
    Output("inspect-quantile-slider-row", "style"),
    Output("inspect-tune-slider-row", "style"),
    Input("inspect-mode", "value"),
    prevent_initial_call=True,
)
def _toggle_mode_controls(mode):
    """Show only the slider relevant to the active threshold mode.

    Robust mode hides both sliders — it ignores the rule dict entirely
    and derives accept/reject from the consensus score directly.
    """
    quantile_visible = {"display": "block"} if mode == "quantile" else {"display": "none"}
    tune_visible = {"display": "block"} if mode == "tune" else {"display": "none"}
    return quantile_visible, tune_visible


@callback(
    Output("inspect-decisions", "data"),
    Input("dataframe", "data"),
    Input("inspect-rules-checklist", "value"),
    Input("inspect-mode", "value"),
    Input("inspect-quantile-slider", "value"),
    Input("inspect-tune-slider", "value"),
    prevent_initial_call=True,
)
def _classify_on_demand(
    sqi_payload, selected_columns,
    mode, quantile_range, tune_target,
):
    """Recompute accept/reject decisions whenever any rule control changes.

    The threshold strategy is governed by *mode*; the relevant slider
    value is read directly from the layout's RangeSlider / Slider state
    so the panel feels live without round-tripping through ``dcc.Store``.
    """
    sqi_df = _sqi_df_from_store(sqi_payload)
    if sqi_df is None:
        return None

    rule_dict = _load_rule_dict()

    # Normalise slider values: the quantile RangeSlider holds (low, high)
    # in absolute percentile terms (e.g. (0.05, 0.05) means trim 5% on
    # each side — i.e. p5/p95).  We treat the higher of the two as the
    # symmetric trim to keep the UX simple.
    if quantile_range and len(quantile_range) == 2:
        # The slider's two handles are intended to lock together; pick
        # whichever the user moved last (max) so the band stays sane.
        q_trim = max(float(quantile_range[0]), float(quantile_range[1]))
    else:
        q_trim = 0.05
    q_trim = float(np.clip(q_trim, 0.0, 0.49))
    quantile_lo = q_trim
    quantile_hi = 1.0 - q_trim
    tune_target = float(np.clip(tune_target or 0.85, 0.5, 0.99))

    if mode == "robust":
        # Robust mode bypasses the rule dict.  It uses every SQI column
        # (modulo the checklist whitelist) to compute a rank-IQR
        # consensus score and then routes through one of three
        # regime-specific classifiers.
        whitelist = selected_columns or None
        if whitelist:
            cols = [c for c in whitelist if c in sqi_df.columns]
            if not cols:
                cols = list(sqi_df.columns)
        else:
            cols = list(sqi_df.columns)
        try:
            result = classify_segments_robust(sqi_df, sqi_names=cols)
        except Exception as exc:
            logger.exception("classify_segments_robust failed")
            return [{"decision": "unknown", "trace": []} for _ in range(len(sqi_df))]
        return _decisions_from_robust_result(result)

    if mode == "manual":
        rules = _build_rules_manual(rule_dict, selected_columns)
    else:
        rules = _build_rules(
            sqi_df, rule_dict, selected_columns=selected_columns,
            mode=mode,
            quantile_lo=quantile_lo, quantile_hi=quantile_hi,
            target_accept_rate=tune_target,
        )
    if not rules:
        return [{"decision": "unknown", "trace": []} for _ in range(len(sqi_df))]
    return _classify(sqi_df, rules)


@callback(
    Output("inspect-regime-banner", "children"),
    Input("inspect-mode", "value"),
    Input("inspect-decisions", "data"),
    prevent_initial_call=True,
)
def _render_regime_banner(mode, decisions):
    """Show a small "Regime: clean (score mean 0.83)" badge in robust mode."""
    if mode != "robust" or not decisions:
        return ""
    first = decisions[0]
    regime = first.get("regime", "unknown")
    info = first.get("regime_info", {}) or {}
    score_mean = info.get("score_mean")
    file_flagged = info.get("file_flagged", False)
    colour = {
        "clean":       "success",
        "bimodal":     "warning",
        "heavy_noise": "danger",
    }.get(regime, "secondary")

    badge_text = f"Regime: {regime}"
    if score_mean is not None:
        badge_text += f"  •  score mean {score_mean:.2f}"
    if file_flagged:
        badge_text += "  •  file flagged"

    import dash_bootstrap_components as dbc
    return dbc.Badge(badge_text, color=colour, className="ms-1")


def _decisions_from_robust_result(result) -> List[dict]:
    """Convert a :class:`RobustResult` into the per-segment dict format.

    The Inspect view's downstream callbacks expect each segment entry to
    have a ``decision`` and a ``trace`` (list of per-rule outcomes).  The
    robust classifier does not produce a per-rule trace — instead it
    yields a scalar consensus score and a recording-level regime.  We
    surface those as a single synthetic trace row labelled
    ``robust_score`` so the detail panel has something meaningful to
    show, and stash the regime / score for the summary callback to read.
    """
    decisions: List[dict] = []
    regime = result.regime
    regime_info = result.regime_info or {}
    for i, dec in enumerate(result.decisions):
        score = float(result.scores[i]) if i < len(result.scores) else float("nan")
        decisions.append({
            "decision": dec,
            "trace": [{
                "name": "robust_score",
                "value": score,
                "outcome": dec,
            }],
            "regime": regime,
            "regime_info": regime_info,
            "score": score,
        })
    return decisions


def _build_rules_manual(
    rule_dict: Dict[str, dict], selected_columns: Optional[List[str]]
) -> List[Rule]:
    """Build :class:`Rule` objects from rule_dict's stored bounds (no auto)."""
    whitelist = set(selected_columns) if selected_columns else None
    rules: List[Rule] = []
    for name, entry in rule_dict.items():
        if whitelist is not None and name not in whitelist:
            continue
        try:
            rule = Rule(name)
            rule.load_def({name: entry})
            rules.append(rule)
        except Exception as exc:
            logger.warning("Skipping manual rule %s: %s", name, exc)
    return rules


# ---------------------------------------------------------------------------
# Drop-strictest button
# ---------------------------------------------------------------------------


@callback(
    Output("inspect-drop-strictest-btn", "disabled"),
    Output("inspect-strictest-hint", "children"),
    Input("inspect-decisions", "data"),
    prevent_initial_call=True,
)
def _update_drop_strictest_state(decisions):
    """Enable the 'Drop strictest rule' button when one rule outliers up."""
    if not decisions or len(decisions) < 2:
        return True, ""
    per_rule = _per_rule_reject_counts(decisions)
    if not per_rule:
        return True, ""
    flagged = strictest_columns(per_rule)
    if not flagged:
        return True, "no clear outlier"
    worst = flagged[0]
    return False, f"would drop: {worst} (rejected {per_rule[worst]} segments)"


@callback(
    Output("inspect-rules-checklist", "value", allow_duplicate=True),
    Input("inspect-drop-strictest-btn", "n_clicks"),
    State("inspect-decisions", "data"),
    State("inspect-rules-checklist", "value"),
    prevent_initial_call=True,
)
def _drop_strictest_rule(n_clicks, decisions, current_value):
    if not n_clicks or not decisions or not current_value:
        raise PreventUpdate
    per_rule = _per_rule_reject_counts(decisions)
    flagged = strictest_columns(per_rule)
    if not flagged:
        raise PreventUpdate
    worst = flagged[0]
    new_value = [c for c in current_value if c != worst]
    return new_value


def _per_rule_reject_counts(decisions: List[dict]) -> "dict[str, int]":
    """Tally how many segments each rule rejected (over the whole timeline)."""
    counts: Dict[str, int] = {}
    for d in decisions:
        if d.get("decision") != "reject":
            continue
        for entry in d.get("trace", []):
            if entry.get("outcome") != "accept":
                counts[entry["name"]] = counts.get(entry["name"], 0) + 1
    return counts


@callback(
    Output("inspect-timeline", "figure"),
    Input("inspect-decisions", "data"),
    prevent_initial_call=True,
)
def _render_timeline(decisions):
    if not decisions:
        return go.Figure(
            layout={
                "annotations": [{
                    "text": "No segments to display",
                    "showarrow": False,
                    "xref": "paper", "yref": "paper",
                    "x": 0.5, "y": 0.5,
                }],
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "margin": {"l": 0, "r": 0, "t": 4, "b": 4},
                "height": 80,
            }
        )
    return _make_timeline_figure(decisions)


@callback(
    Output("inspect-segment-picker", "options"),
    Output("inspect-segment-picker", "value"),
    Input("inspect-decisions", "data"),
    Input("inspect-filter", "value"),
    State("inspect-segment-picker", "value"),
    prevent_initial_call=True,
)
def _render_segment_options(decisions, filter_mode, current_value):
    if not decisions:
        return [], None
    indices = _segment_indices_for_filter(decisions, filter_mode)
    options = [
        {
            "label": f"#{i:04d}  ({decisions[i]['decision']})",
            "value": i,
        }
        for i in indices
    ]
    # Preserve the current pick when it's still visible under the new filter,
    # otherwise fall back to the first option.
    if current_value in indices:
        value = current_value
    else:
        value = indices[0] if indices else None
    return options, value


@callback(
    Output("inspect-segment-picker", "value", allow_duplicate=True),
    Input("inspect-timeline", "clickData"),
    State("inspect-decisions", "data"),
    prevent_initial_call=True,
)
def _click_timeline_to_pick(click_data, decisions):
    if not click_data or not decisions:
        raise PreventUpdate
    try:
        idx = int(click_data["points"][0]["x"])
    except (KeyError, IndexError, TypeError, ValueError):
        raise PreventUpdate
    if 0 <= idx < len(decisions):
        return idx
    raise PreventUpdate


@callback(
    Output("inspect-waveform", "figure"),
    Output("inspect-sqi-table", "children"),
    Output("inspect-rule-trace", "children"),
    Output("inspect-segment-label", "children"),
    Input("inspect-segment-picker", "value"),
    State("raw-waveform", "data"),
    State("segment-milestones", "data"),
    State("dataframe", "data"),
    State("inspect-decisions", "data"),
    prevent_initial_call=True,
)
def _render_segment_detail(seg_idx, waveform_payload, milestones_payload, sqi_payload, decisions):
    sqi_df = _sqi_df_from_store(sqi_payload)
    if seg_idx is None or sqi_df is None:
        return go.Figure(), html.Em("No segment selected."), "", ""

    seg_idx = int(seg_idx)
    if seg_idx < 0 or seg_idx >= len(sqi_df):
        return go.Figure(), html.Em("Segment index out of range."), "", ""

    decision_entry = (decisions or [{}])[seg_idx] if decisions else {}
    decision = decision_entry.get("decision", "unknown")
    trace = decision_entry.get("trace", [])

    # Waveform: need raw signal + milestones.  Both may be absent if the
    # user came here via an "Upload SQI table" path; in that case render a
    # placeholder and skip the line plot.
    figure = go.Figure()
    label = f"Segment {seg_idx} — {decision}"
    if waveform_payload and milestones_payload:
        try:
            lw = LoadedWaveform.from_store(waveform_payload)
            start_idx = int(milestones_payload["start"][seg_idx])
            end_idx = int(milestones_payload["end"][seg_idx])
            figure = _make_waveform_figure(
                lw.signal, lw.sampling_rate, start_idx, end_idx,
                label=label,
            )
            label = _format_segment_label(
                seg_idx, start_idx, end_idx, lw.sampling_rate, decision
            )
        except Exception as exc:
            logger.warning("Could not draw waveform for segment %s: %s", seg_idx, exc)
            figure = go.Figure(layout={"title": "Waveform unavailable"})
    else:
        figure = go.Figure(
            layout={
                "annotations": [{
                    "text": "Raw waveform not available for this session.\n"
                            "Use the Compute view to pick up segment context.",
                    "showarrow": False,
                    "xref": "paper", "yref": "paper",
                    "x": 0.5, "y": 0.5,
                }],
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "height": 320,
            }
        )

    # SQI value table — single-segment summary
    sqi_row = sqi_df.iloc[seg_idx].to_dict()
    sqi_table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th("SQI"), html.Th("value", style={"textAlign": "right"})])),
            html.Tbody(
                [
                    html.Tr([
                        html.Td(name),
                        html.Td(
                            f"{value:.4f}" if isinstance(value, (int, float, np.floating)) and pd.notna(value) else str(value),
                            style={"textAlign": "right"},
                        ),
                    ])
                    for name, value in sqi_row.items()
                ]
            ),
        ],
        striped=True,
        bordered=False,
        hover=True,
        size="sm",
    )

    # Rule trace — flag the first reject so the UI can highlight it
    decisive_marked = False
    trace_rows = []
    for entry in trace:
        outcome = entry["outcome"]
        is_decisive = outcome != "accept" and not decisive_marked
        if is_decisive:
            decisive_marked = True
        cls = "table-danger" if is_decisive else (
            "" if outcome == "accept" else "text-muted"
        )
        trace_rows.append(
            html.Tr(
                [
                    html.Td(entry["name"]),
                    html.Td(
                        f"{entry['value']:.4f}"
                        if isinstance(entry["value"], (int, float, np.floating))
                        and pd.notna(entry["value"])
                        else str(entry["value"])
                    ),
                    html.Td(outcome.upper(), style={"fontWeight": "bold" if is_decisive else "normal"}),
                    html.Td("← decisive" if is_decisive else "", className="text-muted small"),
                ],
                className=cls,
            )
        )
    if not trace_rows:
        trace_node = html.Em(
            "No rules apply to this recording (rule_dict missing or no matching SQI columns)."
        )
    else:
        trace_node = dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("Rule"),
                        html.Th("Value"),
                        html.Th("Outcome"),
                        html.Th(""),
                    ])
                ),
                html.Tbody(trace_rows),
            ],
            striped=True,
            bordered=False,
            hover=True,
            size="sm",
        )

    return figure, sqi_table, trace_node, label
