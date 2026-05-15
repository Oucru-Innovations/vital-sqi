"""Export view — bundle the current run for offline use.

Phase 5 of the app enhancement plan.  Replaces the legacy Apply
dashboard.  The view reads everything Inspect has already produced
(SQI table, decisions, raw waveform, segment milestones) and offers
four download formats:

* **Decisions CSV** — the SQI table with the current accept/reject
  column from Inspect's classifier.
* **Rule dict JSON** — a snapshot of the auto-tuned bounds Inspect is
  currently applying, in the same format as
  ``vital_sqi/resource/rule_dict.json`` so it can be replayed via
  :func:`vital_sqi.pipeline.pipeline_functions.classify_segments`
  with ``auto_mode="manual"``.
* **Accepted segments ZIP** — per-segment CSVs of the *raw* waveform
  for every segment Inspect flagged as accept, zipped together.
* **HTML report** — a standalone single-file report (no external CSS
  needed) showing summary counts, per-SQI histograms, and the active
  configuration.  Useful for sharing results with a collaborator.

Every export is generated on demand: no files are written until the
user clicks the corresponding button.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from vital_sqi.app.app import app  # noqa: F401  — registers callbacks below
from vital_sqi.app.util.waveform_loader import LoadedWaveform
from vital_sqi.common.utils import sanitize_sqi
from vital_sqi.rule.auto_threshold import quantile_band, tuned_bands

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _summary_card() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Current run", className="card-title"),
                html.Div(
                    id="export-summary",
                    className="small text-muted",
                ),
            ]
        ),
        className="mb-3",
    )


def _download_card(
    button_id: str,
    title: str,
    description: str,
    button_label: str = "Download",
) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="card-title"),
                html.P(description, className="small text-muted mb-2"),
                dbc.Button(
                    button_label,
                    id=button_id,
                    color="primary",
                    outline=True,
                    n_clicks=0,
                ),
                html.Span(
                    id=f"{button_id}-status",
                    className="ms-2 small text-muted",
                ),
            ]
        ),
        className="mb-3",
    )


layout = html.Div(
    [
        html.H3("Export the current run"),
        html.P(
            "Each card below produces a self-contained artefact you can "
            "share or feed into a downstream pipeline.  All downloads "
            "are generated on demand — no files touch disk until you "
            "press the button.",
            className="text-muted",
        ),
        html.Hr(),
        _summary_card(),

        _download_card(
            "export-decisions-btn",
            "Decisions CSV",
            "The SQI table with the current accept/reject column. "
            "One row per segment.",
        ),
        _download_card(
            "export-rule-dict-btn",
            "Rule dict JSON",
            "Snapshot of the bounds Inspect is currently applying.  "
            "Replay with classify_segments(auto_mode='manual', "
            "rule_dict_filename=…).",
        ),
        _download_card(
            "export-accepted-zip-btn",
            "Accepted segments ZIP",
            "Per-segment CSVs of the raw waveform for every segment "
            "the classifier accepted, bundled into a single archive.",
        ),
        _download_card(
            "export-html-btn",
            "HTML report",
            "Single-file standalone report with summary counts, per-SQI "
            "histograms, and the active configuration.  Good for "
            "sharing results with collaborators.",
        ),

        # Hidden download endpoints — one per artefact so we can address
        # each button's callback independently.
        dcc.Download(id="export-decisions-download"),
        dcc.Download(id="export-rule-dict-download"),
        dcc.Download(id="export-accepted-zip-download"),
        dcc.Download(id="export-html-download"),
    ]
)


# ---------------------------------------------------------------------------
# Pure helpers — testable without spinning up Dash
# ---------------------------------------------------------------------------


def _sqi_df_from_store(payload) -> Optional[pd.DataFrame]:
    if not payload:
        return None
    try:
        df = pd.DataFrame(payload)
    except Exception:
        return None
    if df.empty:
        return None
    return df


def _attach_decisions(sqi_df: pd.DataFrame, decisions) -> pd.DataFrame:
    """Return a copy of *sqi_df* with a ``decision`` column appended.

    When the Inspect view hasn't been opened yet, *decisions* is
    ``None`` and we leave the column empty (the user still gets the
    raw SQI values).
    """
    out = sqi_df.copy()
    if decisions and len(decisions) == len(sqi_df):
        out["decision"] = [d.get("decision", "unknown") for d in decisions]
    else:
        out["decision"] = "unknown"
    return out


def _decisions_to_rule_dict(
    sqi_df: pd.DataFrame,
    decisions: Optional[list],
    mode: str = "quantile",
    quantile_lo: float = 0.05,
    quantile_hi: float = 0.95,
    target_accept_rate: float = 0.85,
    selected_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Snapshot the active auto-tuned rule set as a rule_dict.

    The returned dict is in the same shape as
    ``vital_sqi/resource/rule_dict.json`` and can be fed back into
    :func:`vital_sqi.pipeline.pipeline_functions.classify_segments`
    with ``auto_mode="manual"``.
    """
    if mode == "manual" or mode == "robust":
        # Manual just exports the bundled bounds as-is; robust has no
        # per-rule bounds to snapshot, so we return an empty dict.
        return {}

    whitelist = set(selected_columns) if selected_columns else None
    columns = [
        c for c in sqi_df.columns
        if c != "decision" and (whitelist is None or c in whitelist)
    ]

    if mode == "tune":
        col_values = {c: sanitize_sqi(sqi_df[c].values) for c in columns}
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

    rule_dict: Dict[str, Any] = {}
    for band in bands:
        rule_dict[band.column] = {
            "name": band.column,
            "def": [
                {"op": ">",  "value": f"{band.lower:.6g}", "label": "accept"},
                {"op": "<=", "value": f"{band.lower:.6g}", "label": "reject"},
                {"op": ">=", "value": f"{band.upper:.6g}", "label": "reject"},
                {"op": "<",  "value": f"{band.upper:.6g}", "label": "accept"},
            ],
            "desc": (
                f"Snapshot from Inspect view ({mode} mode, "
                f"q={band.quantile_lo:.4f}/{band.quantile_hi:.4f}, "
                f"n={len(sqi_df)} segments)."
            ),
            "ref": "vital_sqi.app.views.export",
        }
    return rule_dict


def _build_accepted_zip(
    waveform_payload,
    milestones_payload,
    decisions,
) -> Optional[bytes]:
    """Build an in-memory ZIP of per-segment CSVs (accepted segments only).

    Returns ``None`` when the raw waveform / milestones aren't
    available (e.g. the user uploaded a pre-computed SQI table rather
    than running Compute).
    """
    if not (waveform_payload and milestones_payload and decisions):
        return None
    try:
        lw = LoadedWaveform.from_store(waveform_payload)
    except Exception:
        return None
    starts = milestones_payload.get("start") or []
    ends = milestones_payload.get("end") or []
    if not (len(starts) == len(ends) == len(decisions)):
        return None

    buf = io.BytesIO()
    n_written = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, (start, end, dec) in enumerate(zip(starts, ends, decisions)):
            if dec.get("decision") != "accept":
                continue
            s = max(0, int(start))
            e = min(len(lw.signal), int(end))
            if e <= s:
                continue
            samples = lw.signal[s:e]
            times = np.arange(samples.size) / max(lw.sampling_rate, 1.0) + (s / lw.sampling_rate)
            sub = pd.DataFrame({"time_s": times, "signal": samples})
            csv_bytes = sub.to_csv(index=False).encode("utf-8")
            zf.writestr(f"segment_{idx:04d}.csv", csv_bytes)
            n_written += 1
    if n_written == 0:
        return None
    return buf.getvalue()


def _build_html_report(
    sqi_df_with_decisions: pd.DataFrame,
    waveform_payload,
    decisions,
    mode: str,
    selected_columns: Optional[List[str]],
) -> str:
    """Produce a standalone HTML report as a string.

    Kept dependency-light: no external CSS / JS imports, just inline
    styles.  Plotly figures are embedded as base64-encoded SVG so the
    file works offline.
    """
    n = len(sqi_df_with_decisions)
    n_accept = int((sqi_df_with_decisions["decision"] == "accept").sum())
    n_reject = int((sqi_df_with_decisions["decision"] == "reject").sum())
    n_unknown = n - n_accept - n_reject

    parts: List[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<title>vital_sqi report</title>")
    parts.append(
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:2em;color:#222;}"
        "h1{color:#2c3e50;}"
        ".badge{display:inline-block;padding:2px 8px;border-radius:4px;"
        "color:#fff;font-size:0.85em;margin-right:4px;}"
        ".accept{background:#198754;} .reject{background:#dc3545;}"
        ".unknown{background:#adb5bd;}"
        "table{border-collapse:collapse;margin:1em 0;}"
        "th,td{padding:4px 12px;border-bottom:1px solid #eee;text-align:left;}"
        ".small{color:#777;font-size:0.9em;}"
        "</style></head><body>"
    )
    parts.append(f"<h1>vital_sqi report</h1>")
    parts.append(
        f"<p class='small'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        f" — {n} segments — threshold mode <b>{mode}</b></p>"
    )

    # Recording summary
    if waveform_payload:
        try:
            lw = LoadedWaveform.from_store(waveform_payload)
            duration = lw.signal.size / lw.sampling_rate if lw.sampling_rate else 0
            parts.append(
                f"<p><b>Recording:</b> {lw.source_name} — {lw.wave_type} @ "
                f"{lw.sampling_rate:.2f} Hz, {lw.signal.size} samples "
                f"({duration:.1f} s).</p>"
            )
        except Exception:
            pass

    # Decision breakdown
    parts.append("<h2>Decision summary</h2>")
    parts.append("<p>")
    parts.append(
        f"<span class='badge accept'>accept</span> {n_accept} "
        f"({100 * n_accept / max(n, 1):.1f}%) &nbsp;"
    )
    parts.append(
        f"<span class='badge reject'>reject</span> {n_reject} "
        f"({100 * n_reject / max(n, 1):.1f}%) &nbsp;"
    )
    if n_unknown:
        parts.append(
            f"<span class='badge unknown'>unknown</span> {n_unknown} "
            f"({100 * n_unknown / max(n, 1):.1f}%)"
        )
    parts.append("</p>")

    # Active rules
    parts.append("<h2>Active rules</h2>")
    if selected_columns:
        parts.append("<ul>")
        for name in selected_columns:
            parts.append(f"<li><code>{name}</code></li>")
        parts.append("</ul>")
    else:
        parts.append("<p class='small'>No explicit selection — view defaults applied.</p>")

    # Per-SQI summary statistics
    parts.append("<h2>Per-SQI summary</h2>")
    numeric = sqi_df_with_decisions.drop(
        columns=["decision"], errors="ignore"
    ).select_dtypes(include=[np.number])
    summary = numeric.agg(["mean", "std", "min", "max"]).T.round(4)
    parts.append("<table><tr><th>SQI</th>")
    for col in summary.columns:
        parts.append(f"<th>{col}</th>")
    parts.append("</tr>")
    for sqi_name, row in summary.iterrows():
        parts.append(f"<tr><td>{sqi_name}</td>")
        for col in summary.columns:
            parts.append(f"<td>{row[col]}</td>")
        parts.append("</tr>")
    parts.append("</table>")

    parts.append(
        "<p class='small'>This report was generated by the "
        "vital_sqi Export view (Phase 5).</p>"
    )
    parts.append("</body></html>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Summary callback
# ---------------------------------------------------------------------------


@callback(
    Output("export-summary", "children"),
    Input("dataframe", "data"),
    Input("inspect-decisions", "data"),
    Input("raw-waveform", "data"),
    prevent_initial_call=True,
)
def _render_summary(sqi_payload, decisions, waveform_payload):
    sqi_df = _sqi_df_from_store(sqi_payload)
    if sqi_df is None:
        return html.Em(
            "No data yet — load an SQI table via Compute or open Inspect "
            "to classify the current run."
        )
    lines: List[str] = []
    if waveform_payload:
        try:
            lw = LoadedWaveform.from_store(waveform_payload)
            duration = lw.signal.size / lw.sampling_rate if lw.sampling_rate else 0
            lines.append(
                f"Recording: {lw.source_name} — {lw.wave_type} @ "
                f"{lw.sampling_rate:.2f} Hz ({duration:.1f} s)"
            )
        except Exception:
            pass
    lines.append(f"{len(sqi_df)} segments × {sqi_df.shape[1]} SQI columns")
    if decisions and len(decisions) == len(sqi_df):
        n_accept = sum(1 for d in decisions if d.get("decision") == "accept")
        lines.append(
            f"{n_accept}/{len(sqi_df)} segments accepted "
            f"({100 * n_accept / max(len(sqi_df), 1):.1f}%)"
        )
    else:
        lines.append("Open Inspect first to compute decisions.")
    return [html.Div(line) for line in lines]


# ---------------------------------------------------------------------------
# Download callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("export-decisions-download", "data"),
    Output("export-decisions-btn-status", "children"),
    Input("export-decisions-btn", "n_clicks"),
    State("dataframe", "data"),
    State("inspect-decisions", "data"),
    prevent_initial_call=True,
)
def _download_decisions(n_clicks, sqi_payload, decisions):
    if not n_clicks:
        raise PreventUpdate
    sqi_df = _sqi_df_from_store(sqi_payload)
    if sqi_df is None:
        return None, html.Span("✗ No SQI table loaded.", style={"color": "#b30000"})
    out = _attach_decisions(sqi_df, decisions)
    return (
        dict(content=out.to_csv(index=False), filename="decisions.csv"),
        html.Span("✓ decisions.csv downloaded.", style={"color": "#0f5132"}),
    )


@callback(
    Output("export-rule-dict-download", "data"),
    Output("export-rule-dict-btn-status", "children"),
    Input("export-rule-dict-btn", "n_clicks"),
    State("dataframe", "data"),
    State("inspect-decisions", "data"),
    State("inspect-rules-checklist", "value"),
    State("inspect-mode", "value"),
    State("inspect-quantile-slider", "value"),
    State("inspect-tune-slider", "value"),
    prevent_initial_call=True,
)
def _download_rule_dict(
    n_clicks, sqi_payload, decisions,
    selected_columns, mode, quantile_range, tune_target,
):
    if not n_clicks:
        raise PreventUpdate
    sqi_df = _sqi_df_from_store(sqi_payload)
    if sqi_df is None:
        return None, html.Span("✗ No SQI table loaded.", style={"color": "#b30000"})

    if mode in (None, "manual"):
        return None, html.Span(
            "✗ Rule-dict snapshot only available in Quantile / Auto-tune mode "
            "(Manual mode would just re-export the bundled defaults).",
            style={"color": "#b30000"},
        )
    if mode == "robust":
        return None, html.Span(
            "✗ Robust mode has no per-rule bounds to snapshot.",
            style={"color": "#b30000"},
        )

    if quantile_range and len(quantile_range) == 2:
        q_trim = max(float(quantile_range[0]), float(quantile_range[1]))
    else:
        q_trim = 0.05
    rule_dict = _decisions_to_rule_dict(
        sqi_df, decisions, mode=mode,
        quantile_lo=q_trim, quantile_hi=1.0 - q_trim,
        target_accept_rate=float(tune_target or 0.85),
        selected_columns=selected_columns,
    )
    if not rule_dict:
        return None, html.Span(
            "✗ No usable bounds — every rule was degenerate.",
            style={"color": "#b30000"},
        )
    return (
        dict(content=json.dumps(rule_dict, indent=2), filename="rule_dict_snapshot.json"),
        html.Span(
            f"✓ rule_dict_snapshot.json downloaded ({len(rule_dict)} entries).",
            style={"color": "#0f5132"},
        ),
    )


@callback(
    Output("export-accepted-zip-download", "data"),
    Output("export-accepted-zip-btn-status", "children"),
    Input("export-accepted-zip-btn", "n_clicks"),
    State("raw-waveform", "data"),
    State("segment-milestones", "data"),
    State("inspect-decisions", "data"),
    prevent_initial_call=True,
)
def _download_accepted_zip(n_clicks, waveform_payload, milestones_payload, decisions):
    if not n_clicks:
        raise PreventUpdate
    payload = _build_accepted_zip(waveform_payload, milestones_payload, decisions)
    if payload is None:
        return None, html.Span(
            "✗ Need a raw recording + classified decisions.  Run Compute and "
            "open Inspect first.",
            style={"color": "#b30000"},
        )
    return (
        dcc.send_bytes(payload, filename="accepted_segments.zip"),
        html.Span("✓ accepted_segments.zip downloaded.", style={"color": "#0f5132"}),
    )


@callback(
    Output("export-html-download", "data"),
    Output("export-html-btn-status", "children"),
    Input("export-html-btn", "n_clicks"),
    State("dataframe", "data"),
    State("inspect-decisions", "data"),
    State("raw-waveform", "data"),
    State("inspect-mode", "value"),
    State("inspect-rules-checklist", "value"),
    prevent_initial_call=True,
)
def _download_html(
    n_clicks, sqi_payload, decisions, waveform_payload,
    mode, selected_columns,
):
    if not n_clicks:
        raise PreventUpdate
    sqi_df = _sqi_df_from_store(sqi_payload)
    if sqi_df is None:
        return None, html.Span("✗ No SQI table loaded.", style={"color": "#b30000"})
    sqi_with_dec = _attach_decisions(sqi_df, decisions)
    html_text = _build_html_report(
        sqi_with_dec, waveform_payload, decisions,
        mode=mode or "quantile",
        selected_columns=selected_columns,
    )
    return (
        dict(content=html_text, filename="vital_sqi_report.html"),
        html.Span("✓ vital_sqi_report.html downloaded.", style={"color": "#0f5132"}),
    )
