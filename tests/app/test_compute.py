"""Unit tests for the Compute view pure helpers and layout structure."""

import base64
import io
import json
import os

import numpy as np
import pandas as pd
import pytest
from dash import dcc, html

from vital_sqi.app.views import compute


def _find_components(component, kind):
    """Recursively collect components of ``kind`` from a Dash tree."""
    found = []
    if isinstance(component, kind):
        found.append(component)
    children = getattr(component, "children", None)
    if children is None:
        return found
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if child is not None:
            found.extend(_find_components(child, kind))
    return found


class TestLayout:
    def test_layout_is_html_div(self):
        assert isinstance(compute.layout, html.Div)

    def test_has_waveform_upload(self):
        uploads = _find_components(compute.layout, dcc.Upload)
        ids = {getattr(u, "id", None) for u in uploads}
        assert "compute-upload-waveform" in ids

    def test_has_sqi_dict_upload(self):
        uploads = _find_components(compute.layout, dcc.Upload)
        ids = {getattr(u, "id", None) for u in uploads}
        assert "compute-sqi-dict-upload" in ids

    def test_has_required_stores(self):
        stores = _find_components(compute.layout, dcc.Store)
        ids = {getattr(s, "id", None) for s in stores}
        assert "compute-staged-upload" in ids
        assert "compute-loaded-waveform" in ids
        assert "compute-sqi-dict-staged" in ids

    def test_signal_column_dropdown_present(self):
        drops = _find_components(compute.layout, dcc.Dropdown)
        ids = {getattr(d, "id", None) for d in drops}
        assert "compute-signal-column" in ids

    def test_run_button_is_disabled_initially(self):
        # Search the rendered layout for an element with id 'compute-run-btn'.
        import dash_bootstrap_components as dbc

        buttons = _find_components(compute.layout, dbc.Button)
        run_btn = next(
            (b for b in buttons if getattr(b, "id", None) == "compute-run-btn"),
            None,
        )
        assert run_btn is not None
        assert run_btn.disabled is True


class TestHelpers:
    def test_segments_dataframe_shape(self):
        signal = np.arange(100, dtype=float)
        df = compute._segments_dataframe(signal, sampling_rate=100)
        assert df.shape == (100, 2)
        assert "timestamps" in df.columns
        assert "signal" in df.columns

    def test_resolve_bundled_sqi_dict(self):
        path, tmp = compute._resolve_sqi_dict_path("bundled", None)
        assert os.path.exists(path)
        assert tmp is None

    def test_resolve_custom_sqi_dict(self, tmp_path):
        payload = {"kurt": {"sqi": "kurtosis_sqi", "args": {}}}
        path, tmp = compute._resolve_sqi_dict_path("custom", payload)
        try:
            assert tmp is not None
            assert os.path.isfile(path)
            with open(path) as fh:
                data = json.load(fh)
            assert data == payload
        finally:
            if tmp and os.path.isdir(tmp):
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)

    def test_resolve_custom_falls_back_when_payload_missing(self):
        path, tmp = compute._resolve_sqi_dict_path("custom", None)
        assert os.path.exists(path)
        assert tmp is None

    def test_df_preview_with_data(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = compute._df_preview(df)
        # Result is either a DataTable or our "no data" placeholder.
        from dash import dash_table
        assert isinstance(result, dash_table.DataTable)

    def test_df_preview_empty_returns_em_placeholder(self):
        result = compute._df_preview(pd.DataFrame())
        from dash import html as dh

        assert isinstance(result, dh.Em)
