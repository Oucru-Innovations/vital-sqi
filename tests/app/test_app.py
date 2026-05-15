"""
Unit tests for the Dash app — no browser required.

These tests verify the app object, layout structure, and callback
logic purely in Python without starting a server or launching Chrome.
"""
import base64
import io
import json
import pytest
import pandas as pd
from dash import html, dcc


# ---------------------------------------------------------------------------
# App import
# ---------------------------------------------------------------------------

def _get_app():
    """Import the index module (which sets app.layout) and return the app."""
    import vital_sqi.app.index  # noqa: F401  — side-effect: sets app.layout
    from vital_sqi.app.app import app
    return app


# ---------------------------------------------------------------------------
# App object sanity checks
# ---------------------------------------------------------------------------

class TestAppObject:
    def test_app_is_dash_instance(self):
        import dash
        app = _get_app()
        assert isinstance(app, dash.Dash)

    def test_app_has_layout(self):
        app = _get_app()
        assert app.layout is not None

    def test_layout_is_div(self):
        app = _get_app()
        assert isinstance(app.layout, html.Div)

    def test_suppress_callback_exceptions(self):
        app = _get_app()
        assert app.config.suppress_callback_exceptions is True


# ---------------------------------------------------------------------------
# Layout structure tests  (inspect component tree, no browser)
# ---------------------------------------------------------------------------

def _find_components(component, component_type):
    """Recursively collect all components of a given type in the layout tree."""
    found = []
    if isinstance(component, component_type):
        found.append(component)
    # Walk children regardless of whether the current node matches
    children = getattr(component, "children", None)
    if children is None:
        return found
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if child is None:
            continue
        found.extend(_find_components(child, component_type))
    return found


class TestLayoutStructure:
    @pytest.fixture(autouse=True)
    def app(self):
        self._app = _get_app()

    def test_layout_contains_url_location(self):
        locations = _find_components(self._app.layout, dcc.Location)
        assert len(locations) >= 1
        assert any(loc.id == "url" for loc in locations)

    def test_layout_contains_stores(self):
        stores = _find_components(self._app.layout, dcc.Store)
        store_ids = {s.id for s in stores}
        assert "dataframe" in store_ids
        assert "rule-set-store" in store_ids

    def test_layout_contains_page_content(self):
        divs = _find_components(self._app.layout, html.Div)
        assert any(getattr(d, "id", None) == "page-content" for d in divs)

    def test_sidebar_heading_text(self):
        h2s = _find_components(self._app.layout, html.H2)
        assert any(
            getattr(h, "children", None) == "Menu" for h in h2s
        )

    def test_home_content_has_upload_data(self):
        # upload-data lives in home_content, rendered by display_page("/")
        import vital_sqi.app.index as idx
        home = idx.display_page("/")
        uploads = _find_components(home, dcc.Upload)
        assert any(getattr(u, "id", None) == "upload-data" for u in uploads)

    def test_home_content_has_upload_rule(self):
        import vital_sqi.app.index as idx
        home = idx.display_page("/")
        uploads = _find_components(home, dcc.Upload)
        assert any(getattr(u, "id", None) == "upload-rule" for u in uploads)


# ---------------------------------------------------------------------------
# Callback logic tests  (call the Python functions directly)
# ---------------------------------------------------------------------------

class TestDisplayPageCallback:
    @pytest.fixture(autouse=True)
    def setup(self):
        import vital_sqi.app.index as idx
        self.display_page = idx.display_page
        from vital_sqi.app.views import dashboard1, dashboard2, dashboard3
        self.d1 = dashboard1
        self.d2 = dashboard2
        self.d3 = dashboard3

    def test_home_route_returns_home_content(self):
        result = self.display_page("/")
        assert isinstance(result, html.Div)

    def test_dashboard1_route(self):
        result = self.display_page("/views/dashboard1")
        assert result is self.d1.layout

    def test_dashboard2_route(self):
        result = self.display_page("/views/dashboard2")
        assert result is self.d2.layout

    def test_dashboard3_route(self):
        result = self.display_page("/views/dashboard3")
        assert result is self.d3.layout

    def test_unknown_route_falls_back_to_home(self):
        result = self.display_page("/nonexistent")
        assert isinstance(result, html.Div)


class TestUpdateOutputCallback:
    @pytest.fixture(autouse=True)
    def setup(self):
        import vital_sqi.app.index as idx
        self.update_output = idx.update_output

    def _encode_csv(self, df):
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        encoded = base64.b64encode(buf.getvalue().encode()).decode()
        return f"data:text/csv;base64,{encoded}"

    def test_no_content_returns_none_and_disabled(self):
        data, d1, d2, d3 = self.update_output(None, None, None, None)
        assert data is None
        assert d1 is True and d2 is True and d3 is True

    def test_state_data_preserved_when_no_new_upload(self):
        existing = {"col": {0: 1}}
        data, d1, d2, d3 = self.update_output(None, None, None, existing)
        assert data == existing
        assert d1 is False and d2 is False and d3 is False

    def test_valid_csv_upload_enables_links(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        content = self._encode_csv(df)
        data, d1, d2, d3 = self.update_output(content, "test.csv", None, None)
        assert isinstance(data, dict)
        assert d1 is False and d2 is False and d3 is False


class TestUploadRuleCallback:
    @pytest.fixture(autouse=True)
    def setup(self):
        import vital_sqi.app.index as idx
        self.upload_rule = idx.upload_rule

    def _encode_json(self, obj):
        encoded = base64.b64encode(json.dumps(obj).encode()).decode()
        return f"data:application/json;base64,{encoded}"

    def test_no_content_returns_none(self):
        result = self.upload_rule(None, None, None, None)
        assert result is None

    def test_state_data_returned_when_no_upload(self):
        existing = {"rule": "data"}
        result = self.upload_rule(None, None, None, existing)
        assert result == existing

    def test_json_upload_returns_parsed_data(self):
        obj = {"key": "value"}
        content = self._encode_json(obj)
        result = self.upload_rule(content, "rules.json", None, None)
        assert result == obj
