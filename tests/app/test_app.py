"""Unit tests for the Dash app's top-level layout and routing.

Browser-free: every test imports the app object and exercises the
Python-side logic directly.
"""

import pytest
from dash import dcc, html


def _get_app():
    """Import the index module (which sets app.layout) and return the app."""
    import vital_sqi.app.index  # noqa: F401  — side-effect: sets app.layout
    from vital_sqi.app.app import app
    return app


def _find_components(component, component_type):
    """Recursively collect all components of *component_type* in the layout."""
    found = []
    if isinstance(component, component_type):
        found.append(component)
    children = getattr(component, "children", None)
    if children is None:
        return found
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if child is not None:
            found.extend(_find_components(child, component_type))
    return found


# ---------------------------------------------------------------------------
# App object
# ---------------------------------------------------------------------------


class TestAppObject:
    def test_app_is_dash_instance(self):
        import dash
        assert isinstance(_get_app(), dash.Dash)

    def test_app_has_layout(self):
        assert _get_app().layout is not None

    def test_layout_is_div(self):
        assert isinstance(_get_app().layout, html.Div)

    def test_suppress_callback_exceptions(self):
        assert _get_app().config.suppress_callback_exceptions is True


# ---------------------------------------------------------------------------
# Layout structure (Phase 5: streamlined sidebar)
# ---------------------------------------------------------------------------


class TestLayoutStructure:
    @pytest.fixture(autouse=True)
    def setup(self):
        self._app = _get_app()

    def test_contains_url_location(self):
        locations = _find_components(self._app.layout, dcc.Location)
        assert any(loc.id == "url" for loc in locations)

    def test_contains_required_stores(self):
        stores = _find_components(self._app.layout, dcc.Store)
        ids = {s.id for s in stores}
        # Phase 5 promotes inspect-decisions to the app root so the Export
        # view can read it without re-running the classifier.
        for required in ("dataframe", "raw-waveform", "segment-milestones",
                         "inspect-decisions"):
            assert required in ids, f"missing store: {required}"

    def test_does_not_contain_dropped_stores(self):
        stores = _find_components(self._app.layout, dcc.Store)
        ids = {s.id for s in stores}
        # Phase 5 dropped rule-set-store and rule-dataframe.
        assert "rule-set-store" not in ids
        assert "rule-dataframe" not in ids

    def test_contains_page_content(self):
        divs = _find_components(self._app.layout, html.Div)
        assert any(getattr(d, "id", None) == "page-content" for d in divs)

    def test_sidebar_heading_text(self):
        h2s = _find_components(self._app.layout, html.H2)
        assert any(
            "vital_sqi" in str(getattr(h, "children", "")).lower() for h in h2s
        )

    def test_sidebar_has_only_phase5_tabs(self):
        # Compute, Inspect, Calibrate, Export.  No Home / Rules / Apply.
        import dash_bootstrap_components as dbc

        links = _find_components(self._app.layout, dbc.NavLink)
        link_ids = {getattr(l, "id", None) for l in links}
        assert link_ids == {
            "compute_link", "inspect_link", "calibrate_link", "export_link",
        }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestDisplayPageCallback:
    @pytest.fixture(autouse=True)
    def setup(self):
        import vital_sqi.app.index as idx
        self.display_page = idx.display_page

    def test_root_returns_compute(self):
        # Phase 5: the default landing page is Compute (Home is gone).
        result = self.display_page("/")
        flat = str(result)
        assert "compute-upload-waveform" in flat or "compute-run-btn" in flat

    def test_compute_route(self):
        result = self.display_page("/views/compute")
        assert "compute-run-btn" in str(result)

    def test_inspect_route(self):
        result = self.display_page("/views/inspect")
        flat = str(result)
        assert "inspect-timeline" in flat or "inspect-summary" in flat

    def test_dashboard1_route_redirects_to_inspect(self):
        # Bookmark courtesy: /views/dashboard1 still serves the Inspect view.
        result = self.display_page("/views/dashboard1")
        flat = str(result)
        assert "inspect-timeline" in flat or "inspect-summary" in flat

    def test_calibrate_route(self):
        result = self.display_page("/views/calibrate")
        assert "calibrate-run-btn" in str(result)

    def test_export_route(self):
        result = self.display_page("/views/export")
        assert "export-decisions-btn" in str(result)

    def test_unknown_route_falls_back_to_compute(self):
        result = self.display_page("/nonexistent")
        assert "compute-run-btn" in str(result)


# ---------------------------------------------------------------------------
# Sidebar tab gating
# ---------------------------------------------------------------------------


class TestSidebarGate:
    @pytest.fixture(autouse=True)
    def setup(self):
        import vital_sqi.app.index as idx
        self.gate = idx._gate_tabs

    def test_locked_when_no_data(self):
        inspect_disabled, export_disabled = self.gate(None)
        assert inspect_disabled is True and export_disabled is True

    def test_unlocked_when_data_present(self):
        inspect_disabled, export_disabled = self.gate({"a": {0: 1}})
        assert inspect_disabled is False and export_disabled is False
