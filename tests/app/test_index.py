import pytest
from dash import html
from dash.testing.application_runners import import_app
from dash.testing.browser import Browser
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import os


@pytest.fixture
def app_runner():
    """Fixture to import the Dash app."""
    app = import_app("vital_sqi.app.index")
    return app


def test_app_layout(dash_duo, app_runner):
    """Test if the Dash app layout is rendered correctly."""
    dash_duo.start_server(app_runner)

    # Wait for the sidebar to render
    dash_duo.wait_for_element(".display-4", timeout=10)

    # Check if the sidebar is rendered
    sidebar = dash_duo.find_element(".display-4")
    assert sidebar.text == "Menu"

    # Check if navigation links exist
    nav_links = dash_duo.find_elements("a")
    assert any(link.text == "Home" for link in nav_links)
    assert any(link.text == "Dashboard 1" for link in nav_links)


# def test_display_page_callback(dash_duo, app_runner):
#     """Test page navigation callback."""
#     dash_duo.start_server(app_runner)

#     # Wait for the Dashboard 1 link
#     dashboard_1_link = dash_duo.wait_for_element("#dashboard_1_link", timeout=20)

#     # Enable the link if disabled (simulate app state update)
#     if "disabled" in dashboard_1_link.get_attribute("class"):
#         dash_duo.driver.execute_script(
#             "arguments[0].classList.remove('disabled');", dashboard_1_link
#         )

#     # Force URL update as a fallback
#     dash_duo.driver.execute_script(
#         "window.history.pushState({}, '', '/views/dashboard1');"
#     )
#     dash_duo.driver.execute_script(
#         "window.dispatchEvent(new Event('popstate'));"
#     )

#     # Alternatively, use ActionChains for clicking
#     ActionChains(dash_duo.driver).move_to_element(dashboard_1_link).click().perform()

#     # Wait for content update and validate
#     dash_duo.wait_for_text_to_equal("#page-content", "Dashboard 1 Content", timeout=30)
#     content_text = dash_duo.find_element("#page-content").text
#     assert "Dashboard 1 Content" in content_text, f"Unexpected content: {content_text}"


# def test_update_output_callback(dash_duo, app_runner):
#     """Test the upload-data callback for handling uploads."""
#     dash_duo.start_server(app_runner)

#     # Wait for the upload component to appear
#     upload_component = dash_duo.wait_for_element("#upload-data", timeout=30)

#     # Simulate file upload
#     file_path = os.path.abspath("tests/test_data/ecg_test1.csv")  # Replace with actual file path
#     input_element = dash_duo.driver.find_element(By.CSS_SELECTOR, "#upload-data input[type='file']")
#     input_element.send_keys(file_path)

#     # Validate the callback response
#     dataframe_store = dash_duo.wait_for_element("#dataframe", timeout=30)
#     assert dataframe_store.get_attribute("data") is not None


# def test_upload_rule_callback(dash_duo, app_runner):
#     """Test the upload-rule callback for handling rule uploads."""
#     dash_duo.start_server(app_runner)

#     # Wait for the upload-rule component
#     upload_component = dash_duo.wait_for_element("#upload-rule", timeout=30)

#     # Simulate rule upload
#     file_path = os.path.abspath("tests/test_data/rule_dict_test.json")  # Replace with actual file path
#     input_element = dash_duo.driver.find_element(By.CSS_SELECTOR, "#upload-rule input[type='file']")
#     input_element.send_keys(file_path)

#     # Verify the rule data is updated
#     rule_store = dash_duo.wait_for_element("#rule-set-store", timeout=30)
#     assert rule_store.get_attribute("data") is not None
#     assert dash_duo.get_logs() == [], "Errors found in browser console"
