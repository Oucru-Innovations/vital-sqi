import os
import pytest
from selenium.webdriver import ChromeOptions, FirefoxOptions
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from dash.testing.composite import DashComposite

def pytest_ignore_collect(path):
    if "vital_sqi/app" in str(path):
        return True

def pytest_addoption(parser):
    """Add a command-line option for selecting the browser."""
    parser.addoption(
        "--browser",
        action="store",
        default="chrome",
        help="Browser to use for tests (default: firefox)",
    )


@pytest.fixture
def browser(request):
    return request.config.getoption("--browser")


@pytest.fixture
def dash_duo(request, dash_thread_server, tmpdir) -> DashComposite:
    """Setup DashComposite with the appropriate browser."""
    browser = request.config.getoption("--browser").lower()

    if browser == "chrome":
        options = ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver_path = ChromeDriverManager().install()
        os.environ["PATH"] += os.pathsep + os.path.dirname(driver_path)
    elif browser == "firefox":
        options = FirefoxOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver_path = GeckoDriverManager().install()
        os.environ["PATH"] += os.pathsep + os.path.dirname(driver_path)
    else:
        raise ValueError(f"Unsupported browser: {browser}")

    # Initialize DashComposite
    with DashComposite(
        dash_thread_server,
        browser=browser,
        options=options,
        download_path=tmpdir.mkdir("download").strpath,
    ) as dc:
        yield dc
