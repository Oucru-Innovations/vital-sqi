Development Guide
=================

This section provides a guide for developers who want to contribute to **vital_sqi** or customize its functionalities.

Setting Up a Development Environment
-------------------------------------
To contribute or make changes to **vital_sqi**, follow these steps to set up a local development environment.

1. **Clone the Repository:**
   Clone the official repository from GitHub:

   .. code-block:: bash

      git clone https://github.com/Oucru-Innovations/vital-sqi.git
      cd vital-sqi

2. **Create a Virtual Environment:**
   It is recommended to use a virtual environment to avoid dependency conflicts:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate    # On macOS/Linux
      venv\Scripts\activate       # On Windows

3. **Install Development Dependencies:**
   Install the library along with all necessary development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

   This command installs the library in editable mode and includes tools for testing and linting.

4. **Verify the Development Setup:**
   Run the following commands to check that everything is working:

   .. code-block:: bash

      pytest tests/
      flake8 src/vital_sqi

      # Optional: Run pre-commit hooks
      pre-commit install
      pre-commit run --all-files

Development Workflow
--------------------
1. **Create a Feature Branch:**
   Always create a new branch for each feature or bug fix:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. **Write Code:**
   Add your new functionality or make improvements to the existing codebase. Follow the style guidelines to ensure consistency.

3. **Run Tests:**
   Before committing changes, make sure all tests pass:

   .. code-block:: bash

      pytest tests/

4. **Commit Your Changes:**
   Use clear and concise commit messages:

   .. code-block:: bash

      git add .
      git commit -m "Add feature: Description of your changes"

5. **Push Your Branch and Open a Pull Request:**
   Push your branch to GitHub and open a pull request:

   .. code-block:: bash

      git push origin feature/your-feature-name

Code Style and Guidelines
-------------------------
- **Coding Style**: Follow the `PEP 8` guidelines for Python code.
- **Type Annotations**: Use type hints for all functions and methods.
- **Docstrings**: Write clear and concise docstrings using NumPy style.
- **Pre-Commit Hooks**: Pre-commit hooks are configured to enforce style checks. Run the following command to install them:

   .. code-block:: bash

      pre-commit install

Testing
-------
Write unit tests for any new functionality in the `tests` directory. To run tests, use:

.. code-block:: bash

   pytest tests/

Documentation Updates
---------------------
If your changes include new features or functionality, update the documentation in the `docs` directory. To preview your changes locally:

1. Build the documentation:

   .. code-block:: bash

      cd docs
      make html

2. Open the generated HTML files in `docs/build/html/` to view your changes.

Contributing
------------
If you're interested in contributing, please review the `CONTRIBUTING.md` file in the repository for detailed guidelines.

Support
-------
If you have questions about development or encounter issues, open a discussion or issue on the GitHub repository: https://github.com/Oucru-Innovations/vital-sqi
