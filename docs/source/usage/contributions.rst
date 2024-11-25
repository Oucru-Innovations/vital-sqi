Contributions
=============

Thank you for your interest in contributing to **vital_sqi**! This guide provides a detailed workflow for setting up your development environment, contributing new features or fixes, and generating documentation.

Creating a Virtual Environment
------------------------------
A virtual environment helps to isolate dependencies for different projects, ensuring no conflicts arise between packages. Follow these steps:

**Using `venv` (Python >= 3.3):**

1. **Install `venv`:**
   .. code-block:: console

      python -m pip install venv

2. **Create a Virtual Environment:**
   .. code-block:: console

      python -m venv <environment-name>

3. **Activate the Virtual Environment:**
   .. code-block:: console

      source <environment-name>/bin/activate    # macOS/Linux
      <environment-name>\Scripts\activate       # Windows

**Using `virtualenv`:**

1. **Install `virtualenv`:**
   .. code-block:: console

      pip install virtualenv

2. **Create a Virtual Environment:**
   .. code-block:: console

      virtualenv -p <python-path> <environment-name>

3. **Activate the Virtual Environment:**
   .. code-block:: console

      source <environment-name>/bin/activate    # macOS/Linux

To deactivate the environment, simply run:

.. code-block:: console

   deactivate

.. note::
   For PyCharm users, you can configure virtual environments directly in the IDE. Students may qualify for a free license.

---

Forking the Repository
----------------------
1. Navigate to the official repository: `https://github.com/Oucru-Innovations/vital-sqi`.
2. Click the **Fork** button (top-right corner).
3. This creates a copy of the repository under your GitHub account, including all branches, commits, and files.

---

Setting Up Your Fork Locally
----------------------------
1. **Clone the Forked Repository:**
   .. code-block:: console

      git clone https://github.com/<your-username>/vital-sqi.git
      cd vital-sqi

2. **Add the Upstream Remote:**
   This ensures you can sync your fork with the original repository:

   .. code-block:: console

      git remote add upstream https://github.com/Oucru-Innovations/vital-sqi.git

3. **Verify Remotes:**
   .. code-block:: console

      git remote -v

4. **Optional: Clone Specific Branches:**
   If you only need certain branches, use:

   .. code-block:: console

      git clone -b <branch-name> https://github.com/<your-username>/vital-sqi.git

---

Understanding the Repository Structure
--------------------------------------
After cloning the repository, you will see the following structure:

.. code-block:: text

   vital-sqi/
   ├── docs/                   # Documentation files
   │   ├── build/              # Generated documentation (html)
   │   ├── source/             # Sphinx source files
   │       ├── conf.py         # Sphinx configuration
   │       └── index.rst       # Documentation index
   ├── examples/               # Example scripts for users
   ├── src/                    # Source code for the library
   │   ├── vital_sqi/          # Main package
   │   │   ├── core/           # Core classes and algorithms
   │   │   ├── tests/          # Unit tests
   │   │   ├── utils/          # Utility functions
   ├── requirements.txt        # Project dependencies
   ├── setup.py                # Package installation script
   └── Makefile                # Build automation script (e.g., for docs)

---

Installing the Package in Editable Mode
---------------------------------------
Editable mode allows you to test changes to the source code without reinstalling the package.

1. **Install Dependencies:**
   Navigate to the project root and run:

   .. code-block:: console

      python -m pip install -r requirements.txt

2. **Install the Package:**
   .. code-block:: console

      python -m pip install -e .[dev]

---

Generating Documentation
-------------------------
The documentation is built using **Sphinx**. Follow these steps:

1. **Install Required Libraries:**
   If not already installed:

   .. code-block:: console

      python -m pip install sphinx sphinx-gallery sphinx-rtd-theme matplotlib

2. **Generate HTML Documentation:**
   Navigate to the `docs` folder and run:

   .. code-block:: console

      make html

3. **Preview Locally:**
   Open `docs/build/html/index.html` in your browser.

4. **Deploy to GitHub Pages:**
   Use the pre-configured `make github` command:

   .. code-block:: console

      make github

---

Running Tests
-------------
Tests ensure your changes work as expected and do not break existing functionality.

1. **Install `pytest`:**
   .. code-block:: console

      python -m pip install pytest

2. **Run Tests:**
   .. code-block:: console

      pytest

---

Submitting Contributions
------------------------
1. **Create a Feature Branch:**
   Use a descriptive branch name:

   .. code-block:: console

      git checkout -b feature/your-feature-name

2. **Make Your Changes:**
   Update the code, tests, and documentation as needed.

3. **Run Tests and Build Documentation:**
   Verify all tests pass and documentation builds without errors.

4. **Commit and Push:**
   .. code-block:: console

      git add .
      git commit -m "Add feature: Description of your changes"
      git push origin feature/your-feature-name

5. **Open a Pull Request:**
   On GitHub, open a PR from your branch to the `main` branch of the official repository.

---

Helpful Tips
------------
- **Style Guidelines**: Follow `PEP 8` for coding style.
- **Type Annotations**: Use type hints for all functions.
- **Docstrings**: Write docstrings in the NumPy format.
- **Testing**: Write unit tests for any new functionality.
- **Documentation**: Add or update relevant `.rst` files.

Happy coding and thank you for contributing to **vital_sqi**!
