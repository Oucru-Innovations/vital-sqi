Installation Guide
==================

This section provides instructions for installing the **vital_sqi** library.

Prerequisites
-------------
Before installing **vital_sqi**, ensure you have the following installed:

- Python 3.7 or later
- pip (Python package manager)
- Virtual environment tools like `venv` or `conda` (recommended)

Installation Steps
------------------
Follow the steps below to install **vital_sqi**:

1. **Clone the GitHub repository (optional):**
   If you want to explore or contribute to the source code, clone the official repository:

   .. code-block:: bash

      git clone https://github.com/Oucru-Innovations/vital-sqi.git
      cd vital-sqi

2. **Install from PyPI:**
   You can install the library directly using `pip`:

   .. code-block:: bash

      pip install vitalSQI-toolkit

3. **Install from Source:**
   Alternatively, install the library from the cloned repository:

   .. code-block:: bash

      git clone https://github.com/Oucru-Innovations/vital-sqi.git
      cd vital-sqi
      pip install .

Verifying the Installation
--------------------------
To ensure the library is installed correctly, run the following command:

.. code-block:: python

   import vital_sqi
   print(vital_sqi.__version__)

If the library is installed correctly, this will display the installed version.

Updating **vital_sqi**
----------------------
To update to the latest version, use the following command:

.. code-block:: bash

   pip install --upgrade vitalSQI-toolkit

Uninstallation
--------------
To remove **vital_sqi** from your system, use:

.. code-block:: bash

   pip uninstall vitalSQI-toolkit

Issues and Support
------------------
If you encounter any issues during installation, please refer to the `GitHub Issues page <https://github.com/Oucru-Innovations/vital-sqi/issues>`_ or contact the maintainers.
