.. _installation:

Installation
============

.. note::

   The full Python environment required for Orchestrator is quite large
   (typically several gigabytes). Please ensure you have sufficient disk space
   in your install location before proceeding. PyTorch, in particular, has
   significant memory requirements.

Because the Orchestrator codebase is under active development with frequent
updates, we recommend installing it in an isolated micromamba environment using
an editable installation. An automated installation script is provided to
streamline this process and configure all necessary paths and environment
variables.

Using the Auto Installation Script
----------------------------------

The automatic installation script is located at the root of the Orchestrator
repository. You can also download it directly:

:download:`auto_install_orchestrator.sh <../../../auto_install_orchestrator.sh>`

To get started, you may either copy or download the script manually, or use the
following terminal command:

.. code-block:: console

   $ wget https://raw.githubusercontent.com/LLNL/orchestrator/refs/heads/main/auto_install_orchestrator.sh

Place this script in an **empty directory** where you want the environment to
be installed. Note that micromamba is entirely self-contained and will not
interfere with other Python environments on your system.

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Before running the script, you may wish to configure certain environment
variables:

- ``KIM_API_ENVIRONMENT_COLLECTION``: Set this to a shared-access path if you
  want to share installed potentials with other users on your machine.
- ``LAMMPS_PYTHON_MODULE`` and ``LAMMPS_LIBRARY_PATH``: Set these if you plan
  to use LAMMPS. For your convenience, we provide
  :download:`an example LAMMPS build script <./lammps_build.sh>`.
- ``MPI_LIB``: The script will attempt to automatically detect your MPI
  library path, which needs to be set for ``mpi4py``. If you do not have
  ``mpicc`` available in your path, you should manually set this variable to
  your MPI ``lib/`` path.


.. _script_install:

Running the Script
~~~~~~~~~~~~~~~~~~

To install Orchestrator and its dependencies, run:

.. code-block:: console

   $ bash auto_install_orchestrator.sh

.. note::

   By default, the environment will be named ``orchestrator``. To use a
   different name, set the ``ENV_NAME`` environment variable before running
   the script.

After installation, a ``source_me.sh`` file will be generated. Use this file to
activate the environment:

.. code-block:: console

   $ source /path/to/source_me.sh

Updates to Orchestrator
-----------------------

The auto install script installs Orchestrator in editable mode. To update your
installation, navigate to the cloned repository (located within your chosen
installation directory), and pull the latest changes or switch branches as
needed. You do **not** need to reinstall the environment for updates.

Dependencies
------------

The following key dependencies are required for Orchestrator:

+-------------------+---------------------------------------------------------+
| Dependency        | Description/Link                                        |
+===================+=========================================================+
| PyTorch           | https://pytorch.org/                                    |
+-------------------+---------------------------------------------------------+
| numpy             | https://numpy.org/                                      |
+-------------------+---------------------------------------------------------+
| ASE               | https://wiki.fysik.dtu.dk/ase/                          |
+-------------------+---------------------------------------------------------+
| KIM-API           | https://openkim.org/kim-api/                            |
+-------------------+---------------------------------------------------------+
| kimpy             | https://github.com/openkim/kimpy                        |
+-------------------+---------------------------------------------------------+
| KliFF             | https://github.com/openkim/kliff                        |
+-------------------+---------------------------------------------------------+
| colabfit-kit      | https://pypi.org/project/colabfit-kit/                  |
+-------------------+---------------------------------------------------------+
| kimkit            | https://pypi.org/project/kimkit/                        |
+-------------------+---------------------------------------------------------+
| pandas            | https://pandas.pydata.org/                              |
+-------------------+---------------------------------------------------------+
| scikit-learn      | https://scikit-learn.org/stable/                        |
+-------------------+---------------------------------------------------------+
| scipy             | https://scipy.org/                                      |
+-------------------+---------------------------------------------------------+
| h5py              | https://www.h5py.org/                                   |
+-------------------+---------------------------------------------------------+
| pytest            | https://docs.pytest.org/en/7.4.x/  (for unit tests)     |
+-------------------+---------------------------------------------------------+
| pre-commit        | https://pre-commit.com/ (for contributions)             |
+-------------------+---------------------------------------------------------+
| sphinx            | https://www.sphinx-doc.org/en/master/ (for docs)        |
+-------------------+---------------------------------------------------------+
| furo              | https://github.com/pradyunsg/furo (for docs)            |
+-------------------+---------------------------------------------------------+

.. note::

   All dependencies will be installed automatically by following the
   instructions above. You do **not** need to install them manually.

Optional Dependencies
---------------------

Some Orchestrator modules require additional (optional) dependencies. These
can be installed as needed using:

.. code-block:: console

   $ pip install .[<DEPENDENCY_NAME>]

Replace ``<DEPENDENCY_NAME>`` with the appropriate name from the
``pyproject.toml`` file (for example, ``LTAU``).

The available optional dependency sets are:

- AIIDA
- FIMMATCHING
- LTAU
- QUESTS

.. note::

   By default, the auto install script installs **all** optional dependencies.
   You can modify this behavior by editing the script before running it.

If you have any questions or need further assistance, please consult the
repository documentation, open an issue, or email us at
orchestrator-help@llnl.gov.
