Storage
========

See the full API for the module at :ref:`storage_module`. The abstract base
class :class:`~orchestrator.storage.storage_base.Storage` provides the standard
interface for all of the concrete implementations.

The primary purpose of the Storage module is to deal with the storage of data
within the Orchestrator code. We currently support data storage to disk (local)
and to a Colabfit database. The
:class:`~orchestrator.storage.colabfit.ColabfitStorage` module is the
recommended vehicle for saving and accessing data with Orchestrator. The
interested reader should refer to
`Colabfit <https://github.com/EFuem/postgresql-cfkit>`_ for more details.
The usage of Colabfit requires access to a PostgreSQL database, usually
specified via a credential file.

Once a PostgreSQL instance has been created, construct a JSON credential file
as shown below:

   .. code-block:: json

      {
         "database_path":"<path>",
         "database_name":"<name>",
         "database_port":"<port>",
         "database_password":"<password>",
         "database_user":"<user>",
      }

To use this file directly, create a storage module instance as:

   .. code-block:: python

      from orchestrator.utils.setup_input import init_and_validate_module_type
      storage_inputs = {"storage_type": "COLABFIT", "storage_args": {"credential_file": "sql_credentials.json"}}
      storage = init_and_validate_module_type('storage', {'storage': storage_inputs})

Use Cases
---------

Basic usage
^^^^^^^^^^^

A simple example of how to use the storage class in a standalone application
can be seen below. The example uses the Colabfit storage class, but it can
be easily adapted to Local. Some initial configuration data in extxyz format
should be supplied. It first needs to be converted into
an ASE Atoms object for the storage module to ingest (
:class:`~orchestrator.storage.storage_base.Storage` always uses ASE Atoms
for both input and output).

   .. code-block:: python

      from orchestrator.utils.input_output import ase_glob_read
      from orchestrator.utils.setup_input import init_and_validate_module_type

      storage_inputs = {"storage_type": "COLABFIT", "storage_args": {"credential_file": "sql_credentials.json"}}
      storage = init_and_validate_module_type('storage', {'storage': storage_inputs})

      # If using for first time, tables need to be made
      storage.setup_tables()

      # Load initial data from disk
      init_configs = ase_glob_read('sample_configs/')

      # if the data contains energy/forces, make sure these are correctly read
      # you can let ColabFit know how to locate these properties by specifying
      # a property mapping as below. If they are called 'Energy' and 'force'
      # in the files located in the above directory.
      storage.set_property_map(keys={'energy_field': 'Energy', 'force_field': 'force'})

      # add the data to the database
      dataset_name = 'demo_dataset'
      # we will save the dataset_handle, which uniquely identifies the dataset
      dataset_handle = storage.new_dataset(dataset_name, init_configs)

      # extract the data from the database
      # rename_properties ensures the properties are renamed to their
      # original keys, i.e., Energy and force, when they are returned
      # from the database
      data_from_db = storage.get_data(dataset_handle, rename_properties=True)
      print(f'Number of configs in the original dataset: {len(data_from_db)}')

      # add more data (this could come from an Oracle or Simulator run)
      additional_configs = ase_glob_read('./more_data')
      # Colabfit automatically versions the dataset, so we save the new handle
      updated_handle = storage.add_data(
         dataset_handle,
         additional_configs
      )

      # extract the data from the database
      data_from_db = storage.get_data(updated_handle)
      print(f'Number of configs in the updated dataset: {len(data_from_db)}')

The example is simply reading in sample configurations into the
``init_configs`` variable. If instead, calculation ids or ``calc_ids`` were
provided, the user could pass those ``calc_ids`` one at a time to the
:meth:`~orchestrator.oracle.oracle_base.Oracle.parse_for_storage` method or
the :meth:`~orchestrator.oracle.oracle_base.Oracle.data_from_calc_ids` method.
The latter method calls
:meth:`~orchestrator.oracle.oracle_base.Oracle.parse_for_storage` and
additionally modifies metadata. Currently,
:meth:`~orchestrator.oracle.oracle_base.Oracle.save_labeled_configs` calls
:meth:`~orchestrator.oracle.oracle_base.Oracle.data_from_calc_ids` and pulls
the provided input parameters of the simulations and combines this with other
pieces of metadata to pass to database. An example of how to implement this
into a script could look like:

   .. code-block:: python

      # Should previously have instantiated the oracle, storage, and workflow
      # classes.
      calc_ids = oracle.run(
         path_type="tests",
         input_args=extra_input_args,
         configs=init_configs,
         workflow=workflow,
         job_details=job_details
      )

      workflow.block_until_completed(calc_ids)

      oracle.save_labeled_configs(
         calc_ids,
         storage,
         workflow=workflow
      )

These examples show that
:meth:`~orchestrator.oracle.oracle_base.Oracle.save_labeled_configs` can be
used to upload data. An additional method,
:meth:`~orchestrator.simulator.simulator_base.Simulator.save_configurations`,
may also be used for uploading data into the databsae.

It is important to understand the difference between ``dataset_name`` and
``dataset_handle``. While the ``dataset_name`` is a human readable string,
``dataset_handle`` will generally not be. In Colabfit, a dataset handle will
always start with ``DS_`` followed by a unique hash, followed by the version
index, i.e.: ``DS_inxkic391zv0_0``

.. _upload_external:

Uploading External Calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the case of needing to upload pre-existing calculations and data, one could
call the orchestrator functions needed to upload the data. We will work under
the assumption that there is no existing database and a new one will be
created.

   .. code-block:: python

      import os
      from datetime import datetime
      from orchestrator.oracle.espresso import EspressoOracle
      from orchestrator.workflow.local import LocalWF
      from orchestrator.storage.colabfit import ColabfitStorage
      from orchestrator.utils.data_standard import METADATA_KEY

      # Initialize the oracle, workflow, and storage types.
      # This example will assume Quantum Espresso was used.
      oracle = EspressoOracle(code_path='/path/to/pw.x')
      workflow = LocalWF()
      storage = ColabfitStorage(credential_file='/path/to/credential_file')

      # Set the paths to where the calculations occured.
      # We will assume the paths were saved to a txt file.
      paths = []
      with open('paths.txt', 'r') as infile:
         for line in infile:
            paths.append(line.strip())

      # Currently we enforce that all configurations in a dataset should
      # Have the same input parameters. This will check if they are the same.
      configs = []
      code_parameters = {}
      for path in paths:
         config = oracle.parse_for_storage(path)
         configs.append(config)
         metadata = config.info[METADATA_KEY]
         parameters = metadata.pop('code_parameters', None)

         if not code_parameters.get('code', None):
            code_parameters['code'] = parameters['code']
            code_parameters['universal'] = parameters['universal']
         else:
            if code_parameters['universal'] != parameters['universal']:
               raise ValueError(
                  f'The provided configs, {configs}, have differing '
                  'universal parameters which is not currently '
                  'supported.')

      current_date = datetime.today().strftime('%Y-%m-%d')
      user = os.getlogin()
      authors = f'Uploaded by {user}'
      dataset_metadata = {
         'description': (f'data uploaded by {user} on '
         f'{current_date}'),
         'parameters': code_parameters
      }

      dataset_name = 'Set_a_unique_dataset_name'
      unique = storage.check_if_dataset_name_unique(dataset_name)
      if unique:
         new_handle = storage.new_dataset(dataset_name, configs,
                                          dataset_metadata)
      else:
         raise NameError(f'{dataset_name} already exists in the database.)

Inspecting the Database
^^^^^^^^^^^^^^^^^^^^^^^

To view a summary and overview of the contents of a database, use the provided
:meth:`~orchestrator.storage.colabfit.ColabfitStorage.list_data()` function.
Note that for ColabfitStorage modules, this functionality is tied to the
instance of the Colabfit database and that a list of matching dictionaries may
be provided by setting `capture_output = True`.

Parsing for Storage
-------------------

Other modules can interface with the Storage module by providing data to be
stored. When doing so, they should supply data in a standard format, which is
defined in our ``data_standard.py`` and provides constants such as
``ENERGY_KEY`` to use for consistency. Individual parsing functions are written
to convert the native output from their module into this data standard. See the
:meth:`~orchestrator.oracle.espresso.EspressoOracle.parse_for_storage` method
for an example. Any module can handle the parsing task however it sees fit,
(parsing itself, using tools from ASE or NOMAD, etc.) but is responsible for
providing the data to storage as a list of ASE Atoms.

Development Plan
----------------

The current implementation supports the storage of basic properties to train
an interatomic potential, such as energy, forces, coordinates, cell lattice and
stress tensor. It can also support atomic descriptors and selection masks.
Quality of life methods will continue to be added as well as more robust
metadata handling for data generated by Oracles.

Inheritance Graph
-----------------

.. inheritance-diagram::
   orchestrator.storage.factory
   orchestrator.storage.local
   orchestrator.storage.colabfit
   :parts: 3
