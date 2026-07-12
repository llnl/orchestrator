.. _aiida_setup:

AiiDA
=====

The `AiiDA  <https://www.aiida.net/>`_ software suite is used together with the Orchestrator to manage the
complex simulations that are needed for training the MLIPs. Follow the steps
below to install and configure AiiDA.

Installation Steps
------------------

At this point you should have setup a python virtual environment or conda
environment to run the Orchestrator. The following steps will assist you
with installing the AiiDA software.

1. **Install AiiDA**: When you installed the Orchestrator, you had the option to install AiiDA with

   .. code-block:: bash

      pip install -e '.[AiiDA]'

If you forgot to install with this command, you can install the required packages with

   .. code-block:: bash

      pip install aiida-core aiida-vasp aiida-quantumespresso aiida-pseudo

which will install the main code plus plug-ins for VASP and QuantumEspresso.

2. **Set Up the Persistent services**: AiiDA requires a PostgreSQL database
and RabbitMQ server. Install PostgreSQL and create a database for AiiDA. Also
install a RabbitMQ server. There is plenty of documentation on the web and
this will not be covered in the documentation.

3. **Install a Profile**: Configure AiiDA with a profile using the
`verdi setup` command or a YAML configuration file. An example is given below.

Example YAML Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use a YAML file to set up an AiiDA profile for easier
configuration. Below is an example file:

.. code-block:: yaml

   non_interactive: y
   profile: aiida_username
   email: your@email.com
   first_name: First name
   last_name: Last name
   institution: your_institution
   use_rabbitmq: y
   database_hostname: db_hostname
   database_port: 5432
   database_name: aiida_db_name
   database_username: db_user
   database_password: db_password
   repository_uri: file:///path/to/aiida/repository

An example profile yaml file is provided
:download:`aiida_profile.yaml <../../../orchestrator/oracle/aiida/config_files/aiida_profile.yaml>`.

Steps to Use the YAML File
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Save the YAML file as ``aiida_profile.yaml``.

2. Run the following command to set up the profile:

   .. code-block:: bash

      verdi profile setup core.psql_dos --config aiida_profile.yaml

3. Verify the installation by running:

   .. code-block:: bash

      verdi status

Customizations
##############

- Replace ``localhost``, ``aiida_db``, ``aiida_user``, and ``aiida_password`` with your database connection details.
- Update the ``repository_uri`` to point to a valid directory for storing AiiDA data.
- Modify the ``email``, ``first_name``, and ``last_name`` fields to reflect your user information.

Connecting AiiDA to a RabbitMQ Server
-------------------------------------

AiiDA uses RabbitMQ as a message broker for its process communication system.
Follow the steps below to configure AiiDA to connect to a RabbitMQ server.

Prerequisites
~~~~~~~~~~~~~

Before connecting AiiDA to RabbitMQ, ensure the following:

1. RabbitMQ is installed and running on your system.
2. You have access to the RabbitMQ server's hostname, port, username, and password.
3. AiiDA is installed and configured on your system.

Configuring RabbitMQ for AiiDA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Create a RabbitMQ user**:

   Log in to the RabbitMQ management interface or use the RabbitMQ CLI to
   create a user for AiiDA.

   .. code-block:: bash

      rabbitmqctl add_user aiida_user aiida_password

2. **Set permissions for the user**:

   Assign permissions to the user to access the required resources.

   .. code-block:: bash

      rabbitmqctl set_permissions -p / aiida_user ".*" ".*" ".*"

3. **Note the connection details**:

   - Hostname: ``rabbitmq_server_hostname`` (e.g., ``localhost`` or the server's IP address)
   - Port: ``5672`` (default RabbitMQ port)
   - Username: ``aiida_user``
   - Password: ``aiida_password``

Some services might already setup and provide this information.

Configuring AiiDA to Use RabbitMQ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Edit the AiiDA configuration file**:

   Locate the AiiDA configuration file, typically found at
   ``~/.aiida/config.json`` or if ``$AIIDA_PATH`` has been set it will be found
   at ``$AIIDA_PATH/.aiida.config.json``. Add or update the RabbitMQ connection
   details in the configuration file. You can see where your ``config.json``
   file is located with

   .. code-block:: bash

      verdi status

   Add or update the RabbitMQ connection details in the configuration file.

   Example configuration:

   .. code-block:: json

      {
        "process_control": {
          "backend": "core.rabbitmq",
            "config": {
              "broker_protocol": "amqp",
              "broker_username": "guest",
              "broker_password": "guest",
              "broker_host": "127.0.0.1",
              "broker_port": 5672,
              "broker_virtual_host": ""
            }
          }
        }
      }

   These values are the default values for a RabbitMQ launched on a
   personal device. You will need to update your specific values that you
   setup at installation.

   Additionally, some servers require additional information to allow AiiDA
   to connect to the RabbitMQ service. One such example might look like the
   following. After ``broker_virtual_host`` add the following lines.

   .. code-block:: json

      "broker_parameters": {
        "no_verify_ssl": "1",
        "cafile": "/etc/pki/tls/cert.pem"
      }

2. **Restart the AiiDA daemon**:

   After updating the configuration file, restart the AiiDA daemon to apply
   the changes.

   .. code-block:: bash

      verdi daemon restart

Testing the Connection
~~~~~~~~~~~~~~~~~~~~~~

Verify that AiiDA is successfully connected to RabbitMQ by checking the
status of the daemon.

.. code-block:: bash

  verdi daemon status

or

.. code-block:: bash

  verdi status

If the connection is successful, the daemon should be running without errors.

Troubleshooting
~~~~~~~~~~~~~~~

- **Connection Errors**:
  Ensure the RabbitMQ server is reachable from the machine running AiiDA.
  Check firewall settings and network connectivity.

- **Authentication Errors**:
  Verify the username and password in the AiiDA configuration file match those
  created in RabbitMQ.

- **Port Issues**:
  If RabbitMQ is running on a non-default port, update the ``url`` in the AiiDA
  configuration file accordingly.

Setting Up a Computer and Code in AiiDA
---------------------------------------

After installing and configuring AiiDA, you need to set up a **computer**
(representing the computational resources) and a **code** (representing the
executable). Follow the steps below to configure these components. You are able
to provide the following information line by line on the command line but the
outline below will show how to create the computer and code using yaml files.

Computer Example
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   label: my_computer
   hostname: localhost
   description: Local machine for testing
   scheduler: core.slurm
   transport: core.ssh
   work_dir: /scratch/{username}/aiida
   mpirun_command: mpirun -np {tot_num_mpiprocs}
   mpiprocs_per_machine: "# processors per node"
   prepend_text: ""
   append_text: ""
   shebang: "#!/bin/bash"

``label`` is what the remote server is called within AiiDA and will be
referenced later.

``hostname`` is the address of the remote server, i.e. where you ssh.

The ``mpirun_command`` should be set to the appropriate mpi command you
would use on the designated server. For example, on a Slurm based-system we
use ``srun`` and that line would be set to ``srun -n {tot_num_mpiprocs}``.
This value is taken from the following line, ``mpiprocs_per_machine``, where
you designate the total number of processors per node which is then multiplied
later by the number of requested nodes.

The ``prepend_text`` and ``append_text`` can be used for any default commands
that are needed on the remote server.

Save this yaml file to ``aiida_computer.yaml``. An example is provided here
:download:`computer_example.yaml <../../../orchestrator/oracle/aiida/config_files/computer_example.yaml>`.

Computer Configuration Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   username: "username"
   port: 22
   look_for_keys: true
   key_filename: "/path/to/.ssh/id_ecdsa_key"
   timeout: 60
   allow_agent: true
   proxy_command: ""
   compress: true
   gss_auth: false
   gss_kex: false
   gss_deleg_creds: false
   gss_host: "my_computer"
   load_system_host_keys: true
   key_policy: "RejectPolicy"
   use_login_shell: true
   safe_interval: 10.0

The main values to change here are the ``username`` which is set to your
username, ``key_filename`` which is the path to your ssh security key, and
``gss_host`` which is the name of your computer from the previous file.

Save this information to ``computer_config.yaml``.

Code Example
~~~~~~~~~~~~

.. code-block:: yaml

   label: my_code
   description: Example code for testing
   default_calc_job_plugin: example.plugin
   filepath_executable: /usr/bin/my_code_executable
   computer: my_computer
   prepend_text: ""
   append_text: ""

Here ``label`` should be set to what you want to call the code.

``default_calc_job_plugin`` should be set to vasp.vasp if setting up for VASP
and quantumespresso.pw if setting up for Quantum Espresso.

``computer`` is the name you used previously for the computer label.

Save this to ``aiida_code.yaml``. An example for both
:download:`VASP <../../../orchestrator/oracle/aiida/config_files/code_vasp_std_v6.4.1_example.yaml>`
and :download:`Quantum Espresso <../../../orchestrator/oracle/aiida/config_files/code_qe_example.yaml>`
are provided.

Steps to Use the YAML Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~

As you execute the following steps, there might be some commands which have not
been included in the yaml file. If you do not wish to place a value you can put
``!`` as the response and no value will be set. For example, the default memory
is not included in these instructions and ``!`` should be used so as to not
cause any issues.

1. Run the following commands to set up the computer and code:

   .. code-block:: bash

      verdi computer setup --config aiida_computer.yaml
      verdi computer configure core.ssh my_computer --config computer_config.yaml
      verdi code create core.code.installed --config aiida_code.yaml

2. Verify the setup using:

   .. code-block:: bash

      verdi computer show my_computer
      verdi code show my_code

3. Test computer connection:

   .. code-block:: bash

      verdi computer test my_computer

   If for some reason AiiDA is unable to connect to the server, it is possible that you do not have the desired cluster in your `~/.ssh/known_hosts` file. Attempt to ssh to that cluster to have it added to your file or manually add it to your `./ssh/known_hosts` file.

Customizations
##############

- Replace ``localhost``, ``/scratch/{username}/aiida``, and ``mpirun -np {tot_num_mpiprocs}`` with values specific to your computational environment.
- Update the ``executable_path`` and ``input_plugin`` to match the software you are using.

Installing Pseudopotentials in the AiiDA Database
-------------------------------------------------

This guide provides step-by-step instructions for installing pseudopotentials for **VASP** and **Quantum Espresso** in the **AiiDA** database.

Installing Pseudopotentials for VASP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VASP uses **PAW (Projector Augmented Wave) potentials**, which are stored as `POTCAR` files. To use them in AiiDA, you need to upload the pseudopotentials as a `VaspPotcarData` node.

Steps to Install VASP Pseudopotentials in AiiDA:

1. **Download the Pseudopotentials:**

   - Log in to the **VASP Portal** (https://www.vasp.at/) using your licensed credentials.
   - Download the pseudopotential files (e.g., `POTCAR` files) for the elements you need.

2. **Set Up the AiiDA Environment:**

   - Ensure that your python environment with the Orchestrator and AiiDA is active.

3. **Upload the Pseudopotentials to AiiDA:**

   - Use the `VaspPotcarData` class to upload the pseudopotentials. For example:

   .. code-block:: bash

     verdi data vasp-potcar uploadfamily --path ~/vasp_potentials --name PBE.544 --description "PBE pseudopotentials for VASP v5.4.4"

   - Replace `~/vasp_potentials` with the path to your pseudopotential directory, and `PBE.544` with the appropriate label. This value will be used later when submitting jobs through the Orchestrator.

4. **Verify the Upload:**

   - Check that the pseudopotentials are successfully added to the database:

   .. code-block:: bash

     verdi data vasp-potcar listfamilies

Installing Quantum Espresso Pseudopotentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **aiida-pseudo** plugin simplifies the process of managing pseudopotentials for Quantum Espresso. It provides tools to download, install, and manage pseudopotential families directly in the AiiDA database.

Steps to Install Quantum Espresso Pseudopotentials:

1. **Set Up the AiiDA Environment:**

   - Ensure that your python environment with the Orchestrator and AiiDA is active.

2. **Install the aiida-pseudo Plugin:**

   - If the plugin is not already installed, you can install it using pip:

   .. code-block:: bash

     pip install aiida-pseudo

3. **Download and Install Pseudopotential Families:**

   - The `aiida-pseudo` plugin supports automatic downloading and installation of pseudopotential families from well-known libraries such as **SSSP** (Standard Solid-State Pseudopotentials).
   - Use the following command to download and install a pseudopotential family:

   .. code-block:: bash

     aiida-pseudo install sssp --functional PBE --protocol precision

   - Replace `PBE` with the desired functional (e.g., `LDA`) and `precision` with the desired protocol (e.g., `efficiency` or `precision`).
   - You can see all the options by typing:

   .. code-block:: bash

     aiida-pseudo install --help
     aiida-pseudo install sssp --help

4. **Verify the Installation:**

   - After installation, you can list the available pseudopotential families in the AiiDA database:

   .. code-block:: bash

     aiida-pseudo list

   - This will display the installed families along with their details.

Testing AiiDA code installation
-------------------------------

This document provides example scripts to test your AiiDA installation for **VASP** and **Quantum Espresso (QE)**. These scripts perform basic single-point energy calculations to verify that your AiiDA environment, plugins, and codes are correctly configured.

Quantum Espresso (QE)
~~~~~~~~~~~~~~~~~~~~~

Below is a script to test the QE plugin by running a single-point energy calculation:

.. code-block:: python

   # Import necessary AiiDA modules
   from aiida.engine import submit
   from aiida.orm import load_code, load_group, Dict, StructureData
   from aiida.plugins import DataFactory, WorkflowFactory
   from ase.build import bulk

   # Load the QE code (replace 'my_qe@my_computer' with your configured code label)
   # You can see what your code is called with `verdi code list`
   code = load_code('pw@localhost')
   builder = code.get_builder()

   # Define the structure (simple silicon crystal)
   atoms = bulk('Si', crystalstructure='diamond', a=5.43) * (2,2,2)
   structure = StructureData(ase=atoms)
   builder.structure = structure

   # Define input parameters
   parameters = Dict(dict={
       'CONTROL': {'calculation': 'scf', 'verbosity': 'high'},
       'SYSTEM': {'ecutwfc': 30, 'ecutrho': 240},
       'ELECTRONS': {'conv_thr': 1e-6},
   })
   builder.parameters = parameters

   # Specify pseudopotential family (replace 'SSSP/precision/PBE' with your installed family)
   pseudo_family = load_group('SSSP/1.3/PBEsol/efficiency')
   pseudos = pseudo_family.get_pseudos(structure=structure)
   builder.pseudos = pseudos

   # Specify the Kpoint grid
   KpointsData = DataFactory('core.array.kpoints')
   kpoints = KpointsData()
   kpoints.set_cell_from_structure(structure)
   kpoints.set_kpoints_mesh_from_density(0.15)
   builder.kpoints = kpoints

   # Specify job submission details
   builder.metadata.options.resources = {'num_machines': 1}

   # Run the calculation
   calc = submit(builder)

   # Print the results
   print(f"Job submitted: pk<{calc.pk}>")

VASP
~~~~

Below is a script to test the VASP plugin by running a single-point energy calculation:

.. code-block:: python

   from aiida import load_profile
   from aiida.orm import load_code, load_group, Str, Group, Int
   from aiida.plugins import DataFactory, WorkflowFactory
   from aiida.common.extendeddicts import AttributeDict
   from ase.io import read
   from ase.build import bulk, sort
   from aiida.engine import submit


   #Load default AiiDA profile
   load_profile()

   # Initiate workchain and other inputs
   workchain = WorkflowFactory('vasp.relax')
   inputs = AttributeDict()
   vasp = AttributeDict()
   inputs.vasp = vasp
   settings = AttributeDict()
   dict_data = DataFactory('core.dict')
   kpoints_data = DataFactory('core.array.kpoints')
   Bool = DataFactory('core.bool')

   # Settings
   settings.parser_settings = {
       'include_node': ['energies', 'trajectory'],
       'include_quantity': ['forces', 'stress', 'parameters'],
       'electronic_step_energies': True
   }

   inputs.vasp.settings = dict_data(dict=settings)

   # Set the inputs
   # Code
   code_name = f'vasp_std@localhost'
   inputs.vasp.code = load_code(code_name)

   # Structure information
   StructureData = DataFactory('core.structure')
   atoms = bulk('Si', crystalstructure='diamond', a=5.43)
   structure = StructureData(ase=atoms)
   inputs.structure = structure

   # KPOINTS
   kpoints = kpoints_data()
   kpoints.set_cell_from_structure(structure)
   kpoints.set_kpoints_mesh_from_density(0.15)
   inputs.vasp.kpoints = kpoints

   # INCAR
   inputs.vasp.parameters = dict_data(dict={
       'incar': {
           'algo': 'Conjugate',
           'encut': 500,
           'prec': 'ACCURATE',
           'ediff': 1E-4,
           'ispin': 1,
           'lorbit': 11,
           'ismear': 0,
           'sigma': 0.1,
           'gga': 'PS',
           'kpar': 2,
           'ncore': 14,
           'nelm': 500
       }
   })


   # POTCAR information
   inputs.vasp.potential_family = Str('PBE.544')
   inputs.vasp.potential_mapping = dict_data(dict={'Si': 'Si'})

   # Submission options
   options = AttributeDict()
   options.resources = {'num_machines': 1}
   options.max_wallclock_seconds = 1800
   options.queue_name = 'pdebug'
   options.account = 'bank'
   inputs.vasp.options = dict_data(dict=options)

   # Relax options
   relax = AttributeDict()
   relax.perform = True
   # Select relaxation algorithm
   relax.algo = 'cg'
   # Set force cutoff limit (EDIFFG, but no sign needed)
   relax.force_cutoff = 0.01
   # Turn on relaxation of positions (strictly not needed as the default is on)
   # The three next parameters correspond to the well known ISIF=3 setting
   relax.positions = True
   # Turn on relaxation of the cell shape (defaults to False)
   relax.shape = True
   # Turn on relaxation of the volume (defaults to False)
   relax.volume = True
   # Set maximum number of ionic steps
   relax.steps = 100
   inputs.relax_settings = relax

   # Label
   inputs.vasp.label = Str('Pu2O3 structure optimization')
   inputs.vasp.description = Str('Structure optimization of Pu2O3 without optimized spin states. Will be computed afterwards.')

   inputs.vasp.clean_workdir = False

   # Submit the workchain
   calc = submit(workchain, **inputs)

   print(f'Launched geometry optimization with PK={calc.pk}')

Executing the scripts
~~~~~~~~~~~~~~~~~~~~~

1. Save the script to a file, e.g., ``test_qe.py`` or ``test_vasp.py``.
2. Run the script using ``verdi run``:

   .. code-block:: bash

      verdi run test_qe.py

   or

   .. code-block:: bash

      verdi run test_vasp.py

These scripts are designed to verify that your AiiDA installation, plugins, and codes are correctly configured.

Inspecting calculations
~~~~~~~~~~~~~~~~~~~~~~~

To ensure that the calculations successfully completed, you can take the `PK` value printed and type:

.. code-block:: bash

   verdi process show <PK>

This will show all of the input and output information from the calculation as well as the state of the calculation.
If it is finished, you will find that it says "Finished [0]".

Caching calculations
~~~~~~~~~~~~~~~~~~~~

AiiDA has a very nice caching feature which will simply compare against the database for previous calculations. If there
is a match, it will take the results from the previous simulation and attach them to your current job instead of
submitting the job to be computed. To enable this feature, you must enable it with the following commands.

.. code-block:: bash

   verdi config set caching.enabled_for aiida.calculations:vasp.vasp
   verdi config set --append caching.enabled_for aiida.calculations:quantumespresso.pw

Troubleshooting
~~~~~~~~~~~~~~~

It is possible that the daemons for AiiDA will become stale at some point or cause issues. The Orchestrator is
currently setup to handle some of the cases but every once in a while AiiDA will still have an issue. To attempt
to resolve any issues you may attempt to stop and start the daemon.

.. code-block:: bash

   verdi daemon stop
   verdi daemon start

Inspect the calculations again to see if this runs the jobs currently in the AiiDA queue. If it has not, it is possible
that the jobs are stuck. AiiDA has the following command to attempt to fix the situation.

.. code-block:: bash

   verdi daemon stop
   verdi process repair
   verdi daemon start

This will check on the status on the jobs in the queue to see if there are any issues.

Further documentation
---------------------

For further documentation we recommend that you check the `AiiDA documentation <https://aiida.readthedocs.io/projects/aiida-core/en/stable/>`_.
