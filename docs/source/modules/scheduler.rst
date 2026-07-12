Scheduler
=========

This module handles the submission and retrieval of simulations to either a
local computer or HPC resources, managing the file structure of the simulations,
and retains information on the location of job files.

To see a list of currently implemented job schedulers, see the full API for the
module at :ref:`scheduler_module`. The abstract base class
:class:`~orchestrator.scheduler.scheduler_base.Scheduler` provides the standard
interface for all of the concrete implementations. We also provide an abstract
base class for HPC schedulers:
:class:`~orchestrator.scheduler.scheduler_base.HPCScheduler`

The simplest implementation provides an interface with the local command line,
but interface with job schedulers or other more sophisticated tools, such as
`Merlin <https://merlin.readthedocs.io/en/latest/>`_ is also possible.

Use Cases
---------

:class:`~orchestrator.scheduler.local.LocalScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation for running jobs locally on a personal computer or an interactive job session
:class:`~orchestrator.scheduler.local.LocalScheduler`.  Note that
all of the modules define a default scheduler which is used if a scheduler is
needed but not supplied. This default is an instance of
:class:`~orchestrator.scheduler.local.LocalScheduler` with the root directory set to
the module's name.

:class:`~orchestrator.scheduler.slurm.SlurmScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A default script template for the slurm batch file is provided, but the user
can define their own and provide it's path via the ``default_template``
keyword in the ``scheduler_args`` dictionary passed to the Scheduler constructor.
Also note that if synchronous (blocking) behavior is desired, this can be
toggled with the ``synchronous`` keyword in the ``job_details`` dict provided
to :meth:`~orchestrator.scheduler.scheduler_base.Scheduler.submit_job`.

This scheduler includes the option to submit to a Slurm machine that can be
accessed via SSH from the current machine. To use this functionality,
``remote_machine`` must be set at initialization.

The ``job_details`` dict also hosts any modifications to the batch job desired,
with the default batch template defining all possible options:

.. code-block:: bash

   #!/bin/bash
   #SBATCH -N <NODES>
   #SBATCH -p <QUEUE>
   #SBATCH -A <ACCOUNT>
   #SBATCH -t <WALLTIME>
   <EXTRA_HEADER>

   <PREAMBLE>

   <COMMAND>

   <POSTAMBLE>

In addition to these keywords (which should be set as lowercase, i.e.
'preamble'), default queue, account, walltime, and node parameters can be set.
Lastly, the frequency of calls to squeue are set by ``wait_freq``, which has a
default of 60 seconds.

The scheduler is designed to have flexibility for heterogenous use cases.
To this end, default parameters can be set by the user when constructing the
Scheduler via the ``scheduler_args`` dict, but many of these parameters can be
overridden for any specific job by providing them in the ``job_details`` dict
of the :meth:`~orchestrator.scheduler.scheduler_base.Scheduler.submit_job`
function.

When using an asynchronous scheduler, it is important to use a blocking function
to ensure necessary calculations are done before proceding. An example is
:class:`~orchestrator.scheduler.slurm.SlurmScheduler`'s
:meth:`~orchestrator.scheduler.slrum.SlurmScheduler.block_until_completed` method,
which would be called right before the outcomes of any set of calculations
are needed by subsequent functions or modules.

:class:`~orchestrator.scheduler.lsf.LSFScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~orchestrator.scheduler.lsf.LSFScheduler` is provided as a mirror to
:class:`~orchestrator.scheduler.slurm.SlurmScheduler` that enables the use of
IBM's LSF scheduler. Much of the previous description applies to this
scheduler as well. The differences will be highlighted below. This scheduler
includes the option to submit to an LSF machine that can be accessed via SSH
from the current machine. To use this functionality, ``remote_machine`` must be
set at initialization. Additional paths for the LSF profile or submission
executable may also need to be provided.

:class:`~orchestrator.scheduler.flux.FluxScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~orchestrator.scheduler.flux.FluxScheduler` is provided as a mirror to
:class:`~orchestrator.scheduler.slurm.SlurmScheduler` that enables the use of
the flux scheduler. Much of the previous description applies to this
scheduler as well. This scheduler includes the option to submit to a flux
machine that can be accessed via SSH from the current machine. To use this
functionality, ``remote_machine`` must be set at initialization. Another
feature of flux is the ability to provision jobs within an existing
allocation. To change the command that is used to submit a job (default is
``batch``), the ``default_submit_command`` can be modified at initialization
(i.e. to ``run``) or overridden in ``job_details`` with the ``submit_command``
key.

:class:`~orchestrator.scheduler.aiida.AiidaScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An interface for the AiiDA framework has been implemented as a Scheduler for
the Orchestrator. This must be combined with any of the oracles found in
:ref:`aiida_oracle` API documentation. As
:class:`~orchestrator.scheduler.aiida.AiidaScheduler` inherits from
:class:`~orchestrator.scheduler.scheduler_base.HPCScheduler`, all of the variables
related to job submission are the same. These values can be seen at the
:class:`~orchestrator.scheduler.scheduler_base.HPCScheduler` API documentation.

Restarting Jobs
---------------

All scheduler classes provide a :meth:`~.scheduler_base.HPCScheduler.restart_job`
method to restart a previously submitted job with the same or updated
parameters. This is useful for continuing failed jobs, running additional
timesteps, or modifying job parameters while preserving input files.

The restart process:

1. Retrieves the original job details using the provided ``calc_id``
2. Creates a new run directory following the standard scheduler path structure
3. Copies input files from the original job directory to the new directory
4. Automatically updates any file paths that reference the old directory
5. Submits a new job with the original or updated parameters

By default, all files are copied except output files (``*.log``, ``*.out``,
``*.err``, ``job_done``, ``slurm*``, ``batch*``, ``flux*``). You can customize
which files are copied using the ``copy_pattern`` and ``exclude_pattern``
parameters.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   # Restart a job with the same parameters
   new_calc_id = scheduler.restart_job(calc_id=123)

   # Restart with an updated command
   new_calc_id = scheduler.restart_job(calc_id=123, command="lmp -in restart.in")

   # Restart with updated job details (e.g., longer walltime)
   new_calc_id = scheduler.restart_job(
       calc_id=123,
       job_details={"walltime": "24:00:00", "nodes": 4}
   )

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

   # Copy only specific files
   new_calc_id = scheduler.restart_job(
       calc_id=123,
       copy_pattern="*.data",  # Only copy .data files
       exclude_pattern="temp_*"  # Exclude temporary files
   )

   # Copy multiple file types
   new_calc_id = scheduler.restart_job(
       calc_id=123,
       copy_pattern=["*.in", "*.data", "*.restart"],
       exclude_pattern=["*.dump", "backup_*"]
   )

Notes on ``restart_job``
~~~~~~~~~~~~~~~~~~~~~~~~

- If the original job is still running, a warning is logged but the restart
  proceeds
- The command is retrieved from the original job if not provided
- Job details from the original job are merged with any new details provided
- The new job receives a new ``calc_id`` and runs in a separate directory


Remote SSH Capabilities
-----------------------

All HPC scheduler classes (:class:`~orchestrator.scheduler.slurm.SlurmScheduler`,
:class:`~orchestrator.scheduler.lsf.LSFScheduler`,
and :class:`~orchestrator.scheduler.flux.FluxScheduler`) support submitting jobs to
remote machines via SSH using the common ``remote_machine`` parameter defined
in the :class:`~orchestrator.scheduler.scheduler_base.HPCScheduler` parent class:

.. code-block:: python

   # Example using SlurmScheduler with a remote machine
   scheduler = SlurmScheduler(remote_machine="cluster.example.com", queue="batch", account="myaccount")

   # Example using LSFScheduler with a remote machine
   scheduler = LSFScheduler(remote_machine="lsf-cluster.example.com", queue="batch", account="myaccount")

   # Example using FluxScheduler with a remote machine
   scheduler = FluxScheduler(remote_machine="flux-cluster.example.com", queue="batch", account="myaccount")

When the ``remote_machine`` parameter is provided, the schedulers will
automatically handle SSH connections to the remote machine for job submission
and status checks. Note that the selected scheduler type should match the
scheduler on the remote_machine, which may or may not be different than the
scheduler system on which Orchestrator is being executed.

Slurm and LSF Differences
-------------------------

While Slurm and LSF perform the same function, there are subtle differences in
keyword selection and use cases. The LLNL LC reference pages for
`Slurm <https://hpc.llnl.gov/banks-jobs/running-jobs/slurm-user-manual>`_ and
`LSF <https://hpc.llnl.gov/banks-jobs/running-jobs/lsf-user-manual>`_ are good
places to start for details on these schedulers. Differences in flags used for
specifying the jobs can also be found `in the chart here
<https://hpc.llnl.gov/banks-jobs/running-jobs/\\
slurm-srun-versus-ibm-csm-jsrun>`_.

Full documentation for `Slurm sbatch <https://slurm.schedmd.com/sbatch.html>`_
and `LSF bsub <https://www.ibm.com/docs/en/spectrum-lsf/\\
10.1.0?topic=bsub-options>`_ can be found at the provided links.

Inheritance Graph
-----------------

.. inheritance-diagram::
   orchestrator.scheduler.aiida
   orchestrator.scheduler.factory
   orchestrator.scheduler.flux
   orchestrator.scheduler.local
   orchestrator.scheduler.lsf
   orchestrator.scheduler.slurm
   :parts: 3
