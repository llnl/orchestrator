.. _scheduler_module:

Scheduler Module
================

Abstract Base Classes
---------------------

.. autoclass:: orchestrator.scheduler.scheduler_base.Scheduler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.scheduler.scheduler_base.HPCScheduler
   :members:
   :undoc-members:
   :show-inheritance:

Concrete Implementations
------------------------

Local
^^^^^

.. automodule:: orchestrator.scheduler.local
   :members:
   :undoc-members:
   :show-inheritance:

Slurm (sbatch)
^^^^^^^^^^^^^^

.. autoclass:: orchestrator.scheduler.slurm.SlurmScheduler
   :members:
   :undoc-members:
   :show-inheritance:

LSF (bsub)
^^^^^^^^^^

.. autoclass:: orchestrator.scheduler.lsf.LSFScheduler
   :members:
   :undoc-members:
   :show-inheritance:

Flux (flux batch)
^^^^^^^^^^^^^^^^^

.. autoclass:: orchestrator.scheduler.flux.FluxScheduler
   :members:
   :undoc-members:
   :show-inheritance:

AiiDA
^^^^^

.. autoclass:: orchestrator.scheduler.aiida.AiidaScheduler
   :members:
   :undoc-members:
   :show-inheritance:

Scheduler Builder
-----------------

.. automodule:: orchestrator.scheduler.factory
   :members:
   :undoc-members:
   :show-inheritance:

Job Status
==========

.. autoclass:: orchestrator.scheduler.scheduler_base.JobStatus
   :members:
   :undoc-members:
   :show-inheritance:
