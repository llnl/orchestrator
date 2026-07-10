.. _workflow_module:

Workflow Module
================

Abstract Base Classes
---------------------

.. autoclass:: orchestrator.workflow.workflow_base.Workflow
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.workflow.workflow_base.HPCWorkflow
   :members:
   :undoc-members:
   :show-inheritance:

Concrete Implementations
------------------------

Local
^^^^^

.. automodule:: orchestrator.workflow.local
   :members:
   :undoc-members:
   :show-inheritance:

Slurm (sbatch)
^^^^^^^^^^^^^^

.. autoclass:: orchestrator.workflow.slurm.SlurmWF
   :members:
   :undoc-members:
   :show-inheritance:

LSF (bsub)
^^^^^^^^^^

.. autoclass:: orchestrator.workflow.lsf.LSFWF
   :members:
   :undoc-members:
   :show-inheritance:

Flux (flux batch)
^^^^^^^^^^^^^^^^^

.. autoclass:: orchestrator.workflow.flux.FluxWF
   :members:
   :undoc-members:
   :show-inheritance:

AiiDA
^^^^^

.. autoclass:: orchestrator.workflow.aiida.AiidaWF
   :members:
   :undoc-members:
   :show-inheritance:

Workflow Builder
----------------

.. automodule:: orchestrator.workflow.factory
   :members:
   :undoc-members:
   :show-inheritance:

Job Status
==========

.. autoclass:: orchestrator.workflow.workflow_base.JobStatus
   :members:
   :undoc-members:
   :show-inheritance:
