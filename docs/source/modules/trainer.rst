Trainer
=======

See the full API for the module at :ref:`trainer_module`. The abstract base
class :class:`~orchestrator.trainer.trainer_base.Trainer` provides the standard
interface for all of the concrete implementations.

This module manages both the organization of data (to train, verify, or perform
analysis) as well as the training of IAP models using said data. Trainers may
be specific to a certain model or class of model, or may be universal.

List of Available Trainers
--------------------------

There are currently several primary groups of trainers available:
the :class:`~orchestrator.trainer.fitsnap.FitSnapTrainer`, which is used with the
:class:`~orchestrator.potential.fitsnap.FitSnapPotential` potential,
the :class:`~orchestrator.trainer.chimes.ChIMESTrainer`, which is used with the
:class:`~orchestrator.potential.chimes.ChIMESPotential` potential, and
the :class:`~orchestrator.potential.trainer.kliff.kliff_dunn_trainer.DUNNTrainer` and
the :class:`~orchestrator.potential.trainer.kliff.kliff_parametric_trainer.ParametricModelTrainer`,
which are used with the :class:`~orchestrator.potential.dnn.KliffBPPotential` and the
generic :class:`~orchestrator.potential.kim.KIMPotential` potentials.

Training Modes
--------------

The trainer module defines two training modes: in memory and distributed. These
are executed by the :meth:`~orchestrator.trainer.trainer_base.Trainer.train`
and :meth:`~orchestrator.trainer.trainer_base.Trainer.submit_train` methods,
respectively. The distributed approach saves the current potential to disk and
then creates a job script which restarts the orchestrator on a compute node to
perform the training, saving the trained potential to disk at the end. By
default, both methods will attempt to upload the trained potential to kimkit,
iterating the version number if working off of a current kimkit potential, or
creating a new kimID otherwise. The
:meth:`~orchestrator.test.unit_testers.trainer_potential_scheduler_test` serves
as an example of how to use trainers, though they disable the kimkit upload and
pull data from test databases.

While the in-memory approach can be used for small tests, it is generally
advised to use the distributed version (or use the
:meth:`~orchestrator.trainer.trainer_base.Trainer.train` method within a
compute node). Note also that the
:meth:`~orchestrator.trainer.trainer_base.Trainer.submit_train` method
requires the writing and reading of a model to disk. Some of the
:class:`~orchestrator.potential.kim.KIMPotential` architectures do not support
this writing and therefore can only use the in-memory training version.

Hyperparameter and Settings Control
-----------------------------------

The :meth:`~orchestrator.trainer.trainer_base.Trainer.train` and
:meth:`~orchestrator.trainer.trainer_base.Trainer.submit_train` methods require
four things: a string used to define the name of the directory path that the
:class:`~orchestrator.scheduler.Scheduler` will create, the related
:class:`~orchestrator.potential.potential_base.Potential` class object, a
:class:`~orchestrator.storage.storage_base.Storage` class object, and a list
of the dataset IDs to pull from the storage object. Additionally, the
:meth:`~orchestrator.trainer.trainer_base.Trainer.submit_train` requires
a :class:`~orchestrator.scheduler.Scheduler` object and a dictionary of
job settings (e.g. walltime, # of nodes).

The hyperparameters that control the form of the potential or its descriptors
are controlled by the relevant
:class:`~orchestrator.potential.potential_base.Potential` object. The
:meth:`~orchestrator.trainer.trainer_base.Trainer.train` and
:meth:`~orchestrator.trainer.trainer_base.Trainer.submit_train` methods
will accept additional optional arguments for training specific
hyperparameters, such as data weighting, # of epochs, etc.

See the :ref:`potential_rst` page for potential initialization details,
with details for specific potential types located at
:ref:`specific_potential_details`.

See additional details for the Kliff trainers at
:ref:`kliff_trainer.rst`.


Optional Atomic Data Weighting
------------------------------

The :class:`~orchestrator.trainer.trainer_base.Trainer` currently available
all support atomic-level granularity weighting of force data.

For :class:`~orchestrator.trainer.fitsnap.FitSnapTrainer`, the per_atom_weights
flag can be supplied with data in list/np.ndarray
form, a filepath to a np.loadtxt() compatible file, or a True boolean.
A True boolean will attempt to locate data within the storage object
datasets under the label atomic_weights. FitSNAP Neural Networks use a data
loader structure that does not support individual atom weighting at the moment.
All linear or quadratic type SNAP type models are compatible with per atom weighting.

For :class:`~orchestrator.potential.trainer.kliff.kliff_dunn_trainer.DUNNTrainer` and
:class:`~orchestrator.potential.trainer.kliff.kliff_parametric_trainer.ParametricModelTrainer`
the per_atom_weights currently only accepts a True/False boolean, and
will attempt to locate the data within the storage object datasets under
the label atomic_weights.

Additional Details
------------------

Additional details on the kliff trainers are on these pages:

.. toctree::
   :maxdepth: 1

   kliff_trainer

Inheritance Graph
-----------------

.. inheritance-diagram::
   orchestrator.trainer.chimes
   orchestrator.trainer.factory
   orchestrator.trainer.fitsnap
   orchestrator.trainer.kliff.kliff
   orchestrator.trainer.kliff.kliff_dunn_trainer
   orchestrator.trainer.kliff.kliff_parametric_trainer
   :parts: 3
