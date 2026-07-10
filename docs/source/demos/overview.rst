.. _example_intro:

Examples Overview
=================

In this section, we demonstrate the capabilities and usage of the Orchestrator
in multiple contexts, both as standalone modules and in combination for more
complex workflows. To make these examples accessible, they are provided as
Jupyter notebooks with in-line comments that explain conventions and typical
behaviors. In addition to these examples, you can refer to the unit test
functions found in
``orchestrator/test/<MODULE_NAME>/<MODULE_NAME>_unit_testers.py`` files for
further insights into module usage. The range of these tests are enumerated in
the :ref:`testing` section of the docs.

Jupyter Examples
================

The following notebooks are designed to be self-contained and can be explored
in any order. Depending on the example, you may need to perform some initial
setup, such as adding configurations to the storage module or installing a KIM
potential from openkim.org.

Pruning a Dataset with the Augmentor
------------------------------------

This notebook walks through the process of computing
:class:`~.QUESTSDescriptor` descriptors on a dataset, followed by pruning the
dataset based on these descriptors using the :class:`~.Augmentor` module.
After pruning, you will see how to train models on both the original and
reduced datasets using the :class:`~.FitSnapPotential` and
:class:`~.FitSnapTrainer`, allowing you to compare their performance.

:download:`Download Notebook <../../../examples/augmentor_example.ipynb>`

Train a Potential and Deploy it in a MD Simulation
--------------------------------------------------

Here, you will learn how to train a :class:`~.FitSnapPotential` IAP from
scratch. The example continues by showing how to save and install the trained
model to the KIM API, and then demonstrates how to run a molecular dynamics
simulation with the new potential using a :class:`~.LAMMPSSimulator`.

:download:`Download Notebook <../../../examples/LAMMPS_SNAP_example.ipynb>`

Use a Potential to Evaluate a Material Property
-----------------------------------------------

In this example, you will see how to initialize a :class:`~.KIMPotential` and
use it to evaluate material properties. The notebook guides you through
generating a cold curve using both the potential's
:meth:`~.Potential.evaluate` method and the :class:`~.KIMRun` target property,
illustrating different approaches to property evaluation.

:download:`Download Notebook <../../../examples/potential_and_kimrun_example.ipynb>`

Use Score Modules to Analyze a Dataset
--------------------------------------

This notebook demonstrates how to load a dataset—either from storage or from
an xyz file—and then compute descriptors for further analysis. You will then
see how to assess the dataset using various scoring modules, including
:class:`~.QUESTSEfficiencyScore`, :class:`~.QUESTSDiversityScore`, and
:class:`~.QUESTSDeltaEntropyScore` calculations.

:download:`Download Notebook <../../../examples/quests_score_example.ipynb>`

Refit an Empirical Potential
----------------------------

Starting from an existing empirical potential, this example guides you through evaluating and inspecting its parameters. You will then set up a :class:`~.ParametricModelTrainer` to retrain the model, and finally re-evaluate and save the updated potential, completing the refitting workflow.

:download:`Download Notebook <../../../examples/tersoff_colabfit_fitting_example.ipynb>`

.. _dimer_demo:

Generate Dimer Configurations for PES Evaluation
------------------------------------------------

This example demonstrates the usage of :meth:`~.generate_dimer_configs` to generate a set of dimers for molecular systems, followed by :meth:`~.evaluate_dimer_configs_simulator`, which is used to evaluate the PES of these structures. The example constructs a PES map of a water and ethyl acetate dimer.

:download:`Download Notebook <../../../examples/dimer_demo.ipynb>`
