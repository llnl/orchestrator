TargetProperty
==============

The primary purpose of the TargetProperty class is to perform analysis to
calculate a property of interest such as the melting point or elastic constant
of a materials. This module can utilize other modules within the orchestrator
to carry out the target property calculations. For example, the
:ref:`target_property_module` can be used to perform melting point
calculations by utilizing :ref:`simulator_module` and :ref:`workflow_module`
modules.

The abstract base class :class:`~.TargetProperty` provides the standard
interface for all of the concrete implementations.

See the full API for the module at :ref:`target_property_module`.

Use Cases
---------

Basic usage
^^^^^^^^^^^
A simple example of how to use the target property class in a standalone application
for melting point calculations can be seen below. The target property class uses
functionalities from several other modules such as Simulator, Workflow and Storage.
The example uses the LAMMPS simulator, local storage and local workflow. The user should
modify .json file located in the test_inputs folder according to needs of a specific
calculation.::

    from target_property import target_property_builder
    from storage import storage_builder
    from simulator import simulator_builder
    from workflow import workflow_builder

    # Choose MeltingPoint for target property type.
    # Target property args are read from the input json file.
    built_target_property = target_property_builder.build(
        type=MeltingPoint, target_property_args)

    # Build local storage. The database path and name are not shown here
    built_storage = storage_builder.build(storage_type='LOCAL')

    # Build local workflow.
    # Workflow root and workflow args are not shown here and read from json file.
    built_workflow = workflow_builder.build(workflow_type='LOCAL')

    # To estimate melting temperature using a single calculation
    results_dict = built_target_property.calculate_property(
        workflow=built_workflow, storage=built_storage)
    value = results_dict['property_value']
    value_std = results_dict['property_std']
    value_calc_ids = results_dict['calc_ids']

    # To estimate melting temperature using multiple calculations with a standard deviation
    results_dict = built_target_property.calculate_with_error(
        n_calc=4, workflow=built_workflow)
    avg_value = results_dict['property_value']
    value_std = results_dict['property_std']
    value_calc_ids = results_dict['calc_ids']

.. _kimrun:

KIMRun
^^^^^^

This class uses `KIM Tests <https://openkim.org/doc/evaluation/kim-tests/>`_
to calculate material properties. This is done by invoking the
`KIM Developer Platform <https://openkim.org/doc/evaluation/kim-developer-platform/>`_
as a Podman (default) or Singularity image.
:func:`~orchestrator.target_property.kimrun.KIMRun.calculate_property`
has one required argument, ``get_test_result_args``. This is a dictionary of
keyword arguments that will be passed to a ``get_test_result`` KIM simplified
query. See
`Querying KIM Content <https://openkim.org/doc/usage/kim-query/#get_test_result>`_
for more info.
``get_test_result_args`` may also be a list of dictionaries if multiple query
results are desired.
The ``model`` argument should be omitted, as it is provided
by the ``potential`` argument to
:func:`~orchestrator.target_property.property_base.TargetProperty.calculate_property`.
In addition to the examples on this page, you can experiment with the web GUI at
`query.openkim.org <https://query.openkim.org/>`__ to learn the syntax for
``get_test_result``.

:class:`~orchestrator.target_property.kimrun.KIMRun` works with
:class:`~orchestrator.potential.potential_base.Potential` objects whose
:func:`~orchestrator.potential.potential_base.Potential.save_potential_to_kimkit`
function saves an archive of a directory that is installable using the KIM API.
Alternatively, the ``potential`` argument may be a string naming a potential
that is already archived in KIMKit.

As with the generic example above, various invocations of
:class:`~orchestrator.target_property.kimrun.KIMRun` using a ``json`` file
can be seen in ``test/target_property/test_inputs``. This module does not
use any Simulator or Storage.

Additionally, ``KIMRun`` may be invoked directly. The below example will
be used to demonstrate details of usage::

    from orchestrator.target_property.kimrun import KIMRun

    my_kimrun = KIMRun()

    print(
        my_kimrun.calculate_property(
            potential = "Sim_LAMMPS_MEAM_Lenosky_2017_W__SM_631352869360_000",
            get_test_result_args = [
                    {
                        "test": ["VacancyFormationEnergyRelaxationVolume_bcc_W__TE_197190379294"],
                        "prop": ["monovacancy-neutral-relaxed-formation-potential-energy-crystal-npt"],
                        "keys": ["relaxed-formation-potential-energy"],
                        "units": ["eV"]
                    },
                    {
                        "test": ["VacancyFormationEnergyRelaxationVolume_bcc_W__TE_197190379294"],
                        "prop": ["monovacancy-neutral-relaxation-volume-crystal-npt"],
                        "keys": ["relaxation-volume"],
                        "units": ["angstrom^3"]
                    },
                    {
                        "test": ["SurfaceEnergyCubicCrystalBrokenBondFit_bcc_W__TE_378149060769"],
                        "prop": ["surface-energy-cubic-crystal-npt"],
                        "keys": ["miller-indices","surface-energy"],
                        "units": [None,"eV/angstrom^2"]
                    }
                ]
            )
        )

When specifying the ``test`` key(s), the version and the prefix are optional.
The value ``VacancyFormationEnergyRelaxationVolume_bcc_W__TE_197190379294`` in the example above
automatically runs the latest version of that test found in KIMKit.
``VacancyFormationEnergyRelaxationVolume_bcc_W__TE_197190379294_001``, ``TE_197190379294``, or
``TE_197190379294_001`` would also be valid values that run the same test.

Any Tests requested to run by the ``test`` key(s) in
``get_test_result_args``, as well as their Test Drivers
and their upstream dependencies (i.e. almost any KIM Test depends on the
LatticeConstant test for the corresponding material) must be manually archived in
KIMKit ahead of time. A full listing of OpenKIM tests can be found
`here <https://openkim.org/browse/tests/by-species>`__.

To demonstrate, consider the Test
``VacancyFormationEnergyRelaxationVolume_bcc_W__TE_197190379294_001``.
The source files are available at the bottom of
`its page on openkim.org <https://openkim.org/id/VacancyFormationEnergyRelaxationVolume_bcc_W__TE_197190379294_001>`_
(use the "Files" link at the top right to jump to the bottom).
To make it possible to run the Test, you must download the ``.txz`` archives
for the Test itself
(``VacancyFormationEnergyRelaxationVolume_bcc_W__TE_197190379294_001.txz``), and its Driver
(``VacancyFormationEnergyRelaxationVolume__TD_647413317626_001.txz``) from the linked page.
Additionally note that the listing of source files contains a file named
``dependencies.edn`` containing the text ``[ "TE_155104699590" ]``.
This indicates that this Test depends on the result of another Test, which
must also be installed. By searching for the Short KIM ID in the full
list of KIM Tests, we can see that it is the Lattice Constant test,
`LatticeConstantCubicEnergy_bcc_W__TE_155104699590_007 <https://openkim.org/id/LatticeConstantCubicEnergy_bcc_W__TE_155104699590_007>`_.
Even if you do not need any properties computed by that Test, it must
be installed to run the VacancyFormationEnergyRelaxationVolume test, so you would need to download
the ``.txz`` files for the Test and its Driver in the same way.

Once you have the ``.txz`` files downloaded, you must add them to KIMKit.
For example, assuming you have the 4 ``.txz`` files in question in the
current working directory, the following Python script would add them to
KIMKit. Drivers must be imported before corresponding Tests if the Driver
is not already in KIMKit.::

    import tarfile, kimkit
    kimkit.models.import_item(tarfile.open(
        "LatticeConstantCubicEnergy__TD_475411767977_007.txz"))
    kimkit.models.import_item(tarfile.open(
        "LatticeConstantCubicEnergy_bcc_W__TE_155104699590_007.txz"))
    kimkit.models.import_item(tarfile.open(
        "VacancyFormationEnergyRelaxationVolume_bcc_W__TE_197190379294_001.txz"))
    kimkit.models.import_item(tarfile.open(
        "VacancyFormationEnergyRelaxationVolume__TD_647413317626_001.txz"))

As is evident from the Test names, the choice of Test determines the crystal
structure and species. All results in the example above are for BCC tungsten.

As described in
`Querying KIM Content <https://openkim.org/doc/usage/kim-query/#get_test_result>`_,
the values returned by the ``get_test_result`` queries are the
`KIM Property keys <https://openkim.org/doc/schema/properties-framework>`_
that are requested in the query. The full list of KIM Property Definitions is
`here <https://openkim.org/properties>`__. In the example above, the
``prop`` keys of ``get_test_result_args`` indicate that the properties that
will be queried for are
`monovacancy-neutral-relaxed-formation-potential-energy-crystal-npt <https://openkim.org/properties/show/2015-07-28/staff@noreply.openkim.org/monovacancy-neutral-relaxed-formation-potential-energy-crystal-npt>`_
and so on. The ``keys`` keys of ``get_test_result_args`` indicate which keys will be
extracted from those properties.

By default, The ``property_value`` returned as part of ``results_dict``
is a a doubly nested list. First index is over each dictionary in ``get_test_result_args``.
Second index is over the number of times the queried Test returned the queried property
(e.g. multiple surface energies). Third index is over the keys requested within the property.
The values may be arrays of arbitrary dimension themselves.
If :func:`~orchestrator.target_property.kimrun.KIMRun.calculate_property` is passed the argument
``flatten=True``, then ``property_value`` is flattened into a 1-D array (this is done for testing,
as the current test suite can't handle nested lists with inhomogeneous dimensions). Currently,
uncertainty is not supported.

To demonstrate the organization of the (non-flattened) output, see the output of the Python example
above, with indentation and comments added for clarity::

    [
        # Result of the first query.
        # The VacancyFormationEnergyRelaxationVolume test returns a single
        # instance of the property
        # "monovacancy-neutral-relaxed-formation-potential-energy-crystal-npt",
        # Therefore there is only one element corresponding to returned
        # property instances. Because we only requested one key from this property
        # ("relaxed-formation-potential-energy"), there is only one element
        # in the inner list corresponding to requested keys.
        [[3.19968717282485]],
        # Result of the second query. It is analogous to above, except for
        # vacancy relaxation volume. Even though the two quantities were
        # computed by the same test, they are written to different
        # properties (in this case,
        # "monovacancy-neutral-relaxation-volume-crystal-npt"). Therefore we must
        # query for them separately.
        [[5.640063822313584]],
        # Result of the third query. Here, the
        # SurfaceEnergyCubicCrystalBrokenBondFit test returns 4 instances of the
        # "surface-energy-cubic-crystal-npt" property, corresponding to different
        # Miller indices, so there are 4 elements in the list corresponding to
        # returned property instances. Within each of those lists, there are two
        # elements corresponding to the keys we requested -- "miller-indices" and
        # "surface-energy"
        [
            [[1, 1, 1], 0.2278375012400934],
            [[1, 0, 0], 0.214109530467288],
            [[1, 2, 1], 0.2130027972015625],
            [[1, 1, 0], 0.1817853309873278]
        ]
    ]

.. _calculate_lammps_thermo:

CalculateLammpsThermo
^^^^^^^^^^^^^^^^^^^^^

This class calculates various properties such as density, stress, temperature as a function of time
through thermo function in LAMMPS simulations, extract the user defined properties from the log file
and generate a plot for each property. The thermo properties of interest need to be defined via
thermo_style in the LAMMPS template input files (either lammps_main_npt.lmp or lammps_main_nvt.lmp depending
on the required ensemble to investigate a property). The ``lammps_resources`` folder with all the required
force_field and initial input structure files should be placed in a user-defined location
(such as ``test/target_property/templates``) and this location needs to be defined in the ``json`` file.
Forcefield and initial structure file names need to be defined in the ``json`` file too. The user can select
the properties to calculate/plot using ``property_selections`` in the ``json`` file. The corresponding property
units can be also defined using ``property_units`` (to be used in the plots, can be left blank if they are not needed).


Inheritance Graph
-----------------

.. inheritance-diagram::
   orchestrator.target_property.elastic_constants
   orchestrator.target_property.factory
   orchestrator.target_property.kimrun
   orchestrator.target_property.melting_point
   :parts: 3
