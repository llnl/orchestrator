Miscellaneous Utilities
=======================

See :ref:`util_api` for full API documentation. Beyond the utilities previously covered in this section, there are a number of other utility files which ensure smooth and convenient usage of the Orchestrator. We provide an overview of these components below.

Module Setup
------------

Functions provided in the ``setup_input`` file wrap around builders to directly instantiate and return a module instance, with basic input validation. :meth:`~orchestrator.utils.setup_input.setup_orch_modules` can be used to instantiate multiple modules at once from a given json file.

Templates
---------

Some workflows require the generation of input files which follow a specific structure and may share many similarities. In these cases, the :class:`~orchestrator.utils.templates.Templates` class and its :meth:`~orchestrator.utils.templates.Templates.replace` funciton can be used.

Data Standard
-------------

Orchestrator uses ASE Atoms objects as its internal configuration data structure. Key data is stored in the Atoms arrays and info dict, but ASE does not enforce any naming convention. To ensure consistency throughout the Orchestrator code suite, we define constant :ref:`data_keys` which should be used for both setting and accessing relevant data. Keys exist for quantities such as energies and forces, metadata, and selection/weight masks.

New structure IO
----------------

While Orchestrator will generally handle parsing of its own data, there are instances (typically around new data ingestion) where external data needs to be used. For converting xyz files into the internal Atoms representation, the method :meth:`~orchestrator.utils.input_output.ase_glob_read` is available. Note that this method will not set keys according to the data standard (discussed above) since the xyz file format does not enforce any naming conventions. Thus for users adding data for storage, you will need to manually the keys as appropriate prior to storage with i.e. :meth:`~orchestrator.storage.storage_base.Storage.new_dataset`. A basic example is shown below where the xyz file stores forces as 'force' and energies as 'Energy':

.. code-block:: python

    from orchestrator.utils.input_output import ase_glob_read
    from orchestrator.utils.data_standard import ENERGY_KEY, FORCES_KEY

    new_configs = ase_glob_read('./path/to/xyzfiles')
    for config in new_configs:
        config.set_array(FORCES_KEY, config.arrays['force'])
        config.info[ENERGY_KEY] = config.info['Energy']
        # optionally delete the previous named entries
        del config.arrays['force']
        del config.info['Energy']

Structure Analysis and Manipulation
-----------------------------------

This utility module contains a mixture of functionality that is primarily used by the :class:`~.Augmentor` but may be more broadly useful as well as methods for analyzing and visualizing structural information. These tools can help users better understand underlying structural data in datasets for training IAPs.

The main analysis method is:

* :meth:`~orchestrator.utils.structure_analysis_and_manipulation_tools.analyze_structures`, which can operate on either a single structure or a set of structures

The outputs of these analysis methods can be used on their own, or passed to the plotting functions:

* :meth:`~orchestrator.utils.structure_analysis_and_manipulation_tools.plot_structure_analysis_results`, which shows composite information from one or more structures
* :meth:`~orchestrator.utils.structure_analysis_and_manipulation_tools.plot_structure_analysis_comparisons`, which compares structural information across different configurations in a provided set.

.. _struct_analysis_example:

Structure Analysis and Plotting Example Usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.utils.structure_analysis_and_manipulation_tools import (
        analyze_structures,
        plot_structure_analysis_results,
        plot_structure_analysis_comparisons
    )
    from orchestrator.augmentor import Augmentor
    import numpy as np
    from ase.build import bulk

    augmentor = Augmentor()

    element = 'C'
    lat_type = 'diamond'
    supercell_num = 3
    lat = 3.57
    num_structs = 5
    compression_range = (.8, .95)
    # use the augmentor's generate_shaken_boxes method to make structures
    shaken_configs = augmentor.generate_shaken_boxes(
        element,
        lat_type,
        lat,
        compression_range,
        supercell_num,
        num_structs,
        temperature=6000,
        debeye_temp=800,
        mass_weighted=True,
        seed=42,
        cubic=True,
    )
    shaken_labels = [
        f'{x*100:.3f}% compress ' for x in np.linspace(
            compression_range[0], compression_range[1], num_structs)
    ]

    # also generate a pristine lattice to compare to
    pristine = bulk(element, lat_type, a=lat,
                    cubic=True) * (supercell_num, supercell_num, supercell_num)
    pristine_label = 'pristine'

    # do analysis on structures (NN distances, RDF)
    # Use the unified analyze_structures function for both single and multiple structures
    combined_shaken, shaken_individual = analyze_structures(
        shaken_configs,
        labels=shaken_labels,
    )
    pristine_results = analyze_structures(pristine, label=pristine_label)

    # plot the outputs
    plot_structure_analysis_results(combined_shaken, '.')
    all_individual_results = shaken_individual + [pristine_results]
    plot_structure_analysis_comparisons(all_individual_results, '.')
