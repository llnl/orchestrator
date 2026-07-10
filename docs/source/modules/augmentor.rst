Augmentor
=========

The Augmentor module encapsulates a collection of methods for dataset augmentation and pruning, designed primarily for atomistic data. It provides robust methods to identify, extract, and curate atomic environments from large datasets, leveraging advanced scoring and selection strategies. The module is engineered to support active learning loops, dataset reduction via farthest point sampling (FPS), and redundancy pruning. Given the cost of operating on large datasets, many methods are implemented to utilize parallel computing resources.

Key features include:

#. Pruning and Reduction: Reduce dataset size by removing redundant or less informative environments, using iterative FPS algorithms and efficiency-based scoring. Three options currently exist to facilitate pruning:

   * The simplest is :meth:`~.simple_prune_dataset`, which trims a dataset based on a user-defined cutoff value of a scoring metric (such as uncertainty or importance) or a percentage of the full set size. This method is easy to apply, but it requires knowledge about the underlying data to be used to greatest effect.

   * For more advanced pruning, :meth:`~.iterative_fps_prune` implements an iterative farthest point sampling (FPS) algorithm, using information content metrics to remove human bias from the process. This can be thought of as an automated version of the simple pruning method.

   * To address scalability challenges with the iterative method, we recommend users leverage the :meth:`~.chunked_iterative_fps_prune` method, which speeds up the pruning process by dividing the dataset into chunks, pruning each in parallel, and recombining the resultant subsets in a hierarchical fashion. While this approach is the most scalable, it introduces minor approximations in the final selection, depending on how the data is partitioned.

#. Novelty Detection: Identify atomic environments in candidate configurations that are novel with respect to a reference dataset, using customizable scoring modules. This approach is especially powerful in active learning scenarios, where the goal is to expand a dataset with new information. While this ensures broader coverage and improved generalizability for downstream models, it can be computationally demanding for large datasets and is sensitive to the choice of scoring function. This functionality is mainly encapsulated by the :meth:`~.identify_novel_environments` method, which is also used in the composite :meth:`~.score_and_extract_subcells` method which will also compute the score values for data in an integrated and automated way, reducing human intervention needs.

#. Subcell Extraction: Beyond identifying novel environments, it is critical to be able to extract desired local atomic environments (subcells) from larger configurations, which enables downstream calculations of ground truth data (i.e. DFT energies and forces). The Augmentor's :meth:`~.extract_and_tag_subcells` method encodes this funcitonality.

#. Structure Generation: In constructing training sets for IAPs it is often useful or necessary to include examples of "close contacts" between atoms as this is a part of configuration space that may not normally be sampled during trajectories used to seed training sets. :meth:`~.generate_single_atom_perturbations` and :meth:`~.generate_shaken_boxes` provide two approaches to generating such structures. They achieve this through moving a single atom in a supercell closer to its nearest neighbor (effectively embedding a dimer in a bulk lattice) or straining a pristine supercell and applying thermal noise to all atoms, respectively. One may wish to analyze these structures. See the example for using Orchestrator tools to do this: :ref:`struct_analysis_example`. When dealing with molecular systems, generating structures to probe the PES of a system can provide valuable insight into an IAPs suitability for that system. These structures may also be used to train the potential in turn. :meth:`~.generate_dimer_configs` can be used to generate a set of dimers for molecular systems, while :meth:`~.evaluate_dimer_configs_simulator` can be used to evaluate the PES of these structures. See :ref:`dimer_demo` for an example of using these functions.

#. Checkpointing and Restart: As in other parts of the Orchestrator, multistep function calls benefit from built-in mechanisms for checkpointing and restarting long-running augmentation workflows.

For more details, see the full API for the module at :ref:`augmentor_module`.

Inheritance Graph
-----------------

.. inheritance-diagram::
   orchestrator.augmentor.augmentor_base
   :parts: 3
