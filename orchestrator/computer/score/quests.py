import numpy as np
from ase import Atoms
from typing import Optional, Union, Any
from .score_base import DatasetScore, AtomCenteredScore, ScoreQuantity
from orchestrator.utils.data_standard import (METADATA_KEY, SELECTION_MASK_KEY,
                                              PLACEHOLDER_ARRAY_KEY)

from quests.entropy import (perfect_entropy, diversity, delta_entropy,
                            approx_delta_entropy, DEFAULT_BANDWIDTH,
                            DEFAULT_BATCH, DEFAULT_UQ_NBRS, DEFAULT_GRAPH_NBRS)


class QUESTSEfficiencyScore(DatasetScore):
    """
    An information-based method of quantifying dataset diversity.

    This module wraps the `quests` package, which performs a kernel density
    estimate of the distribution of points in a descriptor space to obtain a
    non-parametric estimate of the information entropy of a dataset. This
    estimate can then be used to identify the "efficiency" of the dataset (the
    lack of redundancy).
    """

    OUTPUT_KEY = 'quests_efficiency'
    supported_score_quantities = [ScoreQuantity.EFFICIENCY]

    def __init__(self, **kwargs):

        super().__init__()

        # QUESTS does not have any init_args
        self._init_args = {}
        self._metadata = {}

    def compute(
        self,
        dataset: list[Atoms],
        score_quantity: int,
        apply_mask: bool = False,
        descriptors_key: str = 'descriptors',
        bandwidth: float = DEFAULT_BANDWIDTH,
        batch_size: Optional[int] = DEFAULT_BATCH,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Computes the efficiency of the dataset.

        The efficiency is a measure of how little oversampling the dataset has.
        If the efficiency is near 1, the dataset has very little redundancy.

        :param dataset: a list of ASE Atoms objects.
        :type dataset: list

        :param score_quantity: the type of score value to compute
        :type score_quantity: int

        :param apply_mask: if True, apply the environment selection mask; can
            only be used if mask already exists for all configurations.
        :type apply_mask: bool

        :param descriptors_key: the key to use for extracting the descriptors
            from an ASE.Atoms object
        :type descriptors_key: str

        :param bandwidth: the bandwidth used by the Gaussian kernel for KDE.
        :type bandwidth: float

        :param batch_size: the maximum batch size to consider when performing a
            distance calculation.
        :type batch_size: int

        :returns: the efficiency of the dataset
        :rtype: float
        """

        if isinstance(score_quantity, str):
            score_quantity = ScoreQuantity[
                score_quantity]  # Enum conversion uses []

        if score_quantity not in self.supported_score_quantities:
            raise RuntimeError(
                f"Requested compute value '{score_quantity}' is "
                "not supported by '{self.__class__.__name__}'."
                " Supported quantities are "
                "'{self.supported_score_quantities}'")

        if isinstance(dataset, Atoms):
            dataset = [dataset]

        x = np.concatenate(
            [atoms.arrays[descriptors_key] for atoms in dataset])

        # Try to apply a selection mask
        if apply_mask:
            try:
                selection_mask = np.concatenate(
                    [atoms.arrays[SELECTION_MASK_KEY] for atoms in dataset])
                selection_mask = selection_mask.astype(bool)

                x = x[selection_mask]
            except KeyError:  # no mask exists for one of the configs
                raise RuntimeError("`apply_mask=True` was provided, but masks"
                                   " could not be found for all atoms.")

        entropy = perfect_entropy(x, h=bandwidth, batch_size=batch_size)
        max_entropy = np.log(x.shape[0])

        efficiency = entropy / max_entropy

        self._metadata = {'bandwidth': bandwidth}

        return np.array([efficiency])

    def get_colabfit_property_definition(self) -> dict[str, Any]:
        return {
            'property-id': 'tag:staff@noreply.colabfit.org,2024-12-09:'
            f'property/{self.OUTPUT_KEY.replace("_", "-")}',

            "property-name": self.OUTPUT_KEY.replace("_", "-"),

            "property-title": "QUESTS data efficiency",

            "property-description": "The efficiency is the ratio of "
            "H/maxH, which measures the dataset entropy relative to the "
            "theoretical maximum entropy of a non-overlapping dataset of the "
            "same size. " "Will be in the range [0, 1].",

            "score": {  # example: 0.90
                "type": "float",
                "has-unit": True,  # depends on log base; usually 'nats'
                "extent": [],
                "required": True,
                "description": "The efficiency H/maxH",
            },

            "bandwidth": {  # example: 0.015
                "type": "float",
                "has-unit": False,
                "extent": [],
                "required": True,
                "description": "The bandwidth used by the Gaussian kernel"
                " for KDE"
            },

            # NOTE: batch_size is not part of the definition because it is
            # only used for computational efficiency
        }


class QUESTSDiversityScore(DatasetScore):
    """
    An information-based method of quantifying dataset diversity.

    This module wraps the `quests` package, which performs a kernel density
    estimate of the distribution of points in a descriptor space to obtain a
    non-parametric estimate of the information entropy of a dataset. This
    estimate can then be used to identify the "diversity" of a dataset.
    """

    OUTPUT_KEY = 'quests_diversity'
    supported_score_quantities = [ScoreQuantity.DIVERSITY]

    def __init__(self, **kwargs):

        super().__init__()

        # QUESTS does not have any init_args
        self._init_args = {}
        self._metadata = {}

    def compute(
        self,
        dataset: list[Atoms],
        score_quantity: int,
        apply_mask: bool = False,
        descriptors_key: str = 'descriptors',
        bandwidth: float = DEFAULT_BANDWIDTH,
        batch_size: Optional[int] = DEFAULT_BATCH,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Computes the diversity of the dataset.

        The diversity is a measure of how well the dataset covers all regions
        of the configuration space that it spans.

        :param dataset: a list of ASE Atoms objects.
        :type dataset: list

        :param score_quantity: the type of score value to compute
        :type score_quantity: int

        :param apply_mask: if True, apply the environment selection mask; can
            only be used if mask already exists for all configurations.
        :type apply_mask: bool

        :param descriptors_key: the key to use for extracting the descriptors
            from an ASE.Atoms object
        :type descriptors_key: str

        :param bandwidth: the bandwidth used by the Gaussian kernel for KDE.
        :type bandwidth: float

        :param batch_size: the maximum batch size to consider when performing a
            distance calculation.
        :type batch_size: int

        :returns: returns the diversity of the dataset
        :rtype: float
        """
        if isinstance(score_quantity, str):
            score_quantity = ScoreQuantity[
                score_quantity]  # Enum conversion uses []

        if score_quantity not in self.supported_score_quantities:
            raise RuntimeError(
                f"Requested compute value '{score_quantity}' is "
                "not supported by '{self.__class__.__name__}'."
                " Supported quantities are "
                "'{self.supported_score_quantities}'")

        if isinstance(dataset, Atoms):
            dataset = [dataset]

        x = np.concatenate(
            [atoms.arrays[descriptors_key] for atoms in dataset])

        # Try to apply a selection mask
        if apply_mask:
            try:
                selection_mask = np.concatenate(
                    [atoms.arrays[SELECTION_MASK_KEY] for atoms in dataset])
                selection_mask = selection_mask.astype(bool)

                x = x[selection_mask]
            except KeyError:  # no mask exists for one of the configs
                raise RuntimeError("`apply_mask=True` was provided, but masks"
                                   " could not be found for all atoms.")

        return np.array(diversity(x, h=bandwidth, batch_size=batch_size))

    def get_colabfit_property_definition(self) -> dict[str, Any]:
        return {
            'property-id': 'tag:staff@noreply.colabfit.org,2024-12-09:'
            f'property/{self.OUTPUT_KEY.replace("_", "-")}',

            "property-name": self.OUTPUT_KEY.replace("_", "-"),

            "property-title": "QUESTS data diversity",

            "property-description": "The diversity is an estimate of how"
            " well the dataset covers the configuration space.",

            "score": {  # example: 0.90
                "type": "float",
                "has-unit": True,  # depends on log base; usually 'nats'
                "extent": [],
                "required": True,
                "description": "The diversity of the dataset",
            },

            "bandwidth": {  # example: 0.015
                "type": "float",
                "has-unit": False,
                "extent": [],
                "required": True,
                "description": "The bandwidth used by the Gaussian kernel"
                " for KDE"
            },

            # NOTE: batch_size is not part of the definition because it is
            # only used for computational efficiency
        }


class QUESTSDeltaEntropyScore(AtomCenteredScore):

    OUTPUT_KEY = 'quests_delta_entropy'
    supported_score_quantities = [
        ScoreQuantity.DELTA_ENTROPY,  # deltaH
    ]

    def compute(
        self,
        atoms: Atoms,
        score_quantity: int,
        reference_set: np.ndarray,
        descriptors_key: str = 'descriptors',
        approx: bool = False,
        bandwidth: float = DEFAULT_BANDWIDTH,
        num_nearest_neighbors: int = DEFAULT_UQ_NBRS,
        graph_neighbors: int = DEFAULT_GRAPH_NBRS,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Calls compute_batch with a single-configuration list.

        :param atoms: a single ASE Atoms objects.
        :type list_of_atoms: Atoms

        :param score_quantity: the type of score value to compute
        :type score_quantity: int

        :param reference_set: an (N, D) matrix with the descriptors of the
            reference.
        :type reference_set: np.ndarray

        :param descriptors_key: the key to use for extracting the descriptors
            from an ASE.Atoms object
        :type descriptors_key: str

        :param approx: if True, uses an approximate nearest neighbor search to
            compute the delta entropy values. Recommended for large data sizes.
        :type approx: bool

        :param bandwidth: the bandwidth used by the Gaussian kernel for KDE.
        :type bandwidth: float

        :param num_nearest_neighbors: number of nearest-neighbors to take into
            account when computing the approximate dH.
        :type num_nearest_neighbors: int

        :param graph_neighbors: a parameter used by pynndescent for performing
            the approximate nearest neighbor search.
        :type graph_neighbors: int

        :returns: returns the delta_entropy
        :rtype: float or np.ndarray
        """
        if isinstance(atoms, Atoms):
            return self.compute_batch(
                [atoms],
                score_quantity,
                descriptors_key=descriptors_key,
                approx=approx,
                bandwidth=bandwidth,
                num_nearest_neighbors=num_nearest_neighbors,
                graph_neighbors=graph_neighbors,
                reference_set=reference_set,
            )[0]
        elif isinstance(atoms, list):
            raise RuntimeError(
                ".compute_batch() should be used to compute this score for"
                " multiple atoms objects")
        else:
            raise RuntimeError(
                f"Invalid input type of '{type(atoms)}' passed to .compute()")

    def compute_batch(
        self,
        list_of_atoms: Union[list[Atoms], Atoms],
        score_quantity: int,
        reference_set: np.ndarray,
        descriptors_key: str = 'descriptors',
        approx: bool = False,
        bandwidth: float = DEFAULT_BANDWIDTH,
        batch_size: Optional[int] = DEFAULT_BATCH,
        num_nearest_neighbors: int = DEFAULT_UQ_NBRS,
        graph_neighbors: int = DEFAULT_GRAPH_NBRS,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Calls compute_batch with a single-configuration list.

        :param list_of_atoms: a list of ASE Atoms objects. If a
            list of ASE Atoms is provided, then 'descriptors_key' should be
            provided in `args` to allow extraction of descriptors.
        :type list_of_atoms: list

        :param score_quantity: the type of score value to compute
        :type score_quantity: int

        :param descriptors_key: the key to use for extracting the descriptors
            from an ASE.Atoms object
        :type descriptors_key: str

        :param approx: if True, uses an approximate nearest neighbor search to
            compute the delta entropy values. Recommended for large data sizes.
        :type approx: bool

        :param bandwidth: the bandwidth used by the Gaussian kernel for KDE.
        :type bandwidth: float

        :param batch_size: the maximum batch size to consider when performing a
            distance calculation.
        :type batch_size: int

        :param num_nearest_neighbors: number of nearest-neighbors to take into
            account when computing the approximate dH.
        :type num_nearest_neighbors: int

        :param graph_neighbors: a parameter used by pynndescent for performing
            the approximate nearest neighbor search.
        :type graph_neighbors: int

        :param reference_set: an (N, D) matrix with the descriptors of the
            reference.
        :type reference_set: np.ndarray

        :returns: the delta_entropy scores for each atom
        :rtype: float or np.ndarray
        """
        if isinstance(score_quantity, str):
            score_quantity = ScoreQuantity[
                score_quantity]  # Enum conversion uses []

        if score_quantity not in self.supported_score_quantities:
            raise RuntimeError(
                f"Requested compute value '{score_quantity}' is "
                "not supported by '{self.__class__.__name__}'."
                " Supported quantities are "
                "'{self.supported_score_quantities}'")

        shapes = [len(_) for _ in list_of_atoms]

        descriptors = np.concatenate(
            [atoms.arrays[descriptors_key] for atoms in list_of_atoms])

        if isinstance(reference_set, str):
            reference_set_name = reference_set  # for saving in _metadata
            reference_set = np.load(reference_set)
        else:
            reference_set_name = PLACEHOLDER_ARRAY_KEY

        if approx:
            results = approx_delta_entropy(descriptors,
                                           reference_set,
                                           h=bandwidth,
                                           n=num_nearest_neighbors,
                                           graph_neighbors=graph_neighbors)
        else:
            results = delta_entropy(descriptors,
                                    reference_set,
                                    h=bandwidth,
                                    batch_size=batch_size)

        self._metadata = {
            'reference_set': reference_set_name,
            'bandwidth': bandwidth,
            'approx': approx,
            'num_nearest_neighbors': num_nearest_neighbors,
            'graph_neighbors': graph_neighbors,
        }

        for atoms in list_of_atoms:
            # NOTE: these are being attached here because ColabFit can't handle
            # nested keys during property extraction. e.g. extracting
            # "cut_name" from
            # atoms.info[METADATA_KEY][self.OUTPUT_KEY]['cut_name']
            for k, v in self._metadata.items():
                atoms.info[f'{self.OUTPUT_KEY}_{k}'] = v

            if METADATA_KEY not in atoms.info:
                atoms.info[METADATA_KEY] = {}

            atoms.info[METADATA_KEY][self.OUTPUT_KEY] = self._metadata

        return np.array_split(results, np.cumsum(shapes)[:-1])

    def get_colabfit_property_definition(self) -> dict[str, Any]:
        return {
            'property-id': 'tag:staff@noreply.colabfit.org,2024-12-09:'
            f'property/{self.OUTPUT_KEY.replace("_", "-")}',

            "property-name": self.OUTPUT_KEY.replace("_", "-"),

            "property-title": "QUESTS differential entropy, dH",

            "property-description": "The estimated increase that a point"
            " will have to the entropy of a reference set",

            "score": {  # example: (N, 1) array of floats
                "type": "float",
                "has-unit": True,  # depends on log base; usually 'nats'
                "extent": [":"],
                "required": True,
                "description": "The estimated differential entropies for each"
                " atom.",
            },

            "reference-set": {  # example: "/path/to/array"
                "type": "string",
                "has-unit": False,
                "extent": [],
                "required": True,
                "description": "The location of the reference descriptors."
            },

            "bandwidth": {  # example: 0.015
                "type": "float",
                "has-unit": False,
                "extent": [],
                "required": True,
                "description": "The bandwidth used by the Gaussian kernel"
                " for KDE"
            },

            "num-nearest-neighbors": {  # example: 10
                "type": "int",
                "has-unit": False,
                "extent": [],
                "required": True,
                "description": "Number of nearest-neighbors to take into"
                " account when computing the approximate dH"
            },

            "graph-neighbors": {  # example: 10
                "type": "int",
                "has-unit": False,
                "extent": [],
                "required": True,
                "description": "A parameter used by pynndescent for "
                "performing the approximate nearest neighbor search"
            },
        }

    def get_colabfit_property_map(self) -> dict[str, Any]:
        return {
            'score': {
                'field': self.OUTPUT_KEY + '_score',
                'units': 'nats'
            },
            'bandwidth': {
                'field': self.OUTPUT_KEY + '_bandwidth',
                'units': None
            },
            'num-nearest-neighbors': {
                'field': self.OUTPUT_KEY + '_num_nearest_neighbors',
                'units': None
            },
        }
