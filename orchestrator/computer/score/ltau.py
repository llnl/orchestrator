import os
import numpy as np
from ase import Atoms
from typing import Optional, Union, Any
from .score_base import AtomCenteredScore, ScoreQuantity
from orchestrator.utils.data_standard import (METADATA_KEY,
                                              PLACEHOLDER_ARRAY_KEY)
from ltau_ff.uq_estimator import UQEstimator


class LTAUForcesUQScore(AtomCenteredScore):
    """
    An ensemble-based UQ method for force predictions.

    This module uses the distributions of force error magnitudes for each atom
    sampled over the course of training to estimate that atom's force
    uncertainty. For test atoms, a nearest-neighbor search is performed (in the
    descriptor space) using the FAISS library to estimate the test atom's
    uncertainty using its nearest neighbors from the training set.
    """

    OUTPUT_KEY = 'ltau_forces_uq'
    supported_score_quantities = [ScoreQuantity.UNCERTAINTY]

    def __init__(
        self,
        train_descriptors: Union[str, np.ndarray],
        error_pdfs: Union[str, np.ndarray],
        bins: Optional[np.ndarray] = None,
        from_error_logs: Optional[bool] = False,
        error_logs_norm: Optional[bool] = False,
        nbins: Optional[int] = None,
        range_limits: Optional[Union[list, tuple, np.ndarray]] = None,
        bin_spacing: Optional[str] = 'log',
        index_type: Optional[str] = 'IndexFlatL2',
        index_args: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """
        :param train_descriptors: (N, D) array of atomic descriptors for the
            entire training set. Can also be string to numpy-readable file.
        :type train_descriptors: np.ndarray or str

        :param error_pdfs: an array of size (N, N_b) containing the PDFs for
            all N training points defined over N_b bins. If `from_error_logs`
            is True, then `error_pdfs` should be a list of M arrays, where each
            array is a size (T_m, N) array of errors logged for each training
            point for an ensemble of M models. T_m is the number of training
            epochs for the m-th model in the ensemble. Can also be a string to
            a numpy-readable file.
        :type error_pdfs: np.ndarray or list or str

        :param from_error_logs: if True, builds the training PDFs from the
            logged training errors.
        :type from_error_logs: bool

        :param error_logs_norm: if True, takes the norm of the error logs along
            axis=-1
        :type error_logs_norm: bool

        :param bins: array of bin edges. If `from_error_logs` is True and
            `bins` is not provided, must provide `nbins`, `range_limits`, and
            `bin_spacing` instead.
        :type bins: np.ndarray or None

        :param nbins: number of bins for PDFs. Required if `bins` is None
        :type nbins: int

        :param range_limits: the upper/lower limits of the bins. If not
            provided, uses the min/max error from `error_pdfs`. Only required
            if `from_error_logs` is True and `bins` is None.
        :type range_limits: tuple

        :param bin_spacing: one of "log" or "linear". Default is "log". Only
            required if `from_error_logs` is True and `bins` is None.
        :type bin_spacing: str

        :param index_type: one of ['IndexFlatL2', 'IndexHNSWFlat',
            'HNSW+IVFPQ'] specifying the index type for FAISS.  Default is
            'IndexFlatL2'. Required if `load_index` is False.
        :type index_type: str

        :param index_args: additional arguments to be passed directly to the
            FAISS index constructor. Required if `load_index` is False.
        :type index_args: dict
        """
        super().__init__()

        # Handle data loading; NOTE: needs to convert to abspath so that child
        # processes still know where to find the file
        if isinstance(train_descriptors, str):
            train_descriptors_name = os.path.abspath(train_descriptors)
            train_descriptors = np.load(train_descriptors)
        else:
            train_descriptors_name = PLACEHOLDER_ARRAY_KEY

        if isinstance(error_pdfs, str):
            error_pdfs_name = os.path.abspath(error_pdfs)
            error_pdfs = np.load(error_pdfs)
        else:
            error_pdfs_name = PLACEHOLDER_ARRAY_KEY

        if isinstance(bins, str):
            bins_name = os.path.abspath(bins)
            bins = np.load(bins)
        elif isinstance(bins, np.ndarray):
            bins_name = PLACEHOLDER_ARRAY_KEY
        else:
            bins_name = None

        if error_logs_norm:
            error_pdfs = np.linalg.norm(error_pdfs, axis=-1)

        # Initialize module
        self.estimator = UQEstimator(
            pdfs=error_pdfs,
            bins=bins,
            descriptors=train_descriptors,
            from_error_logs=from_error_logs,
            error_logs_norm=error_logs_norm,
            nbins=nbins,
            range_limits=range_limits,
            bin_spacing=bin_spacing,
            index_type=index_type,
            index_args=index_args,
        )

        # record input args to allow easy re-construction. Note that it's not
        # identical to _metadata because __init__ may support arguments which
        # may not be necessary for the property definition
        self._init_args = {
            'train_descriptors': train_descriptors,
            'error_pdfs': error_pdfs,
            'bins': self.estimator.bins,
            'from_error_logs': from_error_logs,
            'error_logs_norm': error_logs_norm,
            'nbins': nbins,
            'range_limits': range_limits,
            'bin_spacing': bin_spacing,
            'index_type': index_type,
            'index_args': index_args,
        }

        self._metadata = {
            'train_descriptors': train_descriptors_name,
            'error_trajectory': error_pdfs_name,
            'bins': bins_name,
            'index_type': index_type,
            'index_args': str(index_args),
        }

    def compute(
        self,
        atoms: Union[list[Atoms], Atoms],
        score_quantity: int = ScoreQuantity.UNCERTAINTY,
        descriptors_key: str = 'descriptors',
        num_nearest_neighbors: Optional[int] = 1,
        **kwargs,
    ) -> np.ndarray:
        """
        Calls compute_batch with a single-configuration list.

        :param atoms: the ASE Atoms objects.
        :type list_of_atoms: list

        :param descriptors_key: the key to use for extracting the descriptors
            from an ASE.Atoms object
        :type descriptors_key: str

        :param score_quantity: the type of score value to compute
        :type score_quantity: int

        :param num_nearest_neighbors: the number of neighbors to search for
            when performing UQ on test point
        :type num_nearest_neighbors: int

        :returns: (N, b) array of PDFs of errors for each atom.
        """
        if isinstance(atoms, Atoms):
            return self.compute_batch(
                [atoms],
                score_quantity,
                descriptors_key=descriptors_key,
                num_nearest_neighbors=num_nearest_neighbors,
            )[0]
        elif isinstance(atoms, list):
            return self.compute_batch(
                atoms,
                score_quantity,
                descriptors_key=descriptors_key,
                num_nearest_neighbors=num_nearest_neighbors,
            )[0]
        else:
            raise RuntimeError(
                f"Invalid input type of '{type(atoms)}' passed to .compute()")

    def compute_batch(
        self,
        list_of_atoms: list[Atoms],
        score_quantity: int = ScoreQuantity.UNCERTAINTY,
        descriptors_key: str = 'descriptors',
        num_nearest_neighbors: Optional[int] = 1,
        **kwargs,
    ) -> list[np.ndarray]:
        """
        :param list_of_atoms: a list of ASE Atoms objects. If a
            list of ASE Atoms is provided, then 'descriptors_key' should be
            provided in `args` to allow extraction of descriptors.
        :type list_of_atoms: list[Atoms]

        :param score_quantity: the type of score value to compute
        :type score_quantity: int

        :param descriptors_key: the key to use for extracting the descriptors
            from an ASE.Atoms object
        :type descriptors_key: str

        :param num_nearest_neighbors: the number of neighbors to search for
            when performing UQ on test points
        :type num_nearest_neighbors: int

        :returns: (N, b) array of PDFs of errors for each atom.
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
        if isinstance(list_of_atoms[0], Atoms):
            query_descriptors = np.concatenate(
                [atoms.arrays[descriptors_key] for atoms in list_of_atoms])
        else:
            query_descriptors = np.concatenate(list_of_atoms)

        errors = self.estimator.predict_errors(query_descriptors,
                                               topk=num_nearest_neighbors)
        for atoms in list_of_atoms:
            # NOTE: these are being attached here because ColabFit can't handle
            # nested keys during property extraction. e.g., extracting
            # "cut_name" from
            # atoms.info[METADATA_KEY][self.OUTPUT_KEY]['cut_name']
            for k, v in self._metadata.items():
                atoms.info[f'{self.OUTPUT_KEY}_{k}'] = v

            # these two keys have to be handled differently
            key = f'{self.OUTPUT_KEY}_bins'
            atoms.info[key] = self.estimator.bins

            key = f'{self.OUTPUT_KEY}_num_nearest_neighbors'
            atoms.info[key] = num_nearest_neighbors

            if METADATA_KEY not in atoms.info:
                atoms.info[METADATA_KEY] = {}

            atoms.info[METADATA_KEY][self.OUTPUT_KEY] = self._metadata

        return np.array_split(errors, np.cumsum(shapes)[:-1])

    def get_colabfit_property_definition(self) -> dict[str, Any]:
        return {
            'property-id': 'tag:staff@noreply.colabfit.org,2024-12-09:'
            f'property/{self.OUTPUT_KEY.replace("_", "-")}',

            "property-name": self.OUTPUT_KEY.replace("_", "-"),

            "property-title": "LTAU UQ",

            "property-description": "Predicted uncertainty for a "
            " property value.",

            # the fields that make up the descriptor
            "score": {  # example: (N, D) arrays
                "type": "float",
                "has-unit": True,
                "extent": [":"],
                "required": True,
                "description": "The predicted uncertainties. Rows "
                "correspond to data samples (e.g., atoms), and "
                "columns correspond to property components (e.g., "
                "force components).",
            },
            "train-descriptors": {  # example: "/path/to/array"
                "type": "string",
                "has-unit": False,
                "extent": [],
                "required": True,
                "description": "The location of the descriptor data."
            },
            "error-trajectory": {  # example: "/path/to/array"
                "type": "string",
                "has-unit": False,
                "extent": [],
                "required": True,
                "description": "The location of the error trajectory data"
            },
            "bins": {
                "type": "float",
                "has-unit": False,
                "extent": [":"],
                "required": True,
                "description": "The bin edges used for constructing the error"
                " distributions. Assumed to be same units as 'score'."
            },
            "num-nearest-neighbors": {  # example: 10
                "type": "int",
                "has-unit": False,
                "extent": [],
                "required": True,
                "description": "The number of nearest neighbors used for "
                "the UQ estimate",
            },
            "index-type": {  # example: "IndexFlatL2"
                "type": "string",
                "has-unit": False,
                "extent": [],
                "required": True,
                "description": "The type of FAISS indexer used for the "
                "nearest-neighbor search"
            },
            "index-args": {  # JSON encoding of a dict, or None
                # example: {'M': 32, 'efConstruction': 40, 'efSearch': 16}

                "type": "string",
                "has-unit": False,
                "extent": [],
                "required": False,
                "description": "JSON encoding of a dictionary of "
                "additional arguments passed to the FAISS index "
                "constructor"
            },
        }

    def get_colabfit_property_map(self) -> dict[str, Any]:
        return {
            'score': {
                'field': self.OUTPUT_KEY + '_score',
                'units': 'eV/Ang'
            },
            'num-nearest-neighbors': {
                'field': self.OUTPUT_KEY + '_num_nearest_neighbors',
                'units': None
            },
            'train-descriptors': {
                'field': self.OUTPUT_KEY + '_train_descriptors',
                'units': None
            },
            'error-trajectory': {
                'field': self.OUTPUT_KEY + '_error_trajectory',
                'units': None
            },
            'bins': {
                'field': self.OUTPUT_KEY + '_bins',
                'units': None
            },
            'index-type': {
                'field': self.OUTPUT_KEY + '_index_type',
                'units': None
            },
            'index-args': {
                'field': self.OUTPUT_KEY + '_index_args',
                'units': None
            },
        }
