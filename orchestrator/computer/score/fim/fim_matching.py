from pathlib import Path
from typing import Optional, Union
import json
import numpy as np
from ase import Atoms
from information_matching import ConvexOpt
from information_matching.precondition import preconditioner

from orchestrator.utils.data_standard import (METADATA_KEY,
                                              PLACEHOLDER_ARRAY_KEY)
from orchestrator.computer.score.score_base import (ConfigurationScore,
                                                    ScoreQuantity)
from orchestrator.utils.input_output import try_loading_ase_keys

from .fim_training_set import FIMTrainingSetScore
from .fim_property import FIMPropertyScore
from .utils import FIMError


class FIMMatchingScore(ConfigurationScore):
    """
    A method to quantify the importance of dataset by computing the optimal
    configuration weights via information-matching method.

    This module wraps over `information_matching
    <https://github.com/yonatank93/information-matching>`_ Python package,
    which uses the FIMs of the candidate configurations and target property,
    and perform matrix matching to obtain the optimal weight to assign to each
    candidate configuration. This calculation ensure that the uncertainty of
    the target property is smaller than the predefined target uncertainty. The
    weights also measures the "importance" of the configurations, and their
    reciprocal square-root give the target precision to achieve when
    generating the ground truth data.

    .. note::

       Please use `compute_batch()` method. The optimization problem solved in
       the information-matching method is likely to be infeasible when only a
       single candidate atomic configuration is provided.
    """

    OUTPUT_KEY = 'fim_matching_weight'
    supported_score_quantities = [ScoreQuantity.IMPORTANCE]
    default_solver_kwargs = {'solver': 'SDPA'}
    default_fim_preconditioning_kwargs = {'scale_type': 'max_frobenius'}

    def __init__(self, **kwargs):
        super().__init__()
        # No init_args
        self._init_args = {}
        self._metadata = {}

        # Initialize some useful attributes
        self.convexopt = None
        self.fim_candidates = None
        self.fim_property = None
        self._ncandidates = None

    def compute(self, atoms, score_quantity, **kwargs):
        """
        User should call `compute_batch()` method to do FIM-matching.

        Running information-matching calculation with only a single atomic
        configuration candidate will most likely fail because the problem is
        more likely to be infeasible.
        """
        raise FIMError(
            "Please use `compute_batch` instead with a list of Atoms")

    def compute_batch(self,
                      list_of_atoms: list[Atoms],
                      fim_property: Union[str, np.ndarray],
                      score_quantity: str,
                      convexopt_init_kwargs: Optional[dict] = None,
                      fim_preconditioning_kwargs: Optional[dict] = None,
                      solver_kwargs: Optional[dict] = None,
                      weight_tolerance: Optional[dict] = None,
                      **kwargs) -> list:
        """
        Runs the FIM-matching calculation for a batch of atomic configurations.
        This is intended to be able to be used in a serial (non-distributed)
        manner, outside of a proper orchestrator scheduler.

        **Notes:** In other ConfigurationScore modules, the argument
        `score_quantity` comes as the second positional argument. But for FIM-
        matching, it is weird to separate the placements of `list_of_atoms` and
        `fim_property` arguments. This is not a deal-breaker, just a preference
        based on how each argument have connection to each other.

        .. note::

           If there are multiple properties to target, the FIMs of all those
           target properties should be combined (summed) to give a single FIM.
           Additionally, the target covariance to achieve should already be
           included in this FIM.


        :param list_of_atoms: a list of the ASE Atoms object
        :type list_of_atoms: list of ase.Atoms

        :param fim_property: the FIM of the target properties.
        :type fim_property: Path-like or np.ndarray

        :param score_quantity: The type of score value to compute. For this
            module, the accepted argument in "IMPORTANCE".
        :type score_quantity: str ("IMPORTANCE")

        :param convexopt_init_kwargs: Keyword arguments to instantiate the
            `infomation_matching.ConvexOpt` object. Available keys are:


            - **weight_upper_bound** (*float* or *np.ndarray*): Sets an upper
              bound for the optimal weights.
            - **l1norm_obj** (*bool*): Uses a stricter (but slower) objective
              function.

        :type convexopt_init_kwargs: dict

        :param fim_preconditioning_kwargs: Keyword arguments for
            preconditioning the FIMs to help with solving the convex
            optimization problem in information-matching. The preconditioning
            is done by scaling the FIMs by some numbers. If None, the default
            preconditioning is used (using "max_fobenius" with 0 padding). If
            an empty dictionary is given, no preconditioning is done. Available
            keywords are:

            - **scale_type** (*str*): A string that specifies how the scaling
              factors are calculated. Currently, user can choose between
              "frobenius" or "max_frobenius".
            - **pad** (float): A small padding factor so that the inverse of
              the scaling factor doesn't diverge.

        :type fim_preconditioning_kwargs: dict

        :param solver_kwargs: Keyword arguments containing the convex
            optimization solver settings. An extensive list of available
            keywords is provided in the CVXPY
            `documentation
            <https://www.cvxpy.org/tutorial/solvers/index.html>`__.

        :type solver_kwargs: dict

        :param weight_tolerance: Keyword arguments that specify the tolerance
            for extracting non-zero weights. Available keywords are:


            - **zero_tol** (*float*): Sets the tolerance for the (primal)
              value of the weights.
            - **zero_tol_dual** (*float*): Sets the tolerance for the dual
              value of the weights.

            The optimal weight is set to **zero** when its (primal) value is
            less than `zero_tol`, and its dual value is greater than
            `zero_tol_dual`.

        :type weight_tolerance: dict

        :returns: Optimal weights for the training configurations
        :rtype: list
        """

        if isinstance(score_quantity, str):
            score_quantity = ScoreQuantity[
                score_quantity]  # Enum conversion uses []

        if score_quantity not in self.supported_score_quantities:
            raise RuntimeError(
                f'Requested compute value "{score_quantity}" is '
                'not supported by "{self.__class__.__name__}".'
                ' Supported quantities are '
                '"{self.supported_score_quantities}"')

        # Load information into atoms.info, for ASE>=3.23.0
        list_of_atoms = [
            try_loading_ase_keys(atoms)[0] for atoms in list_of_atoms
        ]

        # Read the FIM property
        self.fim_property = self._read_fim_property(fim_property)

        # Read the FIMs of the candidates
        self.fim_candidates = np.array([
            atoms.info[f'{FIMTrainingSetScore.OUTPUT_KEY}_score']
            for atoms in list_of_atoms
        ])

        # FIM matching
        opt_weights = self.fim_match(self.fim_property, self.fim_candidates,
                                     convexopt_init_kwargs,
                                     fim_preconditioning_kwargs, solver_kwargs,
                                     weight_tolerance)

        # Save metadata
        self._metadata.update({
            'convexopt_init_kwargs':
            str(convexopt_init_kwargs),
            'solver_kwargs':
            str(solver_kwargs),
            'weight_tolerance':
            str(weight_tolerance)
        })
        # Metadata for the FIM property information
        # If path, then store the path string. if numpy array, use a
        # placeholder
        if isinstance(fim_property, (str, Path)):
            metadata_fim_property = str(fim_property)
        else:
            metadata_fim_property = PLACEHOLDER_ARRAY_KEY
        self._metadata.update({'fim_property': metadata_fim_property})
        # Attach the metadata to the atoms.info
        for atoms in list_of_atoms:
            if METADATA_KEY not in atoms.info:
                atoms.info[METADATA_KEY] = {}
            atoms.info[METADATA_KEY][self.OUTPUT_KEY] = self._metadata

        return opt_weights

    def fim_match(self,
                  fim_property: Union[np.ndarray, dict],
                  fim_candidates: Union[np.ndarray, dict],
                  convexopt_init_kwargs: Optional[dict] = None,
                  fim_preconditioning_kwargs: Optional[dict] = None,
                  solver_kwargs: Optional[dict] = None,
                  weight_tolerance: Optional[dict] = None) -> list:
        """
        An alternative method to run information-matching.

        This method is the main machinary to do the FIM-matching using the
        information-matching package. This step includes instantiating the
        solver, solving the convex optimization problem, and extracting the
        optimal weights.

        .. note::

           If targeting multiple target properties, the FIMs of the target
           properties need to be summed first before inputing into this
           method.

        :param fim_property: Target FIM. If a dictionary is given, it should
            follow the convention in the information-matching `repo
            <https://github.com/yonatank93/information-matching/blob/main/information_matching/convex_optimization.type>`_.
        :type fim_property: np.ndarray or dict

        :param fim_candidates: FIMs of the candidate configurations. If a
            dictionary is given, it should follow the convention in the
            information-matching `repo
            <https://github.com/yonatank93/information-matching/blob/main/information_matching/convex_optimization.type>`_.
        :py fim_candidates: 3d-array or dict

        :param convexopt_init_kwargs: Keyword arguments to instantiate the
            `infomation_matching.ConvexOpt` object. Available keys are:


            - **weight_upper_bound** (*float* or *np.ndarray*): Sets an upper
              bound for the optimal weights.
            - **l1norm_obj** (*bool*): Uses a stricter (but slower) objective
              function.

        :type convexopt_init_kwargs: dict

        :param fim_preconditioning_kwargs: Keyword arguments for
            preconditioning the FIMs to help with solving the convex
            optimization problem in information-matching. The preconditioning
            is done by scaling the FIMs by some numbers. If None, the default
            preconditioning is used (using "max_fobenius" with 0 padding). If
            an empty dictionary is given, no preconditioning is done. Available
            keywords are:

            - **scale_type** (*str*): A string that specifies how the scaling
              factors are calculated. Currently, user can choose between
              "frobenius" or "max_frobenius".
            - **pad** (float): A small padding factor so that the inverse of
              the scaling factor doesn't diverge.

        :type fim_preconditioning_kwargs: dict

        :param solver_kwargs: Keyword arguments containing the convex
            optimization solver settings. An extensive list of available
            keywords is provided in the CVXPY
            `documentation
            <https://www.cvxpy.org/tutorial/solvers/index.html>`__.

        :type solver_kwargs: dict

        :param weight_tolerance: Keyword arguments that specify the tolerance
            for extracting non-zero weights. Available keywords are:


            - **zero_tol** (*float*): Sets the tolerance for the (primal)
              value of the weights.
            - **zero_tol_dual** (*float*): Sets the tolerance for the dual
              value of the weights.

            The optimal weight is set to **zero** when its (primal) value is
            less than `zero_tol`, and its dual value is greater than
            `zero_tol_dual`.

        :type weight_tolerance: dict

        :returns: Optimal weights for the training configurations
        :rtype: list
        """
        # Take care of some kwargs
        if not convexopt_init_kwargs:
            convexopt_init_kwargs = {}
        if not solver_kwargs:
            solver_kwargs = self.default_solver_kwargs
        if not weight_tolerance:
            weight_tolerance = {}

        # Setup the problem, including reading the FIMs and instantiating
        # the solver.
        self.setup_problem(fim_property, fim_candidates, convexopt_init_kwargs,
                           fim_preconditioning_kwargs)
        # Solve
        self.convexopt.solve(**solver_kwargs)
        # Optimal weights
        opt_weights = self.convexopt.get_config_weights(**weight_tolerance)
        return self._insert_zero_weights(opt_weights)

    def setup_problem(self, fim_property: Union[np.ndarray, dict],
                      fim_candidates: Union[np.ndarray, dict],
                      convexopt_init_kwargs: Optional[dict],
                      fim_preconditioning_kwargs: Optional[dict]) -> list:
        """
        Setup the FIM-matching problem.

        This step includes reading and preparing the FIMs and instantiating
        the solver.
        """
        # Set default preconditioning kwargs as needed
        if fim_preconditioning_kwargs is None:
            fim_preconditioning_kwargs = (
                self.default_fim_preconditioning_kwargs)
        # Update class attributes
        self._ncandidates = len(fim_candidates)
        # Prepare the FIM arguments, convert the format, and precondition FIMs
        fim_property_dict, fim_candidates_dict = self._convert_fims_format(
            fim_property, fim_candidates, fim_preconditioning_kwargs)

        # Instantiate information-matching
        self.convexopt = ConvexOpt(fim_property_dict, fim_candidates_dict,
                                   **convexopt_init_kwargs)

    @staticmethod
    def _read_fim_property(fim_property: Union[np.ndarray, str]) -> np.ndarray:
        """
        Read the FIM of target properties.

        We allow user to input the FIM of the target property as either
        a numpy array or a path to a numpy or json file. This function will
        handle loading the FIM from either format.
        """
        if isinstance(fim_property, (str, Path)):
            ext = Path(fim_property).suffix
            # print('File extension:', ext)
            if ext == '.npy':
                # The FIM is in a numpy file
                fim_property = np.load(fim_property)
            elif ext == '.json':
                # The FIM is in a JSON file
                with open(fim_property, 'r') as f:
                    # This dictionary contains not only the FIM, but also the
                    # metadata
                    fim_property_data = json.load(f)
                fim_property = np.array(
                    fim_property_data[f'{FIMPropertyScore.OUTPUT_KEY}_score'])
            else:
                raise FIMError(
                    'fim_property input can only be a numpy array or '
                    'a numpy or json file path')
        return fim_property

    def _convert_fims_format(self, fim_property: Union[np.ndarray, dict],
                             fim_candidates: Union[np.ndarray, dict],
                             fim_preconditioning_kwargs: dict) -> (dict, dict):
        """
        Convert the FIMs format into dictionaries and get the scaling factor
        to help `information_matching` calculation to converge.
        """
        fim_candidates_dict = self._convert_fim_candidates_format(
            fim_candidates, fim_preconditioning_kwargs)
        fim_property_dict = self._convert_fim_property_format(
            fim_property, fim_preconditioning_kwargs)
        return fim_property_dict, fim_candidates_dict

    @staticmethod
    def _convert_fim_candidates_format(
            fim_candidates: Union[np.ndarray, dict],
            fim_preconditioning_kwargs: dict) -> dict:
        """
        Convert the format of fim_candidates into a dictionary, and use an
        internal function in `information_matching` to compute the
        preconditioning scaling factor.

        The choice of scaling factor is the "max_frobenius" type. The scaling
        factor is calculated by first computing the Frobenius norm of the FIM
        for each candidate, then take the largest value. The multiplicative
        scaling factor is set to the reciprocal of this number.

        This scaling factor helps the convergence of the optimizer.

        :param fim_candidates: an collection of the FIMs of the cancidate
            configurations, in the format of a 3d array.
        :type fim_candidates: np.ndarray (M, P, P,) or dict

        :param fim_preconditioning_kwargs: additional keyword arguments for
            preconditioning the FIMs to help with solving optimization problem.
        :type fim_preconditioning_kwargs: dict

        :returns: Candidate FIMs and their scaling factor in dictionary
        :rtype: dict
        """
        # Put the candidate FIMs into a dictionary
        if isinstance(fim_candidates, (list, np.ndarray)):
            fim_candidates_dict = {
                ii: fim
                for ii, fim in enumerate(fim_candidates)
            }
        else:
            fim_candidates_dict = fim_candidates.copy()

        if fim_preconditioning_kwargs:
            return preconditioner(fim_candidates_dict,
                                  **fim_preconditioning_kwargs)
        else:
            return fim_candidates_dict

    @staticmethod
    def _convert_fim_property_format(fim_property: Union[np.ndarray, dict],
                                     fim_preconditioning_kwargs: dict) -> dict:
        """
        Convert the format of fim_property into a dictionary, and use an
        internal function in `information_matching` to compute the
        preconditioning scaling factor.

        The multiplicative scaling is set to the reciprocal of the Frobenius
        norm of the FIM.

        :params fim_property: combined FIM of the target property. There should
            only be one array of shape (P, P,) here.
        :type fim_property: np.ndarray (P, P,) or dict

        :param fim_preconditioning_kwargs: additional keyword arguments for
            preconditioning the FIMs to help with solving optimization problem.
        :type fim_preconditioning_kwargs: dict

        :returns: Target FIM with the scaling factor in dictioary
        :rtype: dict
        """
        if fim_preconditioning_kwargs:
            return preconditioner(fim_property, **fim_preconditioning_kwargs)
        else:
            return fim_property

    def _insert_zero_weights(self, weights: np.ndarray) -> list:
        """
        Insert the zero weights to the list of optimal weights.

        The weights we get from `information_matching` package are only for
        the important data. However, we want to return weights for all,
        including the non-important, data. The weights of the non-important
        data will be set to 0.

        :params weights: Non-zero optimal weights
        :type weights: np.ndarray

        :returns: Optimal weights for all candidate configurations
        :rtype: list
        """
        all_weights = [0] * self._ncandidates
        for idx, weight in weights.items():
            all_weights[idx] = weight
        return all_weights

    def get_colabfit_property_definition(self) -> dict:
        return {
            'property-name': self.OUTPUT_KEY,

            'property-title': 'FIM-matching',

            'property-description': 'Estimated optimal weights for given '
            'target property.',

            # the fields that make up the FIM weights
            'score': {  # example: 0.0
                'type': 'float',
                'has-unit': True,
                'extent': [],
                'required': True,
                'description': 'Optimal weight for this training data for '
                'given target property FIM. The unit of the weight depends on '
                'the training quantity used, e.g. eV^{-2} if energy is '
                'used and (eV/Ang)^{-2} if forces is used.',
            },
            'fim-property': {  # example: '/path/to/array'
                'type': 'string',
                'has-unit': False,
                'extent': [],
                'required': True,
                'description': 'The location of the target property FIM.'
            },
            'fimmatch-init-kwargs': {  # JSON encoding of a dict, or None
                # example: {'weight_upper_bound': np.inf, 'l1norm_obj': False}
                'type': 'string',
                'has-unit': False,
                'extent': [],
                'required': False,
                'description': 'JSON encoding of a dictionary of keyword '
                'arguments passed to '
                'information-matching.ConvexOpt constructor.'
            },
            'solver-kwargs': {  # JSON encoding of a dict, or None
                # example: {'solver': 'SDPA', 'verbose': True}
                'type': 'string',
                'has-unit': False,
                'extent': [],
                'required': False,
                'description': 'JSON encoding of a dictionary of the convex '
                'optimization solver settings, passed to '
                'cvxpy.problem.solve method.'
            },
            'weight-tolerance': {  # JSON encoding of a dict, or None
                # example: {'zero_tol': 1e-4, 'zero_tol_dual': 1e-4}
                'type': 'string',
                'has-unit': False,
                'extent': [],
                'required': False,
                'description': 'JSON encoding of a dictionary of keyword '
                'arguments to filter the non-zero weights based on the primal '
                'and dual values returned by the convex optimization solver.'
            },
        }
