from pathlib import Path
import numpy as np
from typing import Optional, Union
import numdifftools as nd
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from multiprocessing import Pool
from information_matching.transform import TransformBase

from orchestrator.computer.score.score_base import (ConfigurationScore,
                                                    ScoreQuantity)
from orchestrator.potential import Potential
from orchestrator.potential import potential_factory
from orchestrator.utils.data_standard import METADATA_KEY
from orchestrator.utils.input_output import try_loading_ase_keys
from orchestrator.utils.isinstance import isinstance_no_import

from .utils import (init_potential, get_column_index_to_parameter_info,
                    init_transform, FIMError)


class FIMTrainingSetScore(ConfigurationScore):
    """
    A module to compute the FIM of the training dataset with configuration
    energy, atomic forces, or stress quantity.

    The FIM is calculated by first computing the Jacobian, i.e., the derivative
    with respect to potential parameters, then take the dot product between
    the Jacobian with itself. The derivative is calculated numerically using
    `numdifftools` package.

    The element of the FIM matrix at row :math:`i`, column :math:`j`,
    approximates the second derivative of the potential predictions with
    respect to the potential parameters at indices :math:`i` and :math:`j`. The
    mapping between parameter indices and their corresponding potential
    parameters is stored in the attribute `self.fim_index_to_parameter`.

    .. note::

       When passing the `evaluate_kwargs` argument, each configuration should
       evaluate only **one** quantity at a time --- either energy, forces, or
       stress. Since these quantities have different physical units, the FIM
       should be computed separately for each.

       If multiple quantities need to be evaluated for a configuration,
       consider duplicating the configuration and assigning only one quantity
       to each duplicate.

       An exception will be raised if more than one quantity is requested for
       evaluation.

    .. note::

       Currently, this module only works with KIM **portable** models.

    """

    OUTPUT_KEY = 'fim_training_set'
    supported_score_quantities = [ScoreQuantity.SENSITIVITY]
    supported_potential_type = ['KIM']

    default_evaluate_kwargs = {
        'compute_energy': False,
        'compute_forces': True,
        'compute_stress': False
    }
    default_derivative_kwargs = {'method': 'central'}

    def __init__(self, **kwargs):
        super().__init__()
        # No init_args
        self._init_args = {}
        self._metadata = {}

        # Initialize some useful attributes
        self.fim_index_to_parameter = None
        self.transform = None
        self._tunable_params_idx = None
        self._params_dict_tpl = None
        self._params_dict_retrieve = None
        self.list_of_jac = None

    def compute(self,
                atoms: Atoms,
                score_quantity: str,
                potential: Union[dict, Potential],
                parameters_optimize: Optional[dict] = None,
                transform: Optional[Union[dict, TransformBase]] = None,
                evaluate_kwargs: Optional[dict] = None,
                derivative_kwargs: Optional[dict] = None,
                mask: Optional[list] = None,
                **kwargs) -> np.ndarray:
        """
        Runs the FIM calculation for a single atomic configuration. This is
        intended to be able to be used in a serial (non-distributed) manner,
        outside of a proper orchestrator scheduler.

        .. note::

           When passing the `evaluate_kwargs` argument, each configuration
           should evaluate only **one** quantity at a time --- either energy,
           forces, or stress. Since these quantities have different physical
           units, the FIM should be computed separately for each.

           If multiple quantities need to be evaluated for a configuration,
           consider duplicating the configuration and assigning only one
           quantity to each duplicate.

           A `UserWarning` will be issued if more than one quantity is
           requested for evaluation.


        :param atoms: the ASE Atoms object
        :type atoms: ase.Atoms

        :param score_quantity: The type of score value to compute. For this
            module, the accepted argument in "SENSITIVITY".
        :type score_quantity: str ("SENSITIVITY")

        :param potential: input dictionary to instantiate the potential (using
            init_potential) or the potential instance itself.
        :type potential: dict (preferred) or orchestrator.potential.Potential

        :param parameters_optimize: Potential parameters to differentiate and
            their values.
        :type parameters_optimize: dict

        :param transform: A dictionary containing information to instantiate
            parameter transformation class. Required keys are "transform_type"
            and "transform_args".
        :type transform_name: dict (preferred) or TransformBase

        :param evaluate_kwargs: specify to compute energy, forces, and/or
            stress, with key `compute_<quantity>` set to boolean value. The
            default is to compute forces only.
        :type evaluate_kwargs: dict

        :param derivative_kwargs: keyword arguments for the Jacobian
            calculation via `numdifftools` Python package. See see
            `numdifftools`
            `documentation <https://numdifftools.readthedocs.io/en/master/
            reference/numdifftools.html#numdifftools.core.Jacobian>`_
            for the list of available keywords.
        :type derivative_kwargs: dict

        :param mask: a binary masking array that can be used to exclude rows of
            the Jacobian matrix. For example, we can use this array if we want
            to compute the FIM of atomic forces of certain atoms in the
            configuration. However, note that since the masking array is
            applied to the Jacobian, then if we want to include all force
            components on atom `i` (zero-base index) in a configuration with
            `N` atoms, then the masking array will look like an array of zeros
            with length `3N`, but with element `3*i : 3*(i+1)` set to 1.
        :type mask: str or np.ndarray

        :returns: (p, p) array of the FIM, where p is the number of potential
            parameters
        :rtype: np.ndarray
        """
        return self.compute_batch([atoms], score_quantity, potential,
                                  parameters_optimize, transform,
                                  evaluate_kwargs, derivative_kwargs, [mask],
                                  **kwargs)[0]

    def compute_batch(self,
                      list_of_atoms: list[Atoms],
                      score_quantity: str,
                      potential: Union[str, Potential],
                      parameters_optimize: dict,
                      transform: Optional[Union[dict, TransformBase]] = None,
                      evaluate_kwargs: Optional[dict] = None,
                      derivative_kwargs: Optional[dict] = None,
                      list_of_mask: Optional[Union[list[str],
                                                   list[np.ndarray]]] = None,
                      nprocs: Optional[int] = 1,
                      **kwargs) -> list:
        """
        Runs the FIM calculation for a batch of atomic configurations. This is
        intended to be able to be used in a serial (non-distributed) manner,
        outside of a proper orchestrator scheduler.

        .. note::

           When passing the `evaluate_kwargs` argument, each configuration
           should evaluate only **one** quantity at a time --- either energy,
           forces, or stress. Since these quantities have different physical
           units, the FIM should be computed separately for each.

           If multiple quantities need to be evaluated for a configuration,
           consider duplicating the configuration and assigning only one
           quantity to each duplicate.

           A `UserWarning` will be issued if more than one quantity is
           requested for evaluation.


        :param list_of_atoms: a list of ASE Atoms objects
        :type list_of_atoms: list of ase.Atoms

        :param score_quantity: The type of score value to compute. For this
            module, the accepted argument in "SENSITIVITY".
        :type score_quantity: str ("SENSITIVITY")

        :param potential: input dictionary to instantiate the potential (using
            init_potential) or the potential instance itself.
        :type potential: dict (preferred) or orchestrator.potential.Potential

        :param parameters_optimize: Potential parameters to differentiate and
            their values.
        :type parameters_optimize: dict

        :param transform: A dictionary containing information to instantiate
            parameter transformation class. Required keys are "transform_type"
            and "transform_args".
        :type transform_name: dict (preferred) or TransformBase

        :param evaluate_kwargs: specify to compute energy, forces, and/or
            stress, with key `compute_<quantity>` set to boolean value. The
            default is to compute forces only.
            For each key, a boolean value or a list of boolean with length 1 or
            `len(list_of_atoms)` should be given.
        :type evaluate_kwargs: dict

        :param derivative_kwargs: keyword arguments for the Jacobian
            calculation via `numdifftools` Python package. See see
            `numdifftools`
            `documentation <https://numdifftools.readthedocs.io/en/master/
            reference/numdifftools.html#numdifftools.core.Jacobian>`_
            for the list of available keywords.
        :type derivative_kwargs: dict

        :param list_of_mask: a list of binary masking array that can be used to
            exclude rows of the Jacobian matrix. For example, we can use this
            array if we want to compute the FIM of atomic forces of certain
            atoms in the configuration. However, note that since the masking
            array is applied to the Jacobian, then if we want to include all
            force components on atom `i` (zero-base index) in a configuration
            with `N` atoms, then the masking array will look like an array of
            zeros with length `3N`, but with element `3*i : 3*(i+1)` set to 1.
        :type list_of_mask: list of str or list of np.ndarray

        :param nprocs: number of parallel processes to use
        :type nprocs: int

        :returns: list of M (P, P) arrays of (P, P) FIM for each of the M
            atomic configurations.
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

        # Deal different format we allow for the potential argument
        # Additionally, check if the potential type is supported.
        if isinstance(potential, dict):
            # Check support
            if potential['potential_type'] in self.supported_potential_type:
                potential_init_args = potential
                # Instantiate the potential
                potential = self._init_potential(potential_init_args)
            else:
                raise FIMError(
                    'Potential is not supported for FIM calculation')
        elif isinstance_no_import(potential, 'Potential'):
            # Get the potential_type in strings
            for key, val in potential_factory._creators.items():
                if isinstance(potential, val):
                    potential_type = key
                    break
            # Put the potential information into a dictionary
            if potential_type in self.supported_potential_type:
                potential_init_args = {
                    'potential_type': potential_type,
                    'potential_args': potential.args
                }
            else:
                # Potential is not supported
                raise FIMError(
                    'Potential is not supported for FIM calculation')

        # Check the parameter transformation
        if not transform:
            self.transform = init_transform('AffineTransform', {})
        elif isinstance(transform, TransformBase):
            self.transform = transform
        elif isinstance(transform, dict):
            self.transform = init_transform(**transform)
        else:
            raise TypeError(
                'Invalid transform input: expected None, a TransformBase '
                'instance, or a dictionary')
        # We also want to store the transformation argument as a dictionary for
        # metadata
        transform_meta = {
            'transform_type': self.transform.__class__.__name__,
            'transform_args': self.transform.jsonable_kwargs
        }

        # Set default quantities to evaluate
        if not evaluate_kwargs:
            evaluate_kwargs = self.default_evaluate_kwargs

        # Set default mask
        nconfigs = len(list_of_atoms)
        # print("Mask:", list_of_mask)
        if list_of_mask is None:
            # Special case: No mask specified, we include all elements of the
            # Jacobian. We simply just multiply the matrix with int 1 later.
            list_of_mask = np.ones(nconfigs)
        elif isinstance(list_of_mask, (list, np.ndarray)):
            # However, for the interface exposed to the user, we shouldn't
            # allow user to input a 1D array. User might think that with 1D
            # array, each element correspond to different configuration. But,
            # user should just remove the configuration from list_of_atoms. Or
            # user might want to apply the same mask to all configurations, but
            # each configuration might have different number of atoms. So to
            # avoid this confusion, let's just enforce the input format.
            for ii, mask in enumerate(list_of_mask):
                if isinstance(mask, list):
                    # Convert to np.array
                    list_of_mask[ii] = np.array(mask, dtype=int)
                elif isinstance(mask, (Path, str)):
                    # This is when the element of the list is a path to the
                    # mask numpy file, and we allow this input
                    if Path(mask).suffix in ['.npy', '.npz']:
                        list_of_mask[ii] = np.load(mask, dtype=int)
                    elif Path(mask).suffix == '.txt':
                        list_of_mask[ii] = np.loadtxt(mask, dtype=int)
                    else:
                        raise TypeError(
                            'The masking array file must be in either .npy or '
                            '.txt format.')
                elif mask is None:
                    # If None, then we include all Jacobian elements. We are
                    # just using an integer because we don't know how many
                    # rows are in the Jacobian.
                    list_of_mask[ii] = 1
                elif isinstance(mask, (int, bool, float)):
                    # But, we don't allow any other single value element.
                    raise TypeError(
                        'Please input an array of binary mask for each '
                        'configuration')

        # Set parameters optimize -- This not only updates the parameters but
        # set a function that does conversion from parameter array to
        # dictionary.
        self._set_parameters_optimize(potential, parameters_optimize)
        self.fim_index_to_parameter = get_column_index_to_parameter_info(
            parameters_optimize)
        # Get the parameters that we will use to evaluate the derivative
        params0_raw = self._params_dict_tpl.copy()
        params0 = self._convert_params_dict_to_array(params0_raw)
        transformed_params0 = self._parameters_transform(params0)

        # The default derivative settings --- Use central difference method and
        # set the step size to be 10% of the parameter values
        if not derivative_kwargs:
            derivative_kwargs = self.default_derivative_kwargs
            derivative_kwargs.update({'step': 0.1 * transformed_params0})

        # Load information to atoms.info, for ASE>=3.23.0
        list_of_atoms = [
            try_loading_ase_keys(atoms)[0] for atoms in list_of_atoms
        ]

        # Convert the format of evaluate_kwargs to a list of dictionary
        nconfigs = len(list_of_atoms)
        # For each key, make the argument a list
        evaluate_kwargs_copy = evaluate_kwargs.copy()
        for key, def_val in self.default_evaluate_kwargs.items():
            if key in evaluate_kwargs_copy:
                val = evaluate_kwargs_copy[key]
            else:
                # Set default value
                val = def_val

            # Take care of the shape and format
            if isinstance(val, (bool, int)):
                # A booloean value is given, set this value for all atoms
                evaluate_kwargs_copy[key] = [val] * nconfigs
            if isinstance(val, (list, np.ndarray, tuple)):
                len_val = len(val)
                if len_val == 1:
                    # Only 1 value specified, use the same value for all atoms
                    evaluate_kwargs_copy[key] = list(val) * nconfigs
                elif len_val != nconfigs:
                    raise FIMError(f'For `{key}` key, please specify '
                                   'either 1 value only or {nconfigs} values')
        # Put the kwargs into a list of dictionary, so we can make an iterable
        # out of it for parallelization
        evaluate_kwargs_list = []
        for ii in range(nconfigs):
            one_atoms_kwargs = {}
            for key in evaluate_kwargs_copy:
                one_atoms_kwargs.update({key: evaluate_kwargs_copy[key][ii]})
            evaluate_kwargs_list.append(one_atoms_kwargs)

        # Create an iterable
        iterable_items = [[
            atoms, mask, ekwargs, potential_init_args, transformed_params0,
            derivative_kwargs
        ] for atoms, mask, ekwargs in zip(list_of_atoms, list_of_mask,
                                          evaluate_kwargs_list)]
        if nprocs > 1:
            with Pool(nprocs) as executor:
                iterable = list(
                    executor.map(self._generate_one_fim_calculation_item,
                                 iterable_items))
        else:
            iterable = list(
                map(self._generate_one_fim_calculation_item, iterable_items))

        # Run
        if nprocs > 1:
            with Pool(nprocs) as executor:
                self.list_of_jac = list(
                    executor.map(self._function_parallel_wrapper, iterable))
        else:
            self.list_of_jac = list(
                map(self._function_parallel_wrapper, iterable))
        list_of_fim = [jac.T @ jac for jac in self.list_of_jac]

        # Metadata
        self._metadata = {
            'potential': {
                'type': potential_init_args['potential_type'],
                'name': potential_init_args['potential_args']['kim_id']
            },
            'parameters_optimize': str(parameters_optimize),
            'transform': transform_meta,
            'derivative_kwargs': str(derivative_kwargs),
            'fim_index_to_parameter': self.fim_index_to_parameter
        }
        for ii, atoms in enumerate(list_of_atoms):
            if METADATA_KEY not in atoms.info:
                atoms.info[METADATA_KEY] = {}
            # For each atom, we might have different values of compute_energy,
            # compute_forces, and compute_stress.
            metadata_atoms = self._metadata.copy()
            metadata_atoms.update(
                {'evaluate_quantities': evaluate_kwargs_list[ii]})
            # We also want to store the information about the masking array
            mask = list_of_mask[ii]
            if isinstance(mask, np.ndarray):
                mask = list(mask)
            metadata_atoms.update({'mask': mask})
            # Put the metadata into atoms.info
            atoms.info[METADATA_KEY][self.OUTPUT_KEY] = metadata_atoms
            # As a temporary patch, at the end of the calculation reassign
            # SinglePointCalculator to the atom
            sp_calc = SinglePointCalculator(atoms)
            atoms.calc = sp_calc

        return list_of_fim

    def _generate_one_fim_calculation_item(self, items):
        """
        A function that generate one dictionary that contains information for
        FIM calculation.
        """
        (atoms, mask, ekwargs, potential_init_args, transformed_params0,
         derivative_kwargs) = items
        # We should just evaluate one quantity for each configuration,
        # because each quantity has different physical unit
        if sum(ekwargs.values()) > 1:
            # Just raise a warning for now
            raise FIMError(
                "You should only evaluate one quantity per configuration, "
                "because they have different physical units. If more than "
                "one quantity is desired to use per configuration, "
                "duplicate the configuration and assign different "
                "quantity for each of the duplicates.")

        return {
            'params': transformed_params0,  # Parameters to evaluate
            'potential_init_args': potential_init_args,  # Potential init args
            'atoms': atoms,  # Configuration
            'evaluate_kwargs': ekwargs,  # Evaluate arguments
            'mask': mask,  # Jacobian element mask
            **derivative_kwargs  # Derivative settings
        }

    def _set_parameters_optimize(self, potential, parameters_optimize):
        """
        This function basically sets necessary information for other
        function(s) that convert parameter format from an array that the FIM
        module uses to a dictionary that the potential module uses.

        The paraameters_optimize argument still has similar format as KLIFF and
        should look like::

           {
               name1: [["default"], [val], ...],  # `extent` times
               name2: [["default", "fix"], [val, "fix"], ...],
           }

        The parameter format that potential module uses looks like::

           {
               name1: [[ext1, ext2, ...], [val1, val2, ...]],
               name2: [[ext1, ext2, ...], [val1, val2, ...]],
           }
        """
        self._tunable_params_idx = []  # [[name, ext, val], ...]
        fixed_params_idx = []  # [[name, ext, val], ...]
        for name, values in parameters_optimize.items():
            # Iterate over the values
            for ii, val in enumerate(values):
                if len(val) == 1:
                    # Only the mode is given -- There are 2 cases: "default" or
                    # value. In both cases, the parameters are tunable.
                    if val[0] == "default":
                        # Retrieve the default parameter value
                        par_val_dict = potential.get_params(**{name: [ii]})
                        par_val = par_val_dict[name][1][0]
                    else:
                        par_val = val[0]
                    # Append the tunable parameter information [name, idx]
                    self._tunable_params_idx.append([name, ii, par_val])
                elif len(val) == 2:
                    # First element is the value, and second element is the
                    # mode. The parameter is always fixed.
                    if val[0] != "default":
                        # Append the fixed parameter information [name, idx] --
                        # Only append IF we don't use default value
                        fixed_params_idx.append([name, ii, val[0]])
        # After collecting the tunable parameters name and indices, let's
        # create a template dictionary to make it easier to generate parameter
        # dictionary later.
        self._params_dict_tpl = {}
        # The template should contain placeholder for the tunable parameters
        for item in self._tunable_params_idx:
            name, idx, val = item
            if name not in self._params_dict_tpl:
                self._params_dict_tpl.update({name: [[], []]})
            self._params_dict_tpl[name][0].append(idx)  # Index
            self._params_dict_tpl[name][1].append(val)  # Value
        # The template should also contain values for fixed parameters, in case
        # they are different from the default values.
        for item in fixed_params_idx:
            name, idx, val = item
            if name not in self._params_dict_tpl:
                self._params_dict_tpl.update({name: [[], []]})
            self._params_dict_tpl[name][0].append(idx)  # Index
            self._params_dict_tpl[name][1].append(val)  # Value - fixed

    def _convert_params_array_to_dict(self, params_array):
        """
        Convert potential parameters from array format to a dictionary format
        that can be used to update the potential parameters in ASE KIM
        calculator.
        """
        params_dict = self._params_dict_tpl.copy()
        for val, item in zip(params_array, self._tunable_params_idx):
            name, idx, _ = item
            loc = params_dict[name][0].index(idx)
            params_dict[name][1][loc] = val
        return params_dict

    def _convert_params_dict_to_array(self, params_dict):
        """
        Convert potential parameters from dictionary format ouput by ASE KIM
        calculator to an array.
        """
        params_array = np.zeros(len(self._tunable_params_idx))
        for ii, (name, idx, _) in enumerate(self._tunable_params_idx):
            ext_list, vals = params_dict[name]
            loc = np.where(np.array(ext_list) == idx)[0][0]
            params_array[ii] = vals[loc]
        return params_array

    def _function_parallel_wrapper(self, kwargs: dict) -> np.ndarray:
        """
        Function for parallelization.
        """
        return self._compute_jacobian(**kwargs)

    def _compute_jacobian(self, params: np.ndarray, potential_init_args: dict,
                          atoms: Atoms, evaluate_kwargs: dict, mask: list,
                          **kwargs) -> np.ndarray:
        """
        Compute the Jacobian for one atoms object.
        """
        # Reinstantiate potential
        potential = self._init_potential(potential_init_args)
        # Compute the Jacobian
        jac_fn = nd.Jacobian(self._potential_evaluate_wrapper, **kwargs)
        jac = jac_fn(params, potential, atoms, **evaluate_kwargs)
        if isinstance(mask, (int, float)):
            return jac * mask
        else:
            return jac * mask.reshape((-1, 1))

    def _potential_evaluate_wrapper(self, params: np.ndarray,
                                    potential: Potential, atoms: Atoms,
                                    **kwargs) -> np.ndarray:
        """
        A wrapper function that will be fed into ``numdifftools.Jacobian``.
        Particularly, this function handles parameter transformation and
        potential evaluation.
        """
        # Update parameters
        params_orig = self._parameters_inverse_transform(params)
        params_orig_dict = self._convert_params_array_to_dict(params_orig)
        potential.set_params(**params_orig_dict)
        # Predictions
        predictions_all = potential.evaluate(atoms)
        # Concatenate the predictions -- By this point, **kwargs should contain
        # all compute_energy, compute_forces, and compute_stress.
        preds_concat = []
        if kwargs['compute_energy']:
            preds_concat = np.append(preds_concat, predictions_all[0])
        if kwargs['compute_forces']:
            preds_concat = np.append(preds_concat,
                                     predictions_all[1].flatten())
        if kwargs['compute_stress']:
            preds_concat = np.append(preds_concat, predictions_all[2])

        # If there is only 1 value, numdifftools wants us to make it a scalar
        # instead of an array of length 1.
        if len(preds_concat) == 1:
            return preds_concat[0]
        else:
            return preds_concat

    @staticmethod
    def _init_potential(input_args: dict) -> Potential:
        """
        Instantiate potential object from potential_args input dictionary.
        """
        return init_potential(input_args)

    def _parameters_transform(self, params: np.ndarray) -> np.ndarray:
        """
        Map the parameters from the original space to the transformed space.
        """
        return self.transform.transform(params)

    def _parameters_inverse_transform(self, params: np.ndarray) -> np.ndarray:
        """
        Map the parameters from the transformed space to the original space.
        """
        return self.transform.inverse_transform(params)

    def get_colabfit_property_definition(self) -> dict:
        return {
            'property-name': self.OUTPUT_KEY,

            'property-title': 'FIM training configuration',

            'property-description': 'The FIM of the training atomic '
            'configuration.',

            # the fields that make up the descriptor
            'score': {  # example: (P, P) array
                'type': 'float',
                'has-unit': True,
                'extent': [':'],
                'required': True,
                'description': 'The Fisher information matrix for the atomic '
                'configuration, calculated through the derivative of energy '
                'or forces with respect to potential parameters.',
            },
            'potential': {  # JSON encoding of a dict, or None
                # example: {'type': 'KIM', 'name': ''}
                'type': 'string',
                'has-unit': False,
                'extent': [],
                'required': True,
                'description': 'JSON encoding of a dictionary containing '
                'information about the potential.'
            },
            'parameters-optimize': {  # JSON encoding of a dict, or None
                # example: {'epsilon': 4.0, 'sigma': 2.0}
                'type': 'string',
                'has-unit': False,
                'extent': [],
                'required': True,
                'description': 'JSON encoding of a dictionary of potential '
                'parameters that are perturbed. '
            },
            'transform': {  # JSON encoding of a dict, or None
                # example: {'name': 'affine', 'args': {'A': 1.0, 'b': 0}}
                'type': 'string',
                'has-unit': False,
                'extent': [],
                'required': False,
                'description': 'JSON encoding of a dictionary containing '
                'information about the potential parameter transformation.'
            },
            'derivative-kwargs': {  # JSON encoding of a dict, or None
                # example: {'step': 0.1, method': 'central'}
                'type': 'string',
                'has-unit': False,
                'extent': [],
                'required': False,
                'description': 'JSON encoding of a dictionary of keyword '
                'arguments passed to numdifftools.Jacobian for the finite '
                'difference derivative calculation.'
            },
            'evaluate_quantities': {  # JSON encoding of a dict, or None
                # example: {'compute_energy': True, 'compute_forces': False}
                'type': 'string',
                'has-unit': False,
                'extent': [],
                'required': False,
                'description': 'JSON encoding of a dictionary of the quantity '
                'to compute for this configuration.'
            },
        }
