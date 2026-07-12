from pathlib import Path
from typing import Optional, Union, List
from copy import deepcopy
import os
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json
import kimkit

from .utils import (init_potential, get_column_index_to_parameter_info,
                    init_transform, FIMError)

from information_matching.transform import TransformBase
from information_matching.fim.finitediff import FiniteDifference
from orchestrator.computer.score.score_base import ModelScore, ScoreQuantity
from orchestrator.potential import Potential, potential_factory
from orchestrator.target_property import (TargetProperty,
                                          target_property_builder)
from orchestrator.storage import storage_builder
from orchestrator.scheduler import scheduler_builder
from orchestrator.utils.data_standard import PLACEHOLDER_ARRAY_KEY
from orchestrator.utils.isinstance import isinstance_no_import


class FIMPropertyScore(ModelScore):
    """
    A module to compute the FIM of the target property with respect to the
    potential parameters.

    The FIM is calculated by first computing the Jacobian, i.e., the derivative
    with respect to potential parameters, then take the dot product between
    the Jacobian with itself. The derivative is calculated numerically using
    finite difference approach, implemented in the `information_matching`
    package.

    The element of the FIM matrix at row :math:`i`, column :math:`j`,
    approximates the second derivative of the potential predictions with
    respect to the potential parameters at indices :math:`i` and :math:`j`. The
    mapping between parameter indices and their corresponding potential
    parameters is stored in the attribute `self.fim_index_to_parameter`.

    User is also required to provide the target covariance matrix of the target
    property. If multiple target properties is given, e.g., when calling
    `compute_batch` method, one covariance matrix needs to be given for each
    target property. The method will then returns the FIM for each target
    property.

    .. note::

       Currently, this module only works with KIM **portable** models that
       support **writing parameters**.

    """

    OUTPUT_KEY = 'fim_property'
    data_file_name = 'score_results.json'
    supported_score_quantities = [ScoreQuantity.SENSITIVITY]
    supported_potential_type = ['KIM']
    property_output_dir = 'fim_property_output_files'

    def __init__(self, **kwargs):
        super().__init__()
        # Set directory to store all the intermediate potentials
        self.potential_dir = os.path.join(self.compute_args_subdir,
                                          'fim_property_potentials')
        if not os.path.isdir(self.potential_dir):
            os.makedirs(self.potential_dir, exist_ok=True)
        # No init_args
        self._init_args = {}
        self._metadata = {}

        # Initialize some useful attributes
        self.fim_index_to_parameter = None
        self.transform = None
        self._potential = None
        self._potential_init_args = None
        # For setting up parameters
        self._parameters_optimize = None
        self._tunable_params_idx = None
        self._params_dict_tpl = None
        self._params_dict_retrieve = None
        # For derivative
        self._params_set = None
        self._fd = None

        # We need this cwd when building the potential
        self._cwd = Path(os.getcwd()).resolve()

        # Default scheduler --- we need this default scheduler so that we can
        # place all the temporary files in a single, same directory.
        self.default_target_property_scheduler = scheduler_builder.build(
            'LOCAL', {'root_directory': self.property_output_dir})
        self._target_property_scheduler = None
        # Need this to cleanup directory

    def compute(self,
                target_property: dict,
                score_quantity: str,
                cov: Union[str, np.ndarray],
                potential: Union[dict, Potential],
                parameters_optimize: dict,
                transform: Optional[Union[dict, TransformBase]] = None,
                derivative_kwargs: Optional[dict] = None,
                return_jacobian: Optional[bool] = False,
                nprocs: Optional[int] = 1,
                **kwargs) -> np.ndarray:
        """
        Run the FIM calculation for a single target property. This is
        intended to be able to be used in a serial (non-distributed) manner,
        outside of a proper orchestrator scheduler.

        .. note::

           If :class:`~orchestrator.target_property.kimrun.KIMRun` is used,
           then the argument `flatten=True` in
           :class:`~orchestrator.target_property.kimrun.KIMRun.calculate_property`
           method is enforced.


        :param target_property: a dictionary about the target property,
            which should contain the following keys:

            - **`init_args`**: Used to instantiate the target property class
            - **`calculate_property_args`**: Contains arguments for
              `TargetProperty.calculate_property` method, excluding the
              potential and `iter_num`, which are automatically inserted during
              this calculation.

        :type target_property: dict

        :param score_quantity: The type of score value to compute. For this
            module, the accepted argument in "SENSITIVITY".
        :type score_quantity: str ("SENSITIVITY")

        :param cov: target covariance matrix of the target property to achieve.
        :type cov: Path-like (preferred) or np.ndarray

        :param potential: input dictionary to instantiate the potential (using
            init_potential) or the potential instance itself.
        :type potential: dict (preferred) or orchestrator.potential.Potential

        :param parameters_optimize: Potential parameters to differentiate and
            their values.
        :type parameters_optimize: dict

        :param transform: A dictionary containing information to instantiate
            parameter transformation class. Required keys are "transform_type"
            and "transform_args".
        :type transform: dict (preferred) or TransformBase

        :param derivative_kwargs: Additional arguments for instantiating
            `information_matching.fim.finitediff.FiniteDifference`, which are
            interpreted as the finite difference settings. Available keywords
            include:


            - **`h`** (*float* or *np.ndarray*): Specifies the finite
              difference step size. If an array is given, each element gives
              the step size for each parameter.
            - **`method`** (*str*): Specifies the finite difference method to
              use. Available methods are `"FD"`, `"FD2"`, `"FD3"`, `"FD4"`,
              `"CD"`, and `"CD4"`.

        :type derivative_kwargs: dict

        :param return_jacobian: If it is True, then the method returns both
            the FIM and Jacobian, respectively

        :type return_jacobian: bool

        :param nprocs: number of parallel processes to use when computing the
            columns of Jacobian.
        :type nprocs: int

        :returns: (P, P) array of the FIM, where P is the number of potential
            parameters
        :rtype: np.ndarray
        """
        result = self.compute_batch([target_property], score_quantity, [cov],
                                    potential, parameters_optimize, transform,
                                    derivative_kwargs, return_jacobian, nprocs,
                                    **kwargs)
        if return_jacobian:
            fim, jac_list = result
            return fim, jac_list[0]
        else:
            return result

    def compute_batch(self,
                      list_of_target_property: List[dict],
                      score_quantity: str,
                      cov: Union[str, np.ndarray, list[str], list[np.ndarray]],
                      potential: Union[dict, Potential],
                      parameters_optimize: dict,
                      transform: Optional[Union[dict, TransformBase]] = None,
                      derivative_kwargs: Optional[dict] = None,
                      return_jacobian: Optional[bool] = False,
                      nprocs: Optional[int] = 1,
                      **kwargs) -> list:
        """
        Runs the FIM calculation for a batch of atomic configurations. This is
        intended to be able to be used in a serial (non-distributed) manner,
        outside of a proper orchestrator scheduler.

        This method returns a list of FIMs, where the order of the list matches
        the order of the `list_of_target_property` input argument.
        Specifically, the first element corresponds to the FIM of the first
        target property, the second element to the second target property, and
        so on.

        .. note::

           If :class:`~orchestrator.target_property.kimrun.KIMRun` is used,
           then the argument `flatten=True` in
           :class:`~orchestrator.target_property.kimrun.KIMRun.calculate_property`
           method is enforced.


        :param list_of_target_property: a list of dictionaries, where each
            dictionary contains information about the target property, and
            should have the following keys:

            - **`init_args`**: Used to instantiate the target property class
            - **`calculate_property_args`**: Contains arguments for
              `TargetProperty.calculate_property` method, excluding the
              potential and `iter_num` , which are automatically inserted
              during this calculation.

        :type list_of_target_property: List[dict]

        :param score_quantity: The type of score value to compute. For this
            module, the accepted argument in "SENSITIVITY".
        :type score_quantity: str ("SENSITIVITY")

        :param cov: target covariance matrix of the target property to achieve.
            If a single matrix is given, then it is treated as the target
            covariance for combined target properties. If a list is given, each
            element gives the target covariance for each target property. Note
            that the later option should not be used if the target properties
            are correlated, as it cannot encode covariance across target
            properties
        :type cov: Path-like str (preferred) or np.ndarray, or a list
            of Path-like or np.ndarray

        :param potential: input dictionary to instantiate the potential (using
            init_potential) or the potential instance itself.
        :type potential: dict (preferred) or orchestrator.potential.Potential

        :param parameters_optimize: Potential parameters to differentiate and
            their values.
        :type parameters_optimize: dict

        :param transform: A dictionary containing information to instantiate
            parameter transformation class. Required keys are "transform_type"
            and "transform_args".
        :type transform: dict (preferred) or TransformBase

        :param derivative_kwargs: Additional arguments for instantiating
            `information_matching.fim.finitediff.FiniteDifference`, which are
            interpreted as the finite difference settings. Available keywords
            include:


            - **`h`** (*float* or *np.ndarray*): Specifies the finite
              difference step size. If an array is given, each element gives
              the step size for each parameter.
            - **`method`** (*str*): Specifies the finite difference method to
              use. Available methods are `"FD"`, `"FD2"`, `"FD3"`, `"FD4"`,
              `"CD"`, and `"CD4"`.

        :type derivative_kwargs: dict

        :param return_jacobian: If it is True, then the method returns both
            the FIM and Jacobian, respectively

        :type return_jacobian: bool

        :param nprocs: number of parallel processes to use when computing the
            columns of Jacobian.
        :type nprocs: 1

        :returns: A (P, P) array of the FIM, where P is the number of potential
            parameters. The FIMs from different target properties summed up
            to be the total FIM.
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

        # Deal with different format we allow for the potential argument
        # Additionally, check if the potential type is supported.
        if isinstance(potential, dict):
            # Check support
            if potential['potential_type'] in self.supported_potential_type:
                self._potential_init_args = potential
                # Instantiate the potential
                potential = self._init_potential(self._potential_init_args)
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
                self._potential_init_args = {
                    'potential_type': potential_type,
                    'potential_args': potential.args
                }
            else:
                # Potential is not supported
                raise FIMError(
                    'Potential is not supported for FIM calculation')
        self._potential = potential

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

        # Set parameters optimize -- This not only updates the parameters but
        # set a function that does conversion from parameter array to
        # dictionary.
        self._parameters_optimize = parameters_optimize
        self._set_parameters_optimize(parameters_optimize)
        self.fim_index_to_parameter = get_column_index_to_parameter_info(
            parameters_optimize)
        # Get the parameters that we will use to evaluate the derivative
        params0_raw = self._potential.get_params(
            **self._params_dict_retrieve).copy()
        params0 = self._convert_params_dict_to_array(params0_raw)
        transformed_params0 = self._parameters_transform(params0)

        # Setup finite difference derivative
        # Check the derivative settings
        if not derivative_kwargs:
            derivative_kwargs = {}
        self._fd = FiniteDifference(transformed_params0, **derivative_kwargs)
        self._params_set = self._generate_parameters_set()

        # One case we use this method is if the target predictions consist of
        # multiple target properties. It can be shown that the total FIM for
        # the target predictions is the sum of the FIM of individual property.
        if isinstance(cov, list):
            if len(cov) != len(list_of_target_property):
                raise FIMError(
                    'Please provide target covariance for each target property'
                )
        # iterate over the target properties
        list_of_jac = []
        for target_property in list_of_target_property:
            target_property_args = target_property['init_args']
            calculate_property_args = target_property[
                'calculate_property_args']
            # Compute Jacobian
            jac = self.compute_jacobian(target_property_args,
                                        calculate_property_args, nprocs)
            list_of_jac.append(jac)
        # Compute FIM
        fim = self.compute_fim(list_of_jac, cov)

        # Metadata
        self._metadata = {
            'potential': {
                'type': self._potential_init_args['potential_type'],
                'name': self._potential_init_args['potential_args']['kim_id']
            },
            'parameters_optimize': str(self._parameters_optimize),
            'transform': transform_meta,
            'derivative_kwargs': str(derivative_kwargs),
            'fim_index_to_parameter': self.fim_index_to_parameter,
            'target_property': [str(tp) for tp in list_of_target_property]
        }
        # Metadata for the target covariance, since we allow paths to numpy
        # files
        cov_metadata = []
        if isinstance(cov, (Path, str, np.ndarray)):
            # There is only 1 covariance matrix given
            cov = [cov]
        for c in cov:
            if isinstance(c, (Path, str)):
                cov_metadata.append(str(c))
            elif isinstance(c, np.ndarray):
                cov_metadata.append(PLACEHOLDER_ARRAY_KEY)
        self._metadata.update({'cov': cov_metadata})

        if return_jacobian:
            return fim, list_of_jac
        else:
            return fim

    def _set_parameters_optimize(self, parameters_optimize: dict):
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
        self._tunable_params_idx = []  # [[name, ext], [name, ext], ...]
        # We also need to save data about the parameters we set fix, in case if
        # we want to fix them to some non default values.
        fixed_params_idx = []  # [[name, ext, val], ...]
        for name, values in parameters_optimize.items():
            # Iterate over the values
            for ii, val in enumerate(values):
                if len(val) == 1:
                    # Only the mode is given -- There are 2 cases: "default" or
                    # value. In both cases, the parameters are tunable.
                    if val[0] != "default":
                        # Update parameter value
                        self._potential.set_params(**{name: [[ii], [val[0]]]})
                    # Append the tunable parameter information [name, idx]
                    self._tunable_params_idx.append([name, ii])
                elif len(val) == 2:
                    # First element is the value, and second element is the
                    # mode. The parameter is always fixed.
                    if val[0] != "default":
                        # Update parameter value
                        self._potential.set_params(**{name: [[ii], [val[0]]]})
                        # Append the fixed parameter information [name, idx] --
                        # Only append IF we don't use default value
                        fixed_params_idx.append([name, ii, val[0]])
        # After collecting the tunable parameters name and indices, let's
        # create a template dictionary to make it easier to generate parameter
        # dictionary later.
        self._params_dict_tpl = {}
        # For retrieving tunable parameters (without parameters we set fixed)
        self._params_dict_retrieve = {}
        # The template should contain placeholder for the tunable parameters
        for item in self._tunable_params_idx:
            name, idx = item
            if name not in self._params_dict_tpl:
                self._params_dict_tpl.update({name: [[], []]})
                self._params_dict_retrieve.update({name: []})
            self._params_dict_tpl[name][0].append(idx)  # Index
            self._params_dict_tpl[name][1].append(None)  # Value - placeholder
            # Update dictionary for retrieving parameters - We just need the
            # indices
            self._params_dict_retrieve[name].append(idx)
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
            name, idx = item
            loc = params_dict[name][0].index(idx)
            params_dict[name][1][loc] = val
        return params_dict

    def _convert_params_dict_to_array(self, params_dict):
        """
        Convert potential parameters from dictionary format ouput by ASE KIM
        calculator to an array.
        """
        params_array = np.zeros(len(self._tunable_params_idx))
        for ii, (name, idx) in enumerate(self._tunable_params_idx):
            ext_list, vals = params_dict[name]
            loc = np.where(np.array(ext_list) == idx)[0][0]
            params_array[ii] = vals[loc]
        return params_array

    def compute_fim(self, list_of_jac: list, cov: Union[str, np.ndarray,
                                                        list]) -> np.ndarray:
        """
        Compute the combined FIM of the target property given the lists of
        Jacobian and covariance matrices.

        :param list_of_jac: Jacobian matrix of the target property.
        :type list_of_jac: np.ndarray

        :param cov: Target covariance matrix or matrices.
        :type cov: str or np.ndarray or list

        :returns: total FIM for the target properties
        :rtype: np.ndarray

        """
        if isinstance(cov, (Path, str, np.ndarray)):
            # One single covariance matrix is given for the entire target
            # properties
            cov = self._read_cov(cov)
            cov_inv = np.linalg.pinv(cov)
            jac = np.vstack(list_of_jac)
            fim = jac.T @ cov_inv @ jac
        elif isinstance(cov, list):
            # One covariance matrix is given per target proeprty, ignoring
            # covariance across properties
            fim = []
            for jac, c in zip(list_of_jac, cov):
                c = self._read_cov(c)
                cov_inv = np.linalg.pinv(c)
                fim.append(jac.T @ cov_inv @ jac)
            fim = np.sum(fim, axis=0)
        return fim

    @staticmethod
    def _read_cov(cov: Union[str, np.ndarray]) -> np.ndarray:
        """
        Read the covariance matrix.
        """
        # If the cov is given as a path, then load
        if isinstance(cov, (np.ndarray, list)):
            covariance = np.array(cov)
        elif isinstance(cov, (Path, str)):
            covariance = np.load(cov)
        else:
            raise FIMError(
                'Unknown cov format, please provide either a numpy array or '
                'str pointing to a numpy file')
        # Final check of the dimension of the covariance
        if covariance.ndim != 2:
            raise FIMError('Covariance should be a 2D matrix')
        return covariance

    def _generate_parameters_set(self) -> list[dict]:
        """
        Generate a set of parameters we need to evaluate to compute the
        derivative of the potential.
        """
        tmp_params_set = self._fd.generate_params_set()
        # The key contains "-", which is not compatible with KIM ID naming
        # system. We need to replace "-" with "_"
        params_set = {
            key.replace('-', '_'): val
            for key, val in tmp_params_set.items()
        }
        return params_set

    def _compute_target_property(self, target_property: TargetProperty,
                                 calculate_property_args: dict,
                                 potential: Potential, direction: str,
                                 pnum: int) -> dict:
        """
        Run the target property calculation.

        :param target_property: the target property instance
        :type target_property: oschestrator.target_property.TargetProperty

        :param calculate_property_args: any additional arguments to be passed
            into `target_property.calculate_property()` method.
        :type calculate_property_args: dict

        :param potential: potential instance
        :type potential: orchestrator.potential.Potential

        :param direction: a string describing the parameter perturbation
            direction.
        :type direction: str

        :param pnum: process number, to distinguish one calculation from the
            other. This is mainly used so we can do parallelization.
        :type pnum: int

        :returns: (q,) np.ndarray of the predictions
        :rtype: np.ndarray
        """
        # Scheduler
        scheduler = calculate_property_args.pop('scheduler', None)
        # Storage
        storage = calculate_property_args.pop('storage', None)

        # Compute predictions - For now, we'll use iter_num to index the
        # process for parallelization. This might not work in the future or for
        # some target properties, e.g., melting point (however, parallelization
        # is still currently not supported currently for melting point).
        preds = target_property.calculate_property(iter_num=pnum,
                                                   potential=potential,
                                                   scheduler=scheduler,
                                                   storage=storage,
                                                   **calculate_property_args)
        return direction, self._convert_target_property_output(preds)

    @staticmethod
    def _convert_target_property_output(
            target_property_output: dict) -> np.ndarray:
        """
        Convert the raw `TargetProperty.calculate_property` output to a vector
        for Jacobian calculation.

        The target property output is a dictionary, and the key we want is
        "property_value". However, the value for this key can be a float, an
        array, or even a dictionary. This function handles the conversion.

        :param target_property_output: Output of the target property
            calculation
        :type target_property_output: dict

        :returns: (q,) np.ndarray of the target property values
        :rtype: np.ndarray
        """
        # Get the prediction values from the dictionary
        values = target_property_output['property_value']
        # Conversion
        if isinstance(values, float):  # Scalar value
            preds = np.array([values])
        elif isinstance(values, (list, np.ndarray)):
            values = np.array(values)  # Ensure np.array
            ndim = np.ndim(values)
            if ndim == 1:  # A vector
                preds = values
            else:
                preds = values.flatten()
        elif isinstance(values, dict):
            raise FIMError(
                "Conversion from dictionary hasn't been implemented")
        return preds

    def _function_parallel_wrapper(self, args: dict) -> dict:
        """
        Wrapper function to evaluate target property for parallelization.

        :param args: a single element of the iterable list. The order of the
            element is:
            * target_property - target property instance
            * calculate_property_args - keyword arguments for
                `target_property.calculate_property()`
            * potential - potential instance
            * direction - direction of perturbation
            * pnum - process number
        :type args: list
        """
        target_property, calc_prop_args, potential, direction, pnum = args
        try:
            pred = self._compute_target_property(target_property,
                                                 calc_prop_args, potential,
                                                 direction, pnum)
        except Exception as e:
            # Just give information about perturbation direction. This is more
            # human readable compared to the entire parameter array.
            raise FIMError(
                'Target property calculation failed, last evaluation: '
                + f'{direction}' + '\n' + e)

        return pred

    def compute_jacobian(self,
                         target_property_args: dict,
                         calculate_property_args: Optional[dict] = {},
                         nprocs: Optional[int] = 1) -> np.ndarray:
        """
        Compute the Jacobian matrix for 1 target property.

        :param target_property: a dictionary for building the target property
            istance
        :type target_property: dict

        :param calculate_property_args: any additional arguments to be passed
            into `target_property.calculate_property()` method.
        :type calculate_property_args: dict

        :param nprocs: number of parallel processes to use when computing the
            Jacobian.
        :type nprocs: 1

        :returns: (q, p) np.ndarray of the Jacobian, where q and p are numbers
            of predictions and parameters, respectively.
        :rtype: np.ndarray
        """

        # Instantiate the target_property
        target_property = target_property_builder.build(
            target_property_args['target_property_type'],
            target_property_args['target_property_args'])

        # If the target property is KIMRun, then we need to save the perturbed
        # potentials into kimkit. The following check determine if this step
        # is needed
        if target_property_args['target_property_type'] == 'KIMRun':
            need_to_save_potential_to_kimkit = True
            # Next, we need to enforce returning the flatten property
            # prediction.

            if 'flatten' not in calculate_property_args:
                calculate_property_args.update({'flatten': True})
            elif calculate_property_args['flatten'] is not True:
                print('Overwrites the `flatten` option to enforce KIMRun '
                      'to return flattened values.')
                calculate_property_args['flatten'] = True
        else:
            need_to_save_potential_to_kimkit = False

        # Before running the target property calculations, I think we should
        # use a single scheduler. At least, I noticed that if we create
        # different scheduler for every calculation instance, the scheduler
        # counter always resets and we cannot update the potential
        modified_args = calculate_property_args.copy()
        # Scheduler
        scheduler_inputs = modified_args.get('scheduler')
        if scheduler_inputs:
            if 'root_directory' not in scheduler_inputs['scheduler_args']:
                scheduler_inputs['scheduler_args']['root_directory'] = (
                    self.default_target_property_scheduler.root_directory)
            modified_args['scheduler'] = scheduler_builder.build(
                scheduler_inputs['scheduler_type'],
                scheduler_inputs['scheduler_args'])
        else:
            modified_args['scheduler'] = self.default_scheduler
        self._target_property_scheduler = modified_args['scheduler']
        # Storage
        storage_inputs = modified_args.get('storage')
        if storage_inputs:
            modified_args['storage'] = storage_builder.build(
                storage_inputs['storage_type'], storage_inputs['storage_args'])
        else:
            modified_args['storage'] = None

        # Iterate over parameter items and compute the predictions using all
        # these parameters. These loop should be parallelizable.
        # Update the parameters, write the potential, and install it. This
        # is not parallelizable, since we need to build the potential, which
        # involve cd operation.
        potential_list = list(
            map(self._write_and_install_potential, self._params_set.items()))
        self.potential_list = potential_list
        # Building the potential needs to be run in serial, because it involves
        # changing directory.
        [self._build_potential(p) for p in potential_list]

        if need_to_save_potential_to_kimkit:
            # This is an additional step if we need to save the potential to
            # kimkit. Note that this steps overwrites potential_list from a
            # list of Potential instance to a list of string, which is what
            # KIMRun uses.
            potential_list = [
                self._save_potential_to_kimkit(potential)
                for potential in potential_list
            ]

        # Run target property calculation -- We should also do this in parallel
        # Prepare the iterables
        iterable = list()
        for pnum, direction in enumerate(self._params_set):
            iterable.append((target_property, modified_args,
                             potential_list[pnum], direction, pnum))
        # Target property calculations
        if nprocs > 1:
            with ThreadPoolExecutor(max_workers=nprocs) as executor:
                preds = list(
                    executor.map(self._function_parallel_wrapper, iterable))
        else:
            preds = list(map(self._function_parallel_wrapper, iterable))
        # Convert the format of predictions from list of dict to dict, so that
        # we can use the internal method of `FiniteDifference` to estimate
        # the derivative
        predictions_set = {}
        for pred_item in preds:
            key, val = pred_item
            # Note that we need to convert "_" back to "-" in the key
            predictions_set.update({key.replace('_', '-'): val})

        # Estimate the derivative of the potential
        jac = self._fd.estimate_derivative(predictions_set)
        # If we use KIMRun, should we catch an exception and clean kimkit?

        if need_to_save_potential_to_kimkit:
            # After all the calculations, clean up the perturbed potentials
            # from kimkit
            [kimkit.models.delete(potential) for potential in potential_list]

        return jac

    def _write_and_install_potential(self, param_item: list) -> Potential:
        """
        Write the perturbed potential and install it.

        Since this method creates its own copy of the potential, it should be
        save to run in parallel.
        """
        # Prepare the copy of potential -- We need to change the kim_id later
        potential_dict = deepcopy(self._potential_init_args)
        potential = self._init_potential(potential_dict)
        # We are still using KLIFF partially
        self._build_potential(potential)

        # Update potential parameter values
        direction, transformed_parameters = param_item
        parameters = self._parameters_inverse_transform(transformed_parameters)
        # We are still using KLIFF partially
        potential.model.set_opt_params(**self._parameters_optimize)
        potential.model.update_model_params(parameters)

        # Generate kim_id
        kim_id = potential.generate_new_kim_id('fim_property_potential')
        # Write potential file
        target_dir = os.path.join(self.potential_dir, kim_id)
        potential.model.write_kim_model(target_dir)
        # model_driver is a required arguments. But, param_files is
        # not required. If we use potential's internal function to retrieve the
        # param_files, it will direct the param_files to the original
        # potential, not the perturbed potential. But, user can specify
        # param_files and we should prioritize this option.
        if "param_files" in self._potential_init_args["potential_args"]:
            potential.param_files = self._potential_init_args[
                "potential_args"]["param_files"]
        else:
            potential.param_files = self._get_param_files(potential)
        # Install potential
        potential.install_potential_in_kim_api(potential_name='kim_potential',
                                               save_path=self.potential_dir,
                                               import_into_kimkit=False)
        return potential

    def _build_potential(self, potential: Potential):
        """
        Build the potential.

        This is designed to be run in series, due to chdir command that use
        relative path.
        Since the potential is not written in the current directory, then it
        needs to build it in the same directory as where the potential is
        written, because we use install locality CWD.
        """
        os.chdir(self.potential_dir)
        potential.build_potential()
        os.chdir(self._cwd)

    def _save_potential_to_kimkit(self, potential: Potential):
        """
        A manual work to save potential to kimkit.

        Note: In the future, I think we will deprecate this method, since
        KIMRun can already supoprt Potential instance with
        `save_potential_to_kimkit` functionality. But, with the current
        Potential class, we need to manually specify model driver and
        (parameter files,). This function automate that process.
        """
        potential_name = potential.kim_id
        # Retrieve the model driver, and parameter files
        model_driver = self._potential_init_args["potential_args"][
            "model_driver"]
        param_files = self._get_param_files(potential)
        # We now are ready to save the new potential to kimkit

        potential_name = potential.save_potential_files(
            kim_id=potential.kim_id,
            param_files=param_files,
            model_driver=model_driver,
            work_dir=self.potential_dir)
        return potential_name

    def _get_param_files(self, potential: Potential):
        """
        Retrieve the parameter files from the potential.

        The parameter files are retrieved from the CMakeLists.txt file. Thus,
        this method assumes that the potential has been written to a file.
        """
        potential_name = potential.kim_id
        # This is the path where this specific potential is written
        potential_path = os.path.join(self.potential_dir, potential_name)
        # Retrieve parameter files from reading CMakeLists.txt
        cmakelists_file = os.path.join(potential_path, "CMakeLists.txt")
        with open(cmakelists_file, "r") as f:
            cmakelists = f.read()
        # Retrieve parameter files
        params_match = re.findall(r'PARAMETER_FILES\s*((?:\s*"[^"]+")+)',
                                  cmakelists)
        param_files = []
        if params_match:
            for match in params_match:
                param_files.extend(re.findall(r'"([^"]+)"', match))
        # Append the potential directory
        param_files = [os.path.join(potential_path, p) for p in param_files]

        return param_files

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

    def read_data(self, read_path: str, **kwargs) -> list[dict]:
        """
        Read the data from a file.
        """
        # Read the json file
        with open(read_path, 'r') as f:
            data_raw = json.load(f)

        # The data can contain a single or multiple target properties
        # If there is only one target property, it will be stored as a
        # dictionary. If there are multiple target properties, it will be
        # stored as a list of dictionary.
        if isinstance(data_raw, list):  # Multiple target properties
            data = data_raw
        elif isinstance(data_raw, dict):  # A single target property
            data = [data_raw]
        return data

    def write_data(self, save_path, data, **kwargs):
        """
        Write the data to a file.
        """
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)

    def get_colabfit_property_definition(self) -> dict:
        return {
            'property-name': self.OUTPUT_KEY,

            'property-title': 'FIM target property',

            'property-description': 'The FIM of the target property.',

            # the fields that make up the descriptor
            'score': {  # example: (P, P) array
                'type': 'float',
                'has-unit': True,
                'extent': [':'],
                'required': True,
                'description': 'The Fisher information matrix for the target '
                'property, encoding the target covariance of the target '
                'property as well. The unit depends on the target property.',
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
            'target_property': {  # JSON encoding of a dict, or None
                'type': 'string',
                'has-unit': False,
                'extent': [':'],
                'required': True,
                'description': 'A list of JSON encoding of dictionary of '
                'describing the target property calculation.'
            },
            'cov': {  # example: (Q, Q) array
                'type': 'float',
                'has-unit': True,
                'extent': [':'],
                'required': True,
                'description': 'The target covariance of the target property. '
                'The unit depends on the target property.',
            },
        }
