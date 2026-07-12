from abc import abstractmethod
from datetime import datetime
from typing import Union, Optional
import pathlib
import ase
import json
import os
import numpy as np
from copy import deepcopy
from ase import Atoms

from ...scheduler import Scheduler
from ...storage import Storage
from ...utils.data_standard import METADATA_KEY
from ...utils.isinstance import isinstance_no_import
from ..oracle_base import Oracle
from aiida import load_profile
from aiida.orm import load_code, KpointsData, Dict, Group, load_group, \
    load_node
from aiida.plugins import WorkflowFactory
from aiida.common.exceptions import NotExistent
from aiida.orm.nodes.data.code.installed import InstalledCode
from aiida.engine.processes.builder import ProcessBuilder
from aiida.engine.processes.workchains.workchain import WorkChain
from aiida.orm.computers import Computer


class AiidaOracle(Oracle):
    """
    Abstract base class for ground truth calculations.

    The oracle class manages the construction and parsing of ground truth
    calculations to provide training or reference data. The input will
    typically consist of an atomic configuration and calculation parameters,
    the output will include the energy of the system, forces on each atom,
    and/or the stress on the cell.
    """

    def __init__(self,
                 code_str: str,
                 workchain: str,
                 clean_workdir: bool = True,
                 group: str = None,
                 **kwargs: dict):
        """
        Set variables and initialize the recorder and default scheduler.

        :param code_str: Name of the code in the AiiDA database.
        :param workchain: Name of the workchain in AiiDA.
        :param clean_workdir: Will clean the working directory on the remote
            machine if True.
        :param group: Creates a group node in AiiDA to store all of the
            calculations for easy parsing afterwards based on the string name.
        """

        super().__init__(**kwargs)

        self.code_str = code_str
        if self.code_str is None:
            raise KeyError(
                'The code string representing the binary in the AiiDA '
                'framework must be provided and is required to instantiate a '
                'specific AiidaOracle.')
        self.workchain = workchain
        if self.workchain is None:
            raise KeyError(
                'The workchain specifying the scheduler in the AiiDA framework'
                ' must be provided and is required to instantiate a specific '
                'AiidaOracle.')
        self.clean_workdir = clean_workdir

        if group:
            try:
                self.group = load_group(group)
            except NotExistent:
                self.group = Group(
                    label=group,
                    description=(
                        f'Created by the {self.__class__.__name__} module.'))
                self.group.store()
        else:
            self.group = None

    def run(
        self,
        path_type: str = None,
        input_args: Union[dict, pathlib.Path, None] = None,
        configs: list[ase.Atoms] = None,
        scheduler=None,
        job_details=None,
    ) -> list[int]:
        """
        Main function to call to run ground truth calculation.

        The run method includes half of the main functionality of the oracle,
        taking atomic configurations as input and handling the submission of
        calculations to obtain the ground truth data. Configs is a dataset of 1
        or more structures. run() will create independent jobs for
        each structure using the supplied scheduler, with job_details
        parameterizing the job submission.

        :param path_type: Specifier for the scheduler path, to differentiate
            calculation types. Legacy input that will be removed in the future.
        :param input_args: Input arguments to fill out the input file.
        :param configs: List of configurations as ASE atoms to run ground truth
            calculations.
        :param scheduler: The scheduler for managing job submission, if none
            are supplied, will use the default scheduler defined in this class
            |default| ``AiiDAScheduler``.
        :param job_details: Dict that includes any additional parameters for
            running the job (passed to :meth:`~.Scheduler.submit_job`)
            |default| ``None``
        :returns: a list of calculation IDs (pks) from the scheduler.
        """

        module_name = self.__class__.__name__
        calc_ids = []
        self.logger.info(
            f'Spinning up {len(configs)} {module_name} calculations')

        self.kspacing = input_args.pop('kspacing', None)
        self.kpoints = input_args.pop('kpoints', None)
        self.potential_family = input_args.pop('potential_family', None)
        if self.potential_family is None:
            raise KeyError('The potential family must be set for the specific '
                           'AiidaOracle you are attempting to use.')
        self.potential_mapping = input_args.pop('potential_mapping', None)
        self.parameters = input_args.pop('parameters', None)
        self.parameters = self.get_parameters(self.parameters)
        if self.parameters is None:
            raise KeyError(
                'The parameters arg is required to provide inputs to the '
                'specified AiidaOracle.')

        # If any other values are in input_args, set them as well.
        # Will be able to set other specific values needed for aiida oracles.
        input_copy = input_args.copy()
        for key in input_copy.keys():
            setattr(self, key, input_args.pop(key))

        # Get the builder from the workchain
        workchain = self.get_workchain(self.workchain)

        if scheduler:
            if not isinstance_no_import(scheduler, 'AiidaScheduler'):
                raise TypeError(
                    'When using AiiDA type Oracles, the AiidaScheduler must '
                    'be specified.')
            else:
                self.scheduler = scheduler
        elif self.scheduler is None:
            raise RuntimeError('AiiDA run must explicitly provide a scheduler')

        if job_details is None:
            job_details = {}

        for config in configs:

            modified_job_details = deepcopy(job_details)
            if config.info.get(METADATA_KEY) is not None:
                try:
                    modified_job_details['extra_args'][
                        METADATA_KEY] = config.info[METADATA_KEY]
                except KeyError:
                    modified_job_details['extra_args'] = {
                        METADATA_KEY: config.info[METADATA_KEY]
                    }

            builder = self._oracle_specific_inputs(workchain, config,
                                                   modified_job_details)

            if 'clean_workdir' in dir(builder):
                builder.clean_workdir = self.clean_workdir

            pk = scheduler.submit_job(builder, modified_job_details)
            if self.group:
                calc = load_node(pk)
                self.group.add_nodes(calc)

            calc_ids.append(pk)
            self.logger.info(f'Done generating {module_name} calculations')

        return calc_ids

    def save_labeled_configs(
        self,
        pks: list[int],
        storage: Storage,
        dataset_name: Optional[str] = None,
        dataset_handle: Optional[str] = None,
        scheduler: Optional[Scheduler] = None,
    ) -> str:
        """
        Extract and save computed data to storage

        The save_labeled_configs method includes the other half of the main
        functionality of the Oracle. Once the calculations are complete, the
        data they generate must be integrated with the strucutral configuration
        in a consistent framework to be used for training. This is done by
        retrieving the nodes in the AiiDA database with the saved pk values
        and can then be inserted into a dataset handled by the
        :class:`~orchestrator.storage.storage_base.Storage` module.

        :param pks: AiiDA pks associated with each config.
        :param storage: Specific module that handles the staroge of data.
        :param dataset_name: The name of the datset in the Storage where the
            configurations should be saved. If ``None``, then the class default
            (date stamped) is used. |default| ``None``.
        :param dataset_handle: The handle for the particular dataset that will
            be modified with additional configurations. If ``None``, then the
            ``dataset_name`` will be used.
        :param scheduler: The scheduler for managing job submission. If not
            specified, will use the default scheduler defined in this class.
            Should be consistent with the scheduler supplied for the run calls.
            |default| ``None``.
        :returns: dataset handle
        """

        if scheduler is None:
            raise RuntimeError(
                'AiiDA save must explicitly provide a scheduler')

        if isinstance(pks, int):
            pks = [pks]
        if not isinstance(pks[0], int):
            try:
                pks = np.array(pks).astype(int)
            except ValueError:
                raise ValueError(
                    'Supplied pks are not in a recognized format.')

        self.logger.info((f'Labelling {len(pks)} '
                          f'{self.__class__.__name__} calculations'))

        # parsed data is a list of atoms object from the Oracle and input
        # parameters that include code specific and universal values.
        configs, parameters = self.data_from_calc_ids(calc_ids=pks,
                                                      scheduler=scheduler)

        current_date = datetime.today().strftime('%Y-%m-%d')
        dataset_metadata = {
            'description': (f'data generated by {self.__class__.__name__} on '
                            f'{current_date}'),
            'parameters':
            parameters
        }

        if dataset_name is None and dataset_handle is None:
            dataset_name = storage.generate_dataset_name(
                f'{self.__class__.__name__}_dataset',
                f'{current_date}',
                check_uniqueness=True,
            )
        elif dataset_name:
            unique = storage.check_if_dataset_name_unique(dataset_name)
            if not unique:
                dataset_handle = storage._get_id_from_name(dataset_name)
        else:
            if dataset_handle[:3] != 'DS_':
                raise ValueError(
                    'dataset handles should be in format DS_************_#')

        # this logic assumes colabfit style naming conventions for dataset IDs
        # if other storage is implemented, should switch to a more generic
        # "if dataset exists" logic check
        if dataset_handle:
            # handle is a colabfit ID, dataset exists
            new_handle = storage.add_data(dataset_handle, configs,
                                          dataset_metadata)
        else:
            # handle is a name, create new dataset
            new_handle = storage.new_dataset(dataset_name, configs,
                                             dataset_metadata)

        return new_handle

    def get_run_command(self):
        """
        """
        raise NotImplementedError('AiiDA oracles do not need a run command.')

    def write_input(self):
        """
        """
        raise NotImplementedError('AiiDA will write the input file.')

    def get_kpoints(self,
                    kpoints: list[int] = None,
                    kspacing: float = None) -> KpointsData:
        """
        Initialize the kpoint mesh based on either a kpoints uniform mesh
        described by a list or through the spacing between kpoints in
        reciprocal space. Can set either kpoints or kspacing inside the
        input file.
        :param kpoints: List of specified kpoints.
        :param kspacing: Value of the kspacing between points in reciprocal
            space

        :returns: KpointsData object that is recognized by AiiDA.
        """

        if kpoints and isinstance(kpoints, list):
            kpoints = KpointsData().set_kpoints_mesh(kpoints)
        elif kspacing:
            kpoints = KpointsData()
            kpoints.set_cell_from_structure(self.structure)
            kpoints.set_kpoints_mesh_from_density(kspacing, force_parity=False)
        else:
            raise ValueError(
                'No value was specified for kpoints or kspacing in the '
                'oracle_args.')

        return kpoints

    def get_code(self, code_str: str) -> InstalledCode:
        """
        Retrieve the code from the AiiDA database.
        :param code_str: String representation of the code installed in AiiDA.

        :returns: AiiDA code object.
        """
        load_profile()
        if code_str is None:
            if self.code_str is None:
                self.logger.info('Job will not run: no code_str')
                raise TypeError('Job will not run: no code_str')
            else:
                code_str = self.code_str
        return load_code(code_str)

    def get_workchain(self, workchain: str) -> WorkChain:
        """
        Retrieve the specified workchain for the AiiDA scheduler.
        :param workchain: Name of the scheduler for the specified code.

        :returns: Workchain object from AiiDA.
        """
        if workchain is None:
            raise TypeError("No workchain was specified in the oracle args.")

        workchain = WorkflowFactory(workchain)

        return workchain

    def get_parameters(self, parameters: Union[str, dict]) -> Dict:
        """
        Retrieve the provided input parameters for the oracle.
        :param parameters: Can be either a string point to a path of a json
            file or a dictionary from the input file.

        :returns: Input parameters required for an AiiDA calculation.
        """

        if isinstance(parameters, str) and os.path.exists(parameters):
            with open(parameters, 'r') as file:
                parameters = json.load(file)

        if parameters:
            parameters = Dict(dict=parameters)
        else:
            raise ValueError('No input parameters were provided.')

        return parameters

    def set_other_oracle_args(self, builder: ProcessBuilder,
                              other_args: Dict) -> ProcessBuilder:
        """
        There exists the possibility to change all of the input values for
        an AiiDA calculation. There are certain input values that will be
        handled specifically by the individual oracles. The rest of the values
        can be set by this function.

        :param builder: Builder object from AiiDA that stores input values.
        :param other_args: Values that were not explicitly set by the oracle.
        :returns: Builder object
        """

        for k, v in other_args.items():
            # Attribute exists
            if k in dir(builder):
                setattr(builder, k, v)
            else:
                self.logger.info(
                    f'The following oracle arg is not recognized by the '
                    f'workchain: {k}')

        return builder

    def get_options(self, job_details: dict, computer: Computer) -> dict:
        """
        Retrieves options for AiiDA job submission.

        Converts the job details from the Orchestrator to an option dictionary
        for an AiiDA job submission.

        :param job_details: Dictionary of job details needed to launch a
            calculation.
        :param computer: AiiDA computer instance
        :returns: Dictionary of options for AiiDA job submission.
        """

        options, resources = {}, {}

        extra_args = job_details.get('extra_args', None)

        if computer.transport_type == 'core.local':
            resources['tot_num_mpiprocs'] = job_details.get(
                'tasks', self.scheduler.default_tasks)
            options['resources'] = resources

        else:
            options['account'] = job_details.get(
                'account', self.scheduler.default_account)
            options['queue_name'] = job_details.get(
                'queue', self.scheduler.default_queue)
            if extra_args:
                options['append_text'] = extra_args.get('preamble', '')
                options['prepend_text'] = extra_args.get('postamble', '')

            resources['num_machines'] = job_details.get(
                'nodes', self.scheduler.default_nodes)
            resources['tot_num_mpiprocs'] = job_details.get(
                'tasks', self.scheduler.default_tasks)
            options['max_wallclock_seconds'] = job_details.get(
                'walltime', self.scheduler.default_walltime) * 60
            options['resources'] = resources
            options['qos'] = job_details.get('qos', self.scheduler.default_qos)

        return options

    @abstractmethod
    def _oracle_specific_inputs(
        self,
        workchain: WorkChain,
        config: Atoms,
        job_details: dict,
    ) -> ProcessBuilder:
        """
        Specify code specific input values.

        Each code in AiiDA might need additional input values to successfully
        submit a workchain.

        :param workchain: AiiDA WorkChain object that will be used to create
            the builder for submitting the job.
        :param config: ASE Atoms object of current configuration.
        :param job_details: Specific job submission information for the oracle.
        :returns: AiiDA builder object with job submission details.
        """

    @staticmethod
    @abstractmethod
    def translate_universal_parameters(self, parameters: dict) -> dict:
        """
        Orchestrator has predefined universal input values for varying codes
        to allow some transferability. Each Oracle will need a function to
        translate those values from the specific code.

        :param parameters: Dictionary containing all the input parameters to
            run the simulation.
        :returns: Dictionary of universal input parameters for database
            storage.
        """
