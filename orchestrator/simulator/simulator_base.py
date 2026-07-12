from abc import ABC, abstractmethod
from ase import Atoms
from datetime import datetime
from random import randrange, seed
from glob import glob
from os.path import basename, join, isdir, isfile
import shutil
from typing import Any, Callable, Optional, Union
from ..storage import Storage
from ..scheduler import Scheduler
from ..scheduler.factory import scheduler_builder
from ..utils.recorder import Recorder
from ..utils.input_output import ase_glob_read


class Simulator(Recorder, ABC):
    """
    Abstract base class to manage and run simulations (exploration)

    The simulator class manages the construction and parsing of molecular
    dynamics calculations using interatomic potentials. The input will
    typically consist of an initial atomic configuration and calculation
    parameters (including the potential to use), while the output will include
    frames or configurations from the simulation as well as information such as
    the energy of the system, forces on each atom, and/or the stress on the
    cell, amongst others.

    :param input_template: path to an input template to build from
    :param kwargs: additional keyword arguments for extensibility
    """

    def __init__(
        self,
        input_template: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Abstract base class to manage and run simulations

        :param input_template: path to an input template to build from
        :param kwargs: additional keyword arguments for extensibility
        """
        super().__init__()

        if input_template is None:
            self.logger.info('No input_template set, one must be provided to '
                             'simulator.run()')
        self.input_template = input_template

        #: default scheduler to use within the Simulator class
        self.default_scheduler = scheduler_builder.build(
            'LOCAL',
            {'root_directory': './simulator'},
        )
        # this flag should be set to True externally
        self.external_setup: bool = False
        # if external_setup is set to True, then external_func needs to be set
        # as a function
        self.external_func: Optional[Callable] = None

    def run(
        self,
        path_type: str,
        simulation_files: Union[str, list[str], None],
        template_fill: dict[str, Any],
        make_config_path: Optional[str] = None,
        make_config_storage: Optional[Storage] = None,
        make_config_handle: Optional[str] = None,
        make_config_atoms: Optional[Union[Atoms, list[Atoms]]] = None,
        make_config_seed: Optional[int] = None,
        input_template: Optional[str] = None,
        scheduler: Optional[Scheduler] = None,
        job_details: Optional[dict[str, Any]] = None,
    ) -> int:
        """
        Setup and execute a Simulator calculation

        Prepare input file and initial configuration. Execute the code (run
        simulation), returning the ``calc_id`` for tracking purposes. If none
        of `make_config_path`, `make_config_storage`, or `make_config_atoms`
        are set, then the template should construct the simulation box.

        :param path_type: specifier for the scheduler path, to differentiate
            calculation types
        :type path_type: str
        :param simulation_files: files or directories that are necessary for
            the simulation to run
        :type simulation_files: str or list[str]
        :param template_fill: values to fill out the input template file,
            passed to :meth:`_write_input`
        :type template_fill: dict
        :param make_config_path: if a separate configuration/structure file
            should be written based on structures read from disk at the path.
            Will be read with :meth:`_get_init_configs_from_path`. Cannot be
            used with `make_config_storage` or `make_config_atoms`. If the path
            contains multiple configurations, the first config will be used,
            unless `make_config_seed` is also set, in which case a random
            structure will be chosen with that seed.
        :type make_config_path: str
        :param make_config_storage: if a separate configuration/structure file
            should be written based on structures read from storage.
            `make_config_handle` must also be provided. Cannot be used with
            `make_config_path` or `make_config_atoms`. If the dataset contains
            multiple configurations, the first config will be used, unless
            `make_config_seed` is also set, in which case a random structure
            will be chosen with that seed.
        :type make_config_storage: Storage
        :param make_config_handle: Used with `make_config_storage` to specify
            the dataset that the initial structure should be taken from
        :type make_config_handle: str
        :param make_config_atoms: list of Atoms passed in memory to select
            from for generating the inital configuration
        :type make_config_atoms: Atoms or list of Atoms
        :param make_config_seed: random seed to use to select an initial
            configuration from a list. If not used, the first configuration
            will be selected.
        :type make_config_seed: int
        :param input_template: input template to use (overriding the default
            possibly set at instantiation). Will raise a RuntimeError if
            neither are set.
        :type input_template: str
        :param scheduler: the scheduler for managing job submission. If None,
            uses default scheduler |default| ``None``
        :type scheduler: Scheduler
        :param job_details: dict that includes any additional parameters for
            running the job (passed to
            :meth:`~.scheduler_base.Scheduler.submit_job`)
            |default| ``None``
        :type job_details: dict
        :returns: calculation ID
        :rtype: int
        """
        # validate input
        if input_template is None:
            if self.input_template is None:
                raise RuntimeError('input_template must be set at '
                                   'Simulator construction or .run()')
            else:
                input_template = self.input_template
        self.logger.info(f'Using input template: {input_template}')

        structure_pool = []
        if make_config_path is not None:
            if (make_config_storage is not None
                    or make_config_atoms is not None):
                raise ValueError('Only one of make_config_path, '
                                 'make_config_storage or make_config_atoms '
                                 'can be used at a time')
            # read from file
            structure_pool = self._get_init_configs_from_path(make_config_path)
        elif make_config_storage is not None:
            if make_config_path is not None or make_config_atoms is not None:
                raise ValueError('Only one of make_config_path, '
                                 'make_config_storage or make_config_atoms '
                                 'can be used at a time')
            # read from storage
            structure_pool = make_config_storage.get_data(make_config_handle)
        elif make_config_atoms is not None:
            if make_config_path is not None or make_config_storage is not None:
                raise ValueError('Only one of make_config_path, '
                                 'make_config_storage or make_config_atoms '
                                 'can be used at a time')
            # read from memory
            if isinstance(make_config_atoms, Atoms):
                structure_pool = [make_config_atoms]
            elif (isinstance(make_config_atoms, list)
                  and isinstance(make_config_atoms[0], Atoms)):
                structure_pool = make_config_atoms
            else:
                raise ValueError('make_config_atoms must either be Atoms or a '
                                 'list of Atoms')

        module_name = self.__class__.__name__
        if scheduler is None:
            scheduler = self.default_scheduler
        if job_details is None:
            job_details = {}
        run_path = scheduler.make_path(module_name, path_type)

        self._load_simulation_files(run_path, simulation_files)

        if len(structure_pool) > 0:
            self.logger.info(f'{module_name} is creating the configuration(s)')
            if make_config_seed is not None:
                self.logger.info(
                    f'Initializing random seed with seed: {make_config_seed}')
                seed(make_config_seed)
            ind = randrange(0, len(structure_pool))
            self.logger.info(f'Using random index: {ind}')
            self._write_initial_config(run_path, structure_pool[ind])
        else:
            self.logger.info('Structure not generated, it should be set by '
                             ' the input template')

        input_file_name = job_details.get('input_file_name')
        self._write_input(
            run_path,
            input_template,
            template_fill,
            input_file_name,
        )

        if self.external_setup:
            self._external_calculation_setup(run_path)

        simulator_command = self._get_run_command(job_details)
        calc_id = scheduler.submit_job(simulator_command, run_path,
                                       job_details)

        return calc_id

    def save_configurations(
        self,
        path_ids: Union[list[Union[int, str]], Union[int, str]],
        storage: Storage,
        dataset_handle: Optional[str] = None,
        scheduler: Optional[Any] = None,
    ) -> str:
        """
        save the configurations associated with path_ids to storage

        :param path_ids: list of ``calc_ids`` or explicit paths associated with
            simulator jobs. If ``calc_ids`` are supplied, the path is extracted
            from the :class:`~orchestrator.scheduler.scheduler_base.JobStatus`.
            Otherwise it is taken verbatim as the input.
        :type path_ids: list of int or str
        :param storage: the storage module where the configurations will be
            saved.
        :type storage: Storage
        :param dataset_handle: the handle to identify where in Storage the
            configurations should be saved. If ``None``, then the class default
            (date stamped) is used. |default| ``None``
        :type dataset_handle: str
        :param scheduler: the scheduler for managing job submission, if none
            are supplied, will use the default scheduler defined in this class.
            Should be consistent with the scheduler supplied for any run calls.
            |default| ``None``
        :type scheduler: Scheduler
        :returns: handle of the dataset which includes the new configurations
        :rtype: str
        """
        if not isinstance(path_ids, list):
            path_ids = [path_ids]

        if scheduler is None:
            scheduler = self.default_scheduler

        # Use scheduler method to resolve calc_ids or paths
        data_paths, _existing_metadata = scheduler.resolve_calc_paths(
            path_ids, allow_paths=True)

        self.logger.info((f'Saving {len(data_paths)} '
                          f'{self.__class__.__name__} trajectories'))

        data = []
        for run_path in data_paths:
            data.extend(self.parse_for_storage(run_path))

        current_date = datetime.today().strftime('%Y-%m-%d')
        dataset_metadata = {
            'description': (f'data generated by {self.__class__.__name__} on '
                            f'{current_date}')
        }

        if dataset_handle is None:
            dataset_handle = storage.generate_dataset_name(
                f'{self.__class__.__name__}_dataset',
                f'{current_date}',
                check_uniqueness=True,
            )

        # this logic assumes colabfit style naming conventions for dataset IDs
        # if other storage is implemented, should switch to a more generic
        # "if dataset exists" logic check
        if dataset_handle[:3] == 'DS_':
            # handle is a colabfit ID, dataset exists
            new_handle = storage.add_data(dataset_handle, data,
                                          dataset_metadata)
        else:
            # handle is a name, create new dataset
            new_handle = storage.new_dataset(dataset_handle, data,
                                             dataset_metadata)

        return new_handle

    def _external_calculation_setup(self, path: str) -> None:
        """
        Utility function to call an attached external function for input setup

        If self.external_setup is set to ``True``, then this method will be
        called. The external code which set the setup flag to True should also
        set the external_func to the desired function. It should take in the
        path to write output as its only parameter.

        :param path: location where input files should be written, passed to
            the attached external_func
        :type path: str
        """
        if callable(self.external_func):
            self.external_func(path)
        else:
            raise AttributeError('Set external_func to a callable function!')

    def _get_init_configs_from_path(
        self,
        config_path: str,
        file_ext: str = '.xyz',
        file_format: str = 'extxyz',
    ) -> list[Atoms]:
        """
        Read the initial configuration for the simulator input from path

        Loads the configurations present in the ``config_path`` and all of its
        sub-directories into a list of ASE Atoms.

        :param config_path: path of the root directory where configuration
            files are stored
        :type config_path: str
        :param file_ext: file extension |default| ``'.xyz'``
        :type file_ext: str
        :param file_ext: file format |default| ``'extxyz'``
        :type file_ext: str
        :returns: dataset as list of Atoms
        :rtype: list of Atoms
        """
        return ase_glob_read(config_path, file_ext, file_format, index=':')

    def _load_simulation_files(
        self,
        run_path: str,
        simulation_files: Union[str, list[str], None],
    ) -> None:
        """
        Ensure files needed for the simulation are provided in run directory

        Make the trained model accessible for simulations, i.e. through loading
        a KIM potential or ensuring the potential files are present in the
        requisite folder. If none are provided, then simulation should be able
        to run without any additional files present in the working directory.

        :param run_path: root path where simulations will run and potential
            should be loaded/copied
        :type run_path: str
        :param simulation_files: files or directories that are necessary for
            the simulation to run
        :type simulation_files: str
        """
        if simulation_files is None:
            self.logger.info('Simulation files not provided, simulation '
                             'should be able to run without any file input')
        else:
            if not isinstance(simulation_files, list):
                simulation_files = [simulation_files]
            for simulation_file in simulation_files:
                sim_file_base_name = basename(simulation_file)
                if isdir(simulation_file):
                    shutil.copytree(simulation_file,
                                    join(run_path, sim_file_base_name))
                elif isfile(simulation_file):
                    shutil.copyfile(simulation_file,
                                    join(run_path, sim_file_base_name))
                else:
                    prefixed_files = glob(f'{simulation_file}*')
                    if len(prefixed_files) == 0:
                        self.logger.info(f'{simulation_file} does not match '
                                         'any files or directories')
                    for prefixed_file in prefixed_files:
                        prefixed_basename = basename(prefixed_file)
                        shutil.copyfile(prefixed_file,
                                        join(run_path, prefixed_basename))

    @abstractmethod
    def _write_input(
        self,
        run_path: str,
        input_template: str,
        template_fill: dict[str, Any],
        input_file_name: Optional[str] = None,
    ) -> None:
        """
        Generate an input file for running a simulator calculation

        generate an input file using the ``input_template`` and
        ``template_fill`` for the given structural configuration, written as
        an external file by :meth:`write_initial_config`

        :param run_path: root path where simulations will run
        :type run_path: str
        :param input_template: input template to use
        :type input_template: str
        :param template_fill: additional arguments for the template, model
            specific
        :type template_fill: dict
        :param input_file_name: name for the input file
        :type input_file_name: str
        """
        pass

    @abstractmethod
    def _write_initial_config(
        self,
        run_path: str,
        atoms: Union[Atoms, list[Atoms]],
    ) -> None:
        """
        Generate an input file for the initial structural configuration

        :param run_path: path where the configuration file will be written
        :type run_path: str
        :param atoms: the ASE Atoms object
        :type atoms: Atoms
        """
        pass

    @abstractmethod
    def _get_run_command(self, args: Optional[dict[str, Any]] = None) -> str:
        """
        Return the command to run a simulator calculation.

        This method formats the run command based on the ``code_path`` internal
        variable set at instantiation of the Simulator, which the
        :class:`~.scheduler_base.Scheduler` will execute in
        the proper ``run_path``. The args dictionary can be used to pass any
        necessary extra parameters to the specific implementations.

        :param args: dictionary for parameters to decorate or enable the run
            command |default| ``None``
        :type args: dict
        :returns: command to run the simulator
        :rtype: str
        """
        pass

    @abstractmethod
    def parse_for_storage(
        self,
        run_path: str = '',
        calc_id: Union[int, str] = None,
        scheduler: Scheduler = None,
    ) -> list[Atoms]:
        """
        Process calculation output to extract data in a consistent format.

        :param run_path: directory where the simulator output file resides.
            If not provided, will be extracted from the scheduler using calc_id
        :type run_path: str
        :param calc_id: Calculation ID to look up via scheduler.get_job_path().
            Can be int or str depending on scheduler implementation.
        :type calc_id: int or str
        :param scheduler: Scheduler object of Orchestrator.
        :type scheduler: Scheduler
        :returns: list of ASE Atoms of the configurations and any attached
            properties. Metadata with the configuration source information is
            attached to the METADATA_KEY in the info dict.
        :rtype: Atoms list
        """
        pass
