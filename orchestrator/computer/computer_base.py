from abc import ABC, abstractmethod
from datetime import datetime
from ase import Atoms
import numpy as np
import os
import json
import shutil

from typing import Optional, Union, Any

from ..scheduler.factory import scheduler_builder
from ..utils.isinstance import isinstance_no_import
from ..utils.recorder import Recorder
from ..utils.exceptions import DatasetDoesNotExistError
from orchestrator.scheduler import Scheduler
from orchestrator.storage import Storage


class Computer(Recorder, ABC):
    """Abstract base class for the computer."""

    # should be overloaded by child classes
    # NOTE: should be a valid name for a KIM property, in order to work with
    # ColabFit (alphanumerics and dashes; no underscores)
    # NOTE: update PER_ATOM_PROPERTIES in utils/data_standard.py if the
    # computed value should be stored in the .arrays dict instead of .info
    OUTPUT_KEY = None

    # NOTE: if child class accepts np.ndarray arguments as input, then it
    # should also support reading those arrays from a file in order to handle
    # passing in-memory arrays to jobs via .run()
    _init_args = None  # used by .run() for re-initialzing within a batch job
    _metadata = None  # extra info to store in database when saving results

    def __init__(self, **kwargs):
        super().__init__()

        # set up default scheduler and storage
        self.default_scheduler = scheduler_builder.build(
            'LOCAL',
            {'root_directory': './computer'},
        )

    def compute(self, atoms: Atoms, **kwargs) -> np.ndarray:
        """
        Runs the calculation for a single atomic configuration. This is
        intended to be able to be used in a serial (non-distributed) manner,
        outside of a proper orchestrator scheduler.

        :param atoms: the ASE Atoms object
        :type atoms: Atoms
        :returns: some value; depends upon sub-class
        """
        raise NotImplementedError

    def compute_batch(self, list_of_atoms: list[Atoms],
                      **kwargs) -> list[np.ndarray]:
        """
        Runs the calculation for a batch of atomic configurations. This is
        intended to be able to be used in a serial (non-distributed) manner,
        outside of a proper orchestrator scheduler.

        :param list_of_atoms: a list of ASE Atoms objects
        :type list_of_atoms: list
        :param args: any additional arguments to be passed to calculator method
        :type args: dict
        :returns: a list of values equivalent to
            ``[self.compute(atoms, args) for atoms in list_of_atoms]``
        :rtype: list
        """
        raise NotImplementedError

    @abstractmethod
    def get_run_command(self, **kwargs) -> str:
        """
        Return the command to run calculations within a scheduler. This allows
        for distributed execution of ``compute()``.

        This method formats the run command, while the args dictionary
        can be used to pass any necessary extra parameters to the specific
        implementations.

        :returns: implementation dependent
        :rtype: implementation dependent
        """
        raise NotImplementedError

    @abstractmethod
    def get_batched_run_command(self, **kwargs) -> str:
        """
        Similar to ``get_run_command()``, this function is meant to support
        executing ``compute_batched()`` within a scheduler.

        :returns: implementation dependent
        :rtype: implementation dependent
        """
        raise NotImplementedError

    @abstractmethod
    def run(self,
            path_type: str,
            scheduler: Optional[Scheduler] = None) -> list[int]:
        """
        Executes the calculation across a provided scheduler. Note that
        sub-classes may have implementations with additional arguments.

        :param path_type: specifier for the scheduler path, to differentiate
            calculation types.
        :type path_type: str
        :param scheduler: the scheduler for managing job submission, if none
            are supplied, will use the default scheduler defined in this class
            |default| ``None``
        :type scheduler: Scheduler
        :returns: a list of calculation IDs from the scheduler.
        :rtype: list
        """
        raise NotImplementedError

    def save_labeled_configs(self,
                             data_pointers: list[Union[int, Atoms]],
                             storage: Optional[Storage] = None,
                             dataset_name: Optional[str] = None,
                             dataset_handle: Optional[str] = None,
                             scheduler: Optional[Scheduler] = None,
                             cleanup: Optional[bool] = True) -> str:
        """
        Extract and save computed data to storage.

        Once the calculations are complete, the data they generate must be
        integrated with the structural configuration in a consistent framework
        to be used for training. This is done by parsing and ingesting the
        configuration and attached data into a dataset handled by the
        :class:`~orchestrator.storage.storage_base.Storage` module.

        :param data_pointers: configs or calc_ids or explicit paths associated
            with each config. If calc_ids or explicit paths are supplied, they
            should point to ASE-readable files from which to load the Atoms
            objects.  If calc_ids are supplied, the path is extracted from the
            :class:`~orchestrator.scheduler.scheduler_base.JobStatus`. Calc IDs
            are generally prefered as they can also carry metadata with them.
        :type data_pointers: list (of Atoms or int or str)
        :param storage: specific module that handles the staroge of data.
            |default| ``None``
        :type storage: Storage
        :param dataset_name: Name of the dataset in the database. If ``None``,
            then the class default (date stamped) is used. |default| ``None``
        :type dataset_name: str
        :param dataset_handle: the handle to identify where in Storage the
            configurations should be saved.
        :type dataset_handle: str
        :param scheduler: the scheduler for managing job submission, if none
            are supplied, will use the default scheduler defined in this class.
            Should be consistent with the scheduler supplied for the run calls.
            |default| ``None``
        :type scheduler: Scheduler
        :param cleanup: a flag indicating whether to delete the temporary
            files. |default| ``True``
        :type cleanup: bool
        :returns: dataset handle
        :rtype: str
        """
        if len(data_pointers) == 0:
            self.logger.info(
                f"No data pointers supplied to {self.__class__.__name__}. "
                "Returning")
            return None

        if isinstance(data_pointers[0], Atoms):
            data = data_pointers  # data is already loaded
        else:
            data = self.data_from_calc_ids(data_pointers, scheduler, cleanup)

        current_date = datetime.today().strftime('%Y-%m-%d')
        dataset_metadata = {
            'description': (f'data generated by {self.__class__.__name__} on '
                            f'{current_date}')
        }

        if dataset_name is None and dataset_handle is None:
            dataset_name = storage.generate_dataset_name(
                f'{self.__class__.__name__}_dataset',
                f'{current_date}',
                check_uniqueness=True,
            )
        elif dataset_name:  # but dataset_handle is None
            if isinstance_no_import(storage, 'ColabfitStorage'):
                # Try extracting the dataset_handle from the storage
                try:
                    dataset_handle = storage._get_id_from_name(dataset_name)
                except DatasetDoesNotExistError:
                    # Backend is Colabfit, dataset_name was provided, but
                    # doesn't exist yet, so we'll make a new one below
                    pass
        else:
            if dataset_handle[:3] != 'DS_':
                raise ValueError(
                    'dataset handles should be in format DS_************_#')

        if dataset_handle:
            # dataset_name is None, but handle is a colabfit ID
            new_handle = storage.add_data(dataset_handle, data,
                                          dataset_metadata)
        else:
            # dataset_handle is still None, so either the backend is ColabFit
            # and the dataset doesn't exist, or the backend is Local. Either
            # way, just make a new dataset. Note that LocalStorage.new_dataset
            # checks to avoid overwrites, and instead just adds to existing.
            new_handle = storage.new_dataset(dataset_name, data,
                                             dataset_metadata)

        if cleanup:
            self.cleanup(
            )  # for removing any temporary files created by .run()

        return new_handle

    @abstractmethod
    def write_input(self, run_path: str, input_args: dict[str, Any]):
        """
        Writes any input data necessary for the calculation to the run path
        """
        raise NotImplementedError

    @abstractmethod
    def parse_for_storage(
        self,
        run_path: str = '',
        cleanup: bool = True,
        calc_id: Union[int, str] = None,
        scheduler: Scheduler = None,
    ):
        """
        Process calculation output to extract data in a consistent format

        :param run_path: directory where the output resides.
            If not provided, will be extracted from the scheduler using calc_id
        :type run_path: str
        :param cleanup: a flag indicating whether to delete the temporary
            files. |default| ``True``
        :type cleanup: bool
        :param calc_id: Calculation ID to look up via scheduler.get_job_path().
            Can be int or str depending on scheduler implementation.
        :type calc_id: int or str
        :param scheduler: Scheduler object of Orchestrator.
        :type scheduler: Scheduler
        :returns: depends upon implementation
        :rtype: depends upon implementation, but should always be a list
        """
        raise NotImplementedError

    @abstractmethod
    def save_results(
        self,
        compute_results: Union[list[np.ndarray], np.ndarray],
        save_dir: str,
        **kwargs,
    ):
        """
        Save calculation output to a file. Implementation dependent.

        Note that this function should also store any metadata associated with
        the calculation.

        :param compute_results: the output of .compute() or .compute_batch()
        :type compute_results: np.ndarray or list[np.ndarray]
        :param save_path: folder in which to save the results
        :type save_path: str
        """
        raise NotImplementedError

    def cleanup(self, run_path: Union[str, None] = None):
        """
        Removes any temporary files that were created for job execution.

        :param run_path: the parent directory containing the temp file subdir.
            If None, it is not being called by a batch job, so it should delete
            the init_args
        :type run_path: str
        """

        if run_path is None:  # not called by a job, so remove init_args
            # NOTE: init_args_subdir should be an abspath by now
            if os.path.isdir(self.init_args_subdir):
                shutil.rmtree(os.path.join(self.init_args_subdir))
        else:  # called by a job, so delete compute_args
            tmpdir = os.path.join(run_path, self.compute_args_subdir)
            if os.path.isdir(tmpdir):
                shutil.rmtree(os.path.join(tmpdir))

    def _write_args_to_temp_dir(self, args: dict[str, Any], tmpdir: str,
                                args_file_name: str):
        """
        Writes arguments to a temporary directory.

        Creates a temporary directory and saves any np.ndarray arguments to the
        folder, overwriting the value in the args dict with the abspath to the
        new file. The updated args dict is then saved as a JSON file.

        :param args: dictionary of arguments
        :type args: dict
        :param tmpdir: name of the directory that should be created
        :type tmpdir: str
        :param args_file_name: name for the JSON file
        :type str:
        """
        if not os.path.isdir(tmpdir):
            os.makedirs(tmpdir)

        # Write any in-memory compute_args to temporary files,
        # then overwrite compute_args with their new path
        for k, v in args.items():
            if isinstance(v, np.ndarray):
                # Create the temp dir if it doesn't exist yet
                temp_file_path = os.path.abspath(
                    os.path.join(tmpdir, f"{k}.npy"))

                np.save(temp_file_path, v)
                args[k] = temp_file_path

        with open(os.path.join(tmpdir, args_file_name), 'w') as f:
            json.dump(args, f, indent=4)

    def data_from_calc_ids(
        self,
        data_pointers: list[Union[int, str]],
        scheduler: Optional[Scheduler] = None,
        cleanup: Optional[bool] = True,
    ) -> list[Any]:
        """
        Return the parsed data from a list of calculation IDs.

        :param data_pointers: list of calc_ids (int or str) for extracting
            computed results. String calc_ids are supported for schedulers like
            Flux that use string job IDs.
        :type data_pointers: list[Union[int, str]]
        :param scheduler: the scheduler for managing job submission, if none
            are supplied, will use the default scheduler defined in this class
            |default| ``None``
        :type scheduler: Scheduler
        :param cleanup: a flag indicating whether to delete the temporary
            files. |default| ``True``
        :type cleanup: bool
        :returns: a list of the computed values
        :rtype: list
        """
        if scheduler is None:
            scheduler = self.default_scheduler

        # Ensure all calculations completed before extracting
        self.logger.info('Ensuring all calculations completed...')
        scheduler.block_until_completed(data_pointers)

        # Use scheduler method to resolve calc_ids to paths
        # allow_paths=False means only calc_ids are accepted, not file paths
        data_paths, existing_metadata = scheduler.resolve_calc_paths(
            data_pointers, allow_paths=False)

        self.logger.info((f'Extracting configs from {len(data_paths)} '
                          f'{self.__class__.__name__} calculations'))

        data = []  # for loaded data
        for run_path in data_paths:
            batched_results = self.parse_for_storage(run_path, cleanup)
            if not isinstance(batched_results, list):  # handle non-ASE outputs
                batched_results = [batched_results]
            data += batched_results

        return data

    def get_colabfit_property_definition(self) -> dict[str, Any]:
        """
        A 'property definition' is a dictionary used by the ColabFit storage
        module for exactly specifying the details (data type, shape,
        description, etc.) of each field required for uniquely defining a given
        property. This function must be implemented in order to support storage
        of the computed results in the ColabFit module.

        :returns: the property definition
        :rtype: dict
        """
        raise NotImplementedError

    def get_colabfit_property_map(self) -> dict[str, Any]:
        """
        Returns a default property map that can be used to extract a ColabFit
        property from an ASE.Atoms object. This assumes that the values being
        extracted are stored in their default locations based on the specific
        Computer module (usually within the compute() or compute_batch()
        functions).

        A 'property map' is similar to a 'property definition', but instead
        tells ColabFit how to extract the keys specified in the property
        definition from an ASE.Atoms object. This function must be implemented
        in order to support storage of the computed results in the ColabFit
        module.

        :returns: the property map
        :rtype: dict
        """
        raise NotImplementedError
