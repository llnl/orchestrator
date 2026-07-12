import os
import numpy as np
from abc import abstractmethod
from copy import deepcopy
from tqdm import tqdm
from ase import Atoms
from orchestrator.utils.input_output import safe_write

from typing import Optional, Any, Union

from .. import Computer
from orchestrator.utils.data_standard import METADATA_KEY
from orchestrator.utils.input_output import safe_read
from orchestrator.scheduler import Scheduler


class DescriptorBase(Computer):
    """
    Abstract base class for descriptor calculations

    The descriptor class manages the construction and parsing of atomic
    descriptors to provide training or reference data. The input will
    consist of an atomic configuration and calculation parameters,
    and the output will be the descriptors corresponding to that configuration.
    These may be environment-level, configuration-level, or something else
    depending upon the implementation.
    """

    OUTPUT_KEY = 'descriptors'  # overloaded by specific implementations

    atoms_file_name = 'atoms_with_descriptors.xyz'
    init_args_file_name = 'descriptor_init_args.json'
    init_args_subdir = 'descriptor_init_args_temp_files'
    compute_args_file_name = 'descriptor_compute_args.json'
    compute_args_subdir = 'descriptor_compute_args_temp_files'
    script_file_name = 'descriptor_compute_script.py'

    @abstractmethod
    def compute(self, atoms: Atoms, **kwargs) -> np.ndarray:
        """
        Runs the calculation for a single atomic configuration. This is
        intended to be able to be used in a serial (non-distributed) manner,
        outside of a proper orchestrator scheduler.

        :param atoms: the ASE Atoms object
        :type atoms: Atoms
        :returns: (N, D) array of D-dimensional descriptors for all N atoms.
        :rtype: np.ndarray
        """
        raise NotImplementedError

    @abstractmethod
    def compute_batch(self, list_of_atoms: list[Atoms],
                      **kwargs) -> list[np.ndarray]:
        """
        Runs the calculation for a batch of atomic configurations. This is
        intended to be able to be used in a serial (non-distributed) manner,
        outside of a proper orchestrator scheduler.

        :param list_of_atoms: a list of ASE Atoms objects
        :type list_of_atoms: list
        :returns: list of (N, D) arrays of D-dimensional descriptors
            corresponding to the descriptors of each atomic configuration
        :rtype: list
        """
        raise NotImplementedError

    def get_run_command(self, **kwargs) -> str:
        """
        Return the command to run calculations within a scheduler. This allows
        for distributed execution of ``compute()``.

        :returns: string for execution via command line
        :rtype: str
        """

        # Note: uses the same execution script as get_batched_run_command().
        # The input file may contain multiple configurations.
        return f'python {self.script_file_name}'

    def get_batched_run_command(self, **kwargs) -> str:
        """
        Similar to ``get_run_command()``, this function is meant to support
        executing ``compute_batch()`` within a scheduler.

        :returns: string for execution via command line
        :rtype: str
        """

        return f'python {self.script_file_name}'

    @abstractmethod
    def _write_runfile(self, run_path: str):
        """
        Generates a python script which will be called by .run()

        :param run_path: directory in which to save the script
        :type run_path: str
        """
        raise NotImplementedError

    def run(self,
            path_type: str,
            compute_args: dict,
            configs: list[Atoms],
            scheduler: Optional[Scheduler] = None,
            job_details: Optional[dict[str, Any]] = None,
            batch_size: Optional[int] = 1,
            verbose: Optional[bool] = False) -> list[int]:
        """
        Main function to compute the descriptors for a collection of atomic
        configurations.

        The run method includes half of the main functionality of the computer,
        taking atomic configurations as input and handling the submission of
        calculations to obtain the computed results. `configs` is a dataset of
        1 or more structures. run() will create independent jobs for
        each batch of structures using the supplied scheduler, with job_details
        parameterizing the job submission.

        :param path_type: specifier for the scheduler path, to differentiate
            calculation types
        :type path_type: str
        :param compute_args: input arguments to fill out the input file
        :type compute_args: dict
        :param configs: list of configurations as ASE atoms to run ground truth
            calculations for
        :type configs: list
        :param scheduler: the scheduler for managing job submission, if none
            are supplied, will use the default scheduler defined in this class
            |default| ``None``
        :type scheduler: Scheduler
        :param job_details: dict that includes any additional parameters for
            running the job (passed to
            :meth:`~.scheduler_base.Scheduler.submit_job`)
            |default| ``{}``
        :type job_details: dict
        :param batch_size: number of configurations to pass to ``compute()`` at
            once. Default of 1 does not do any batching.
        :type batch_size: int
        :param verbose: if True, show progress
        :type batch_size: bool
        :returns: a list of calculation IDs from the scheduler.
        :rtype: list
        """
        module_name = self.__class__.__name__
        calc_ids = []

        if job_details is None:
            job_details = {}

        if scheduler is None:
            scheduler = self.default_scheduler

        path_base = scheduler.make_path_base(module_name, path_type)
        self.path_base = path_base
        num_calcs = len(configs)
        self.logger.info(f'Spinning up {num_calcs} {module_name} calculations')

        num_batches = int(np.ceil(len(configs) / batch_size))
        batch_indices = np.array_split(np.arange(len(configs)), num_batches)

        # Write self._init_args to a temporary directory, that way all batch
        # jobs can read from it.
        self._write_args_to_temp_dir(self._init_args, self.init_args_subdir,
                                     self.init_args_file_name)

        # update the directory to abspath so that it can be deleted later
        self.init_args_subdir = os.path.abspath(self.init_args_subdir)

        for frames in tqdm(batch_indices,
                           desc='Computing...',
                           disable=not verbose):
            run_path = scheduler.make_path(module_name, path_type)
            self.write_input(run_path, compute_args,
                             [configs[i] for i in frames])
            modified_job_details = deepcopy(job_details)
            try:
                modified_job_details['extra_args'][METADATA_KEY] = [
                    configs[i].info[METADATA_KEY] for i in frames
                ]
            except KeyError:
                if METADATA_KEY in configs[frames[0]].info:
                    modified_job_details['extra_args'] = {
                        METADATA_KEY:
                        [configs[i].info[METADATA_KEY] for i in frames]
                    }

            # NOTE: if we decide that get_run_command(arg) is redundant and can
            # be always replaced with get_batched_run_command([arg]), then this
            # if-else branch can be removed
            if batch_size > 1:
                computer_command = self.get_batched_run_command(
                    **modified_job_details)
            else:
                computer_command = self.get_run_command(**modified_job_details)

            calc_id = scheduler.submit_job(
                computer_command,
                run_path,
                job_details=modified_job_details,
            )
            calc_ids.append(calc_id)
        self.logger.info(f'Done generating {module_name} calculations')

        return calc_ids

    def write_input(self, run_path: str, compute_args: dict[str, Any],
                    configs: list[Atoms]) -> str:
        """
        Generate input files for running the calculation.

        This method will write the requisite input files in the run_path.
        Specific implementations may leverage additional helper functions to
        construct the input. Notably, any arguments that are passed as
        in-memory arrays will be written out to temporary files, which will be
        removed later by .cleanup().

        :param run_path: directory path where the file is written
        :type run_path: str
        :param compute_args: arguments for the computer
        :type compute_args: dict
        :param configs: the configurations as an Atoms objects.
        :type configs: list or Atoms
        :returns: name of written input file
        :rtype: str
        """

        self._write_args_to_temp_dir(
            compute_args, os.path.join(run_path, self.compute_args_subdir),
            self.compute_args_file_name)

        # Write atomic configuration to job folder
        save_path = os.path.join(run_path, self.atoms_file_name)

        if os.path.isfile(save_path):
            self.logger.info(
                f'Warning, overwriting {self.atoms_file_name} at {run_path}')

        safe_write(save_path, configs, format='extxyz')

        self.logger.info(f'Atoms written to {save_path}')

        # Write execution script
        self._write_runfile(run_path)

        self.logger.info(f'Execution script written to to {run_path}')

        return save_path

    def parse_for_storage(
        self,
        run_path: str = '',
        cleanup: bool = True,
        calc_id: Union[int, str] = None,
        scheduler: Scheduler = None,
    ) -> list[Atoms]:
        """
        Process calculation output as ASE Atoms, then clean up.

        Use ASE's read() function to parse the xyz file written by this module,
        then run cleanup() to remove any unnecessary temporary files.

        :param run_path: directory where the output file resides. If not
            provided, will be extracted from the scheduler using calc_id.
        :type run_path: str
        :param cleanup: a flag indicating whether to delete the temporary
            files. |default| ``True``
        :type cleanup: bool
        :param calc_id: Calculation ID to look up via scheduler.get_job_path().
            Can be int or str depending on scheduler implementation.
        :type calc_id: int or str
        :param scheduler: Scheduler object of Orchestrator.
        :type scheduler: Scheduler
        :returns: Atoms of the configurations with attached properties and
            metadata
        :rtype: list of Atoms
        """
        if not run_path:
            run_path = scheduler.get_job_path(calc_id)
        data_file = os.path.join(run_path, self.atoms_file_name)

        results = safe_read(data_file, format='extxyz', index=':')

        key_name = self.OUTPUT_KEY + '_descriptors'

        for i, atoms in enumerate(results):
            assert ((key_name in atoms.arrays) or (key_name in atoms.info)
                    ), f"{key_name} key not found on Atoms {i} in {data_file}"

        if cleanup:
            self.cleanup(run_path)  # delete compute_args

        return results


class AtomCenteredDescriptor(DescriptorBase):

    def save_results(
        self,
        descriptors: Union[list[np.ndarray], np.ndarray],
        save_dir: str = '.',
        list_of_configs: Union[list[Atoms], Atoms] = None,
        **kwargs,
    ):
        """
        Save descriptors to a file.

        Since these results are per-atom descriptors, they will be
        saved in the .arrays dictionary of an Atoms object. Note that this code
        assumes that the ASE file used to compute the results already exists in
        save_dir.

        :param descriptors: the computed descriptors
        :type descriptors: np.ndarray or list[np.ndarray]

        :param list_of_configs: the atomic configurations for which the
            descriptors were computed. Must be provided so that descriptors can
            be attached and saved on the correct Atoms objects.
        :type list_of_configs: list or Atoms

        :param save_path: folder in which to save the results
        :type save_path: str
        """

        if list_of_configs is None:
            raise RuntimeError("Must provide Atoms to "
                               "AtomCenteredScore.save_results")

        if isinstance(descriptors, np.ndarray):
            descriptors = [descriptors]

        if isinstance(list_of_configs, Atoms):
            list_of_configs = [list_of_configs]

        if len(descriptors) != len(list_of_configs):
            raise RuntimeError("Length of descriptors list "
                               f"({len(descriptors)}) does not match "
                               f"number of configs ({len(list_of_configs)})")

        for atoms, d in zip(list_of_configs, descriptors):
            atoms.arrays[self.OUTPUT_KEY + "_descriptors"] = d

        # atoms.info[METADATA_KEY] |= self._metadata

        safe_write(os.path.join(save_dir, self.atoms_file_name),
                   list_of_configs,
                   format="extxyz")

    def _write_runfile(self, run_path: str):
        """
        Generates a python script which will be called by .run()

        :param run_path: directory in which to save the script
        :type run_path: str
        """
        with open(os.path.join(run_path, self.script_file_name), 'w') as f:
            f.writelines('\n'.join([
                'import numpy as np',
                ('from orchestrator.utils.setup_input import read_input,'
                 ' init_and_validate_module_type'),
                'from orchestrator.utils.input_output import safe_read',
                (f'init_args = read_input('
                 f'"{self.init_args_subdir}/{self.init_args_file_name}")'),
                (f'compute_args = read_input('
                 f'"{self.compute_args_subdir}/{self.compute_args_file_name}")'
                 ),
                ('input_args = {'
                 f'"descriptor_type": "{self.__class__.__name__}"'
                 '}'),
                'input_args["descriptor_args"] = init_args',
                ('descriptor = init_and_validate_module_type("descriptor", '
                 'input_args, single_input_dict=True)'),
                ('list_of_configs = safe_read(descriptor.atoms_file_name, '
                 'format="extxyz", index=":")'),
                ('results = descriptor.compute_batch(list_of_configs, '
                 '**compute_args)'),
                'descriptor.save_results(results, ".", list_of_configs)',
            ]))


class ConfigurationDescriptor(DescriptorBase):
    """For generating configuration-level descriptors."""

    def save_results(
        self,
        descriptors: Union[list[np.ndarray], np.ndarray],
        save_dir: str = '.',
        list_of_configs: Union[list[Atoms], Atoms] = None,
        **kwargs,
    ):
        """
        Save descriptors to a file.

        Since these results are configuration-level descriptors, they will be
        saved in the .info dictionary of an Atoms object. Note that this code
        assumes that the ASE file used to compute the results already exists in
        save_dir.

        :param descriptors: the computed descriptors
        :type descriptors: np.ndarray or list[np.ndarray]

        :param list_of_configs: the atomic configurations for which the
            descriptors were computed. Must be provided so that descriptors can
            be attached and saved on the correct Atoms objects.
        :type list_of_configs: list or Atoms

        :param save_path: folder in which to save the results
        :type save_path: str
        """

        if list_of_configs is None:
            raise RuntimeError("Must provide Atoms to "
                               "AtomCenteredScore.save_results")

        if isinstance(descriptors, np.ndarray):
            descriptors = [descriptors]

        if isinstance(list_of_configs, Atoms):
            list_of_configs = [list_of_configs]

        if len(descriptors) != len(list_of_configs):
            raise RuntimeError("Length of descriptors list "
                               f"({len(descriptors)}) does not match "
                               f"number of configs ({len(list_of_configs)})")

        for atoms, d in zip(list_of_configs, descriptors):
            atoms.info[self.OUTPUT_KEY + "_descriptors"] = d

        atoms.info[METADATA_KEY] = self._metadata

        safe_write(os.path.join(save_dir, self.atoms_file_name),
                   list_of_configs,
                   format="extxyz")

    def _write_runfile(self, run_path: str):
        """
        Generates a python script which will be called by .run()

        :param run_path: directory in which to save the script
        :type run_path: str
        """
        with open(os.path.join(run_path, self.script_file_name), 'w') as f:
            f.writelines('\n'.join([
                'import numpy as np',
                ('from orchestrator.utils.setup_input import read_input,'
                 ' init_and_validate_module_type'),
                'from orchestrator.utils.input_output import safe_read',
                (f'init_args = read_input('
                 f'"{self.init_args_subdir}/{self.init_args_file_name}")'),
                (f'compute_args = read_input('
                 f'"{self.compute_args_subdir}/{self.compute_args_file_name}")'
                 ),
                ('input_args = {'
                 f'"descriptor_type": "{self.__class__.__name__}"'
                 '}'),
                'input_args["descriptor_args"] = init_args',
                ('descriptor = init_and_validate_module_type("descriptor", '
                 'input_args, single_input_dict=True)'),
                ('list_of_configs = safe_read(descriptor.atoms_file_name, '
                 'format="extxyz", index=":")'),
                ('results = descriptor.compute_batch(list_of_configs, '
                 '**compute_args)'),
                'descriptor.save_results(results, ".", list_of_configs)',
            ]))
