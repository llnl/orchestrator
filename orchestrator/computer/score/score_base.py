import os
import json
import numpy as np
from abc import abstractmethod
from copy import deepcopy
from tqdm import tqdm
from ase import Atoms
from enum import Enum
import shutil

from typing import Optional, Any, Union

from .. import Computer
from orchestrator.utils.data_standard import METADATA_KEY
from orchestrator.utils.input_output import safe_read, safe_write
from orchestrator.scheduler import Scheduler


# so that external modules can query the possible score quantities without
# having to instantiate a Score
class ScoreQuantity(Enum):
    IMPORTANCE = 0  # example: weight from FIM matching
    UNCERTAINTY = 1  # example: LTAU
    EFFICIENCY = 2  # example: 1 - H/maxH from QUESTS
    DIVERSITY = 3  # example: D from QUESTS
    DELTA_ENTROPY = 4  # example: dH from QUESTS
    SENSITIVITY = 5  # example: FIM matrix


class ScoreBase(Computer):
    """
    Abstract base class for an object which returns a "score" of its inputs.
    For example, an "importance" score or an "uncertainty" score. Details of
    the inputs or outputs will vary across implementations.
    """

    OUTPUT_KEY = 'score'  # overloaded by specific implementations

    # overloaded by children; a subset of SCORE_QUANTITIES
    suppported_score_quantities = []

    data_file_name = 'score_results.xyz'  # change if child is not ASE output
    output_file_name = data_file_name
    init_args_file_name = 'score_init_args.json'
    init_args_subdir = 'score_init_args_temp_files'
    compute_args_file_name = 'score_compute_args.json'
    compute_args_subdir = 'score_compute_args_temp_files'
    script_file_name = 'score_compute_script.py'

    @abstractmethod
    def compute(
        self,
        atoms: Atoms,
        score_quantity: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Runs the calculation for a single atomic configuration. This is
        intended to be able to be used in a serial (non-distributed) manner,
        outside of a proper orchestrator scheduler.

        :param atoms: the ASE Atoms object
        :type atoms: Atoms
        :param score_quantity: the type of score value to compute
        :type score_quantity: int
        :returns: the score, where the first dimension should be the number of
            atoms.
        :rtype: np.ndarray
        """
        raise NotImplementedError

    @abstractmethod
    def compute_batch(
        self,
        list_of_atoms: list[Atoms],
        score_quantity: int,
        **kwargs,
    ) -> list[np.ndarray]:
        """
        Runs the calculation for a batch of atomic configurations. This is
        intended to be able to be used in a serial (non-distributed) manner,
        outside of a proper orchestrator scheduler.

        :param list_of_atoms: a list of ASE Atoms objects
        :type list_of_atoms: list[Atoms]
        :param score_quantity: the type of score value to compute
        :type score_quantity: int
        :returns: a list of scores for each atomic configuration
        :rtype: list
        """
        raise NotImplementedError

    def get_run_command(
        self,
        **kwargs,
    ) -> str:
        """
        Return the command to run calculations within a scheduler. This allows
        for distributed execution of ``compute()``.

        :returns: string for execution via command line
        :rtype: str
        """

        # Note: uses the same execution script as get_batched_run_command().
        # The input file may contain multiple configurations.
        return f'python {self.script_file_name}'

    def get_batched_run_command(
        self,
        **kwargs,
    ) -> str:
        """
        Similar to ``get_run_command()``, this function is meant to support
        executing ``compute_batch()`` within a scheduler.

        :returns: string for execution via command line
        :rtype: str
        """

        return f'python {self.script_file_name}'

    @abstractmethod
    def _write_runfile(
        self,
        run_path: str,
    ):
        """
        Generates a python script which will be called by .run()

        :param run_path: directory in which to save the script
        :type run_path: str
        """
        raise NotImplementedError

    def run(
        self,
        path_type: str,
        configs: Union[list[Atoms], list[Any]],
        compute_args: dict[str, Any],
        scheduler: Optional[Scheduler] = None,
        job_details: Optional[dict[str, Any]] = None,
        batch_size: Optional[int] = 1,
        verbose: Optional[bool] = False,
    ) -> list[int]:
        """
        Main function to compute the score for a collection of atomic
        configurations.

        The run method includes half of the main functionality of the score
        module, taking atomic configurations as input and handling the
        submission of calculations to obtain the computed results. `configs` is
        a dataset of 1 or more structures. run() will create independent jobs
        for each structure using the supplied scheduler, with job_details
        parameterizing the job submission.

        :param path_type: specifier for the scheduler path, to differentiate
            calculation types
        :type path_type: str
        :param compute_args: input arguments to fill out the input file
        :type compute_args: dict
        :param configs: list of configurations or data samples to be used for
            score calculation. Each item can be an ASE Atoms object or any
            other format supported by the score module.
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
        :type verbose: bool
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
            self.write_input(run_path, deepcopy(compute_args),
                             [configs[i] for i in frames])
            modified_job_details = deepcopy(job_details)
            if isinstance(configs[0], Atoms):
                # This only applies when configs is a list of Atoms
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

    def write_input(
        self,
        run_path: str,
        compute_args: dict[str, Any],
        configs: Union[list[Atoms], list[Any]],
    ) -> str:
        """
        Generate input files for running the calculation.

        This method will write the requisite input files in the run_path.
        Specific implementations may leverage additional helper functions to
        construct the input. Notably, and arguments that are passed as
        in-memory arrays will be written out to temporary files, which will be
        removed later by .cleanup().

        :param run_path: directory path where the file is written
        :type run_path: str
        :param compute_args: arguments for the computer
        :type compute_args: dict
        :param configs: list of configurations or data samples to be used for
            score calculation. Each item can be an ASE Atoms object or any
            other format supported by the score module.
        :type configs: list
        :returns: name of written input file
        :rtype: str
        """

        self._write_args_to_temp_dir(
            compute_args, os.path.join(run_path, self.compute_args_subdir),
            self.compute_args_file_name)

        # Write atomic configuration to job folder
        save_path = os.path.join(run_path, self.data_file_name)

        if os.path.isfile(save_path):
            self.logger.info(f'Warning, overwriting {self.data_file_name} '
                             f'at {run_path}')

        self.write_data(save_path, configs)

        self.logger.info(f'Score data written to {save_path}')

        # Write execution script
        self._write_runfile(run_path)

        self.logger.info(f'Execution script written to to {run_path}')

        return save_path

    def _write_args_to_temp_dir(
        self,
        args: dict[str, Any],
        tmpdir: str,
        args_file_name: str,
    ):
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
        if args is not None:
            for k, v in args.items():
                if isinstance(v, np.ndarray):
                    # Create the temp dir if it doesn't exist yet
                    temp_file_path = os.path.abspath(
                        os.path.join(tmpdir, f"{k}.npy"))

                    np.save(temp_file_path, v)
                    args[k] = temp_file_path

                if isinstance(v, str):
                    if os.path.exists(v):
                        args[k] = os.path.abspath(v)

                # TODO: write out ASE Atoms objects to avoid serialization.
                # Will also need code to handle reading this atoms from the job

        with open(os.path.join(tmpdir, args_file_name), 'w') as f:
            json.dump(args, f, indent=4)

    def cleanup(
        self,
        run_path: Union[str, None] = None,
    ):
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

    def read_data(self, read_path: str, **kwargs) -> list[Atoms]:
        """
        Read the configurations or data from a file.
        """
        return safe_read(read_path, format='extxyz', index=':')

    def write_data(self, save_path: str, configs: list[Atoms], **kwargs):
        """
        Write the configurations or data to a file.
        """
        safe_write(save_path, configs, format='extxyz')


class AtomCenteredScore(ScoreBase):

    def save_results(
        self,
        compute_results: Union[list[np.ndarray], np.ndarray],
        save_dir: str = '.',
        list_of_configs: Union[list[Atoms], Atoms] = None,
        **kwargs,
    ):
        """
        Save calculation output to a file.

        Since these results are per-atom scores, they will be
        saved in the .arrays dictionary of an Atoms object. Note that this code
        assumes that the ASE file used to compute the results already exists in
        save_dir.

        :param compute_results: the output of .compute() or .compute_batch()
        :type compute_results: np.ndarray or list[np.ndarray]

        :param save_dir: folder in which to save the results
        :type save_dir: str

        :param list_of_configs: the atomic configurations for which the results
            were computed. Must be provided so that results can be attached and
            saved on the correct Atoms objects.
        :type list_of_configs: list or Atoms

        """

        if list_of_configs is None:
            raise RuntimeError("Must provide Atoms to "
                               "AtomCenteredScore.save_results")

        if isinstance(compute_results, np.ndarray):
            compute_results = [compute_results]

        if isinstance(list_of_configs, Atoms):
            list_of_configs = [list_of_configs]

        if len(compute_results) != len(list_of_configs):
            raise RuntimeError("Length of compute results "
                               f"({len(compute_results)}) does not match "
                               f"number of configs ({len(list_of_configs)})")

        for atoms, d in zip(list_of_configs, compute_results):
            atoms.arrays[self.OUTPUT_KEY + "_score"] = d

        atoms.info[METADATA_KEY] = self._metadata

        self.write_data(os.path.join(save_dir, self.output_file_name),
                        list_of_configs)

    def parse_for_storage(
        self,
        run_path: str = '',
        cleanup: bool = True,
        calc_id: Union[int, str] = None,
        scheduler: Scheduler = None,
    ) -> list[Atoms]:
        """
        Process calculation output to extract data in a consistent format, then
        run cleanup() to remove any unnecessary temporary files.

        :param run_path: directory where the output file resides.
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
        :returns: Atoms of the configurations with attached properties and
            metadata
        :rtype: list of Atoms
        """
        if not run_path:
            run_path = scheduler.get_job_path(calc_id)
        data_file = os.path.join(run_path, self.output_file_name)

        results = self.read_data(data_file)

        key_name = self.OUTPUT_KEY + '_score'

        for i, atoms in enumerate(results):
            assert (key_name in atoms.arrays), (
                f"{key_name} key not found on "
                f"results[{i}].arrays in {data_file}: {atoms.arrays.keys()=}")

        if cleanup:
            self.cleanup(run_path)  # delete compute_args

        return results

    def _write_runfile(
        self,
        run_path: str,
    ):
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
                (f'init_args = read_input('
                 f'"{self.init_args_subdir}/{self.init_args_file_name}")'),
                (f'compute_args = read_input('
                 f'"{self.compute_args_subdir}/{self.compute_args_file_name}")'
                 ),
                ('input_args = {'
                 f'"score_type": "{self.__class__.__name__}"'
                 '}'),
                'input_args["score_args"] = init_args',
                ('score = init_and_validate_module_type("score", input_args, '
                 'single_input_dict=True)'),
                'list_of_configs = score.read_data(score.data_file_name)',
                ('results = score.compute_batch('
                 'list_of_configs, **compute_args'
                 ')'),
                'score.save_results(results, ".", list_of_configs)',
            ]))


class ConfigurationScore(ScoreBase):

    def save_results(
        self,
        compute_results: Union[list[np.ndarray], np.ndarray],
        save_dir: str = '.',
        list_of_configs: Union[list[Atoms], Atoms] = None,
        **kwargs,
    ):
        """
        Save calculation output to a file.

        Since these results are per-atom scores, they will be
        saved in the .info dictionary of an Atoms object. Note that this code
        assumes that the ASE file used to compute the results already exists in
        save_dir.

        :param compute_results: the output of .compute() or .compute_batch()
        :type compute_results: np.ndarray or list[np.ndarray]

        :param save_dir: folder in which to save the results
        :type save_dir: str

        :param list_of_configs: the atomic configurations for which the results
            were computed. Must be provided so that results can be attached and
            saved on the correct Atoms objects.
        :type list_of_configs: list or Atoms
        """

        if list_of_configs is None:
            raise RuntimeError("Must provide Atoms to "
                               "ConfigurationScore.save_results")

        if isinstance(compute_results, np.ndarray):
            compute_results = [compute_results]

        if isinstance(list_of_configs, Atoms):
            list_of_configs = [list_of_configs]

        if len(compute_results) != len(list_of_configs):
            raise RuntimeError("Length of compute results "
                               f"({len(compute_results)}) does not match "
                               f"number of configs ({len(list_of_configs)})")

        for atoms, d in zip(list_of_configs, compute_results):
            atoms.info[self.OUTPUT_KEY + "_score"] = d

        atoms.info[METADATA_KEY] = self._metadata

        self.write_data(os.path.join(save_dir, self.output_file_name),
                        list_of_configs)

    def parse_for_storage(
        self,
        run_path: str = '',
        cleanup: bool = True,
        calc_id: Union[int, str] = None,
        scheduler: Scheduler = None,
    ) -> list[Atoms]:
        """
        Process calculation output to extract data in a consistent format, then
        run cleanup() to remove any unnecessary temporary files.

        :param run_path: directory where the output file resides.
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
        :returns: Atoms of the configurations with attached properties and
            metadata
        :rtype: list of Atoms
        """
        if not run_path:
            run_path = scheduler.get_job_path(calc_id)
        data_file = os.path.join(run_path, self.output_file_name)

        results = self.read_data(data_file)

        key_name = self.OUTPUT_KEY + '_score'

        for i, atoms in enumerate(results):
            assert (key_name in atoms.info), f"{key_name} key not found on "
            f"results[{i}].info in {data_file}"

        if cleanup:
            self.cleanup(run_path)  # delete compute_args

        return results

    def _write_runfile(
        self,
        run_path: str,
    ):
        """
        Generates a python script which will be called by .run(). Almost
        identical to AtomCenteredDescripitor._write_runfile(), but saves
        results to .info instead of .arrays.

        :param run_path: directory in which to save the script
        :type run_path: str
        """
        with open(os.path.join(run_path, self.script_file_name), 'w') as f:
            f.writelines('\n'.join([
                'import numpy as np',
                ('from orchestrator.utils.setup_input import read_input,'
                 ' init_and_validate_module_type'),
                (f'init_args = read_input('
                 f'"{self.init_args_subdir}/{self.init_args_file_name}")'),
                (f'compute_args = read_input('
                 f'"{self.compute_args_subdir}/'
                 f'{self.compute_args_file_name}")'),
                ('input_args = {'
                 f'"score_type": "{self.__class__.__name__}"'
                 '}'),
                'input_args["score_args"] = init_args',
                ('score = init_and_validate_module_type("score", input_args, '
                 'single_input_dict=True)'),
                'list_of_configs = score.read_data(score.data_file_name)',
                ('results = score.compute_batch('
                 'list_of_configs, **compute_args'
                 ')'),
                'score.save_results(results, ".", list_of_configs)',
            ]))


class DatasetScore(ScoreBase):

    output_file_name = 'score_results.json'

    def compute_batch(self, list_of_atoms: list[Atoms], score_quantity: int,
                      **kwargs) -> list[np.ndarray]:
        # because you have to consider the whole dataset at once
        raise RuntimeError("Batching is not supported for Dataset scores.")

    def save_results(
        self,
        compute_results: Union[list[np.ndarray], np.ndarray],
        save_dir: str = '.',
        **kwargs,
    ):

        if isinstance(compute_results, np.ndarray):
            compute_results = [compute_results]

        output_dict = {
            self.OUTPUT_KEY + '_score':
            [cr.tolist() for cr in compute_results]
        }
        output_dict[METADATA_KEY] = self._metadata

        with open(os.path.join(save_dir, self.output_file_name), 'w') as f:
            json.dump(output_dict, f, indent=4)

    def parse_for_storage(
        self,
        run_path: str = '',
        cleanup: bool = True,
        calc_id: Union[int, str] = None,
        scheduler: Scheduler = None,
    ) -> list[Atoms]:
        """
        Process calculation output to extract data in a consistent format, then
        run cleanup() to remove any unnecessary temporary files.

        :param run_path: directory where the output file resides.
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
        :returns: Atoms of the configurations with attached properties and
            metadata
        :rtype: list of np.ndarray
        """
        if not run_path:
            run_path = scheduler.get_job_path(calc_id)
        data_file = os.path.join(run_path, self.output_file_name)
        with open(data_file, 'r') as f:
            results = json.load(f)

        key_name = self.OUTPUT_KEY + '_score'

        assert key_name in results, f"{key_name} not found in {data_file}"

        if cleanup:
            self.cleanup(run_path)  # delete compute_args

        return results

    def _write_runfile(
        self,
        run_path: str,
    ):
        """
        Generates a python script which will be called by .run().

        Almost identical to AtomCenteredDescripitor._write_runfile(), but saves
        results to .info instead of .arrays.

        :param run_path: directory in which to save the script
        :type run_path: str
        """
        with open(os.path.join(run_path, self.script_file_name), 'w') as f:
            f.writelines('\n'.join([
                'import numpy as np',
                ('from orchestrator.utils.setup_input import read_input,'
                 ' init_and_validate_module_type'),
                (f'init_args = read_input('
                 f'"{self.init_args_subdir}/{self.init_args_file_name}")'),
                (f'compute_args = read_input('
                 f'"{self.compute_args_subdir}/'
                 f'{self.compute_args_file_name}")'),
                ('input_args = {'
                 f'"score_type": "{self.__class__.__name__}"'
                 '}'),
                'input_args["score_args"] = init_args',
                ('score = init_and_validate_module_type("score", input_args, '
                 'single_input_dict=True)'),
                'dataset = score.read_data(score.data_file_name)',
                'results = score.compute(dataset, **compute_args)',
                'score.save_results(results)',
            ]))

    def run(
        self,
        path_type: str,
        configs: list[Atoms],
        compute_args: dict[str, Any],
        scheduler: Optional[Scheduler] = None,
        job_details: Optional[dict[str, Any]] = None,
        batch_size: Optional[int] = 1,
        verbose: Optional[bool] = False,
    ) -> list[int]:
        """
        Custom .run() for handling dataset inputs instead of configurations.

        Sets the batch size to the entire dataset size, and raises an error if
        batch_size is not 1 or the full dataset size.
        """

        if (batch_size > 1) and (batch_size != len(configs)):
            raise RuntimeError("DatasetScore.run() should only ever be called"
                               " with `batch_size=1` or the full dataset "
                               "size.")

        batch_size = len(configs)  # to make sure it's all passed at same time

        # returns the calc ids
        return super().run(path_type=path_type,
                           configs=configs,
                           compute_args=compute_args,
                           scheduler=scheduler,
                           job_details=job_details,
                           batch_size=batch_size,
                           verbose=verbose)


class ModelScore(ScoreBase):
    """
    A class for storing score results related to a single model, rather than
    for a single atom or atomic configuration.
    """

    output_file_name = 'score_results.json'

    def compute(
        self,
        data: Any,
        score_quantity: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Unlike the other score modules, which always act on the atomic
        configurations, model score doesn't have this restriction. For
        example, it can act on some target property, etc., although it can
        still act on the atomic configurations.

        :param data: general data for the score calculation, a generalization
            of the `atoms` argument in the other score modules
        :type data: Any
        :param score_quantity: the type of score value to compute
        :type score_quantity: int
        :returns: the score, where the first dimension should be the number of
            atoms.
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def compute_batch(
        self,
        list_of_data: list[Any],
        score_quantity: int,
        **kwargs,
    ) -> list[np.ndarray]:
        """
        Unlike the other score modules, which always act on the atomic
        configurations, model score doesn't have this restriction and can act
        on a more general object or quantity. For example, it can act on some
        target property, etc., although it can still act on the atomic
        configurations.

        :param list_of_data: list of general data for the score calculation, a
            generalization of the `list_of_atoms` argument in the other score
            modules
        :type data: list[Any]
        :param score_quantity: the type of score value to compute
        :type score_quantity: int
        :returns: a list of scores for each atomic configuration
        :rtype: list
        """
        raise NotImplementedError

    def save_results(
        self,
        compute_results: Union[list[np.ndarray], np.ndarray],
        save_dir: str = '.',
        **kwargs,
    ):
        """
        Save calculation output to a file.

        Save the results as a JSON file, so that we can also store the
        metadata.
        The order of list elements stored under the `self.OUTPUT_KEY_score`
        key matches the order of elements in the `compute_results` input
        argument, which is typically the same as the output order of the
        `compute_batch()` method.

        :param compute_results: the output of .compute() or .compute_batch()
        :type compute_results: np.ndarray or list[np.ndarray]

        :param save_dir: folder in which to save the results
        :type save_dir: str
        """

        if isinstance(compute_results, np.ndarray):
            compute_results = [compute_results]

        output_dict = {
            self.OUTPUT_KEY + '_score':
            [cr.tolist() for cr in compute_results]
        }
        output_dict[METADATA_KEY] = self._metadata

        with open(os.path.join(save_dir, self.output_file_name), 'w') as f:
            json.dump(output_dict, f, indent=4)

    def parse_for_storage(
        self,
        run_path: str = '',
        cleanup: bool = True,
        calc_id: Union[int, str] = None,
        scheduler: Scheduler = None,
    ) -> dict:
        """
        Process calculation output to extract data in a consistent format, then
        run cleanup() to remove any unnecessary temporary files.

        :param run_path: directory where the output file resides.
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
        :returns: A dictionary with the score value(s) and metadata
        :rtype: dict
        """
        if not run_path:
            run_path = scheduler.get_job_path(calc_id)
        with open(os.path.join(run_path, self.output_file_name), 'r') as f:
            results = json.load(f)

        if cleanup:
            self.cleanup(run_path)  # delete compute_args

        return results

    def _write_runfile(
        self,
        run_path: str,
    ):
        """
        Modified ._write_runfile() for ModelScore.

        The dataset is read using a custom .read_data(), which needs to be
        defined separately.

        :param run_path: directory in which to save the script
        :type run_path: str
        """
        with open(os.path.join(run_path, self.script_file_name), 'w') as f:
            f.writelines('\n'.join([
                'import numpy as np',
                ('from orchestrator.utils.setup_input import read_input,'
                 ' init_and_validate_module_type'),
                (f'init_args = read_input('
                 f'"{self.init_args_subdir}/{self.init_args_file_name}")'),
                (f'compute_args = read_input('
                 f'"{self.compute_args_subdir}/'
                 f'{self.compute_args_file_name}")'),
                ('input_args = {'
                 f'"score_type": "{self.__class__.__name__}"'
                 '}'),
                'input_args["score_args"] = init_args',
                ('score = init_and_validate_module_type("score", input_args, '
                 'single_input_dict=True)'),
                'dataset = score.read_data(score.data_file_name)',
                'results = score.compute_batch(dataset, **compute_args)',
                'score.save_results(results, ".")',
            ]))

    def read_data(self, read_path: str, **kwargs) -> list[Any]:
        """
        Read the data from a file.

        This method should handle whatever data format used by the score
        module.
        """
        raise NotImplementedError

    def write_data(self, save_path, data, **kwargs):
        """
        Write the data to a file.

        This method should handle whatever data format used by the score
        module.
        """
        raise NotImplementedError
