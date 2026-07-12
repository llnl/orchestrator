from abc import ABC, abstractmethod
from copy import deepcopy
from ..utils.recorder import Recorder
from ..utils.restart import restarter
from ..utils.templates import Templates
from ..utils.exceptions import UnidentifiedPathError
from os import system, PathLike
import os
import pickle
from time import sleep
from typing import Any, Optional, Union
import glob
import shutil
import os.path
import subprocess as sp


class JobStatus:
    """
    data class for collecting meta-data about a job

    Minimum attribute set for a JobStatus object is ``path``, ``state``, and
    ``exit_code``, where ``path`` defines the location where inputs and outputs
    are stored, ``state`` defines the jobs status (i.e. done, pending, running)
    , and ``exit_code`` shows the result of the job once completed, with 0
    indicating success. Note that ``state`` can change over time and
    ``exit_code`` may not be known when the object is first created.

    Additional attributes ``command``, ``job_details``, and ``metadata`` can be
    provided to store the command string, job resource details, and arbitrary
    metadata for the job.
    """

    def __init__(
        self,
        path: Union[str, PathLike],
        state: str = None,
        exit_code: Union[str, int] = None,
        command: str = None,
        job_details: dict = None,
        metadata: dict = None,
    ):
        """
        :param path: directory path where inputs and outputs are stored for
            the job
        :type path: str
        :param state: jobs status (i.e. done, pending, running)
        :type state: str
        :param exit_code: result of the job
        :type exit_code: int or str
        :param command: command string that defines the job to be executed
        :type command: str
        :param job_details: job resource details (nodes, tasks, walltime, etc.)
        :type job_details: dict
        :param metadata: arbitrary metadata associated with the job
        :type metadata: dict
        """
        self.path = path
        self.state = state
        self.exit_code = exit_code
        self.command = command
        self.job_details = job_details if job_details is not None else {}
        self.metadata = metadata if metadata is not None else {}


class Scheduler(Recorder, ABC):
    """
    Abstract base class to manage schedulers

    Responsibilities include directory creation, job creation, job status
    checking. A given scheduler class will use the ``root_directory`` as the
    base for all calculation inputs and outputs in a directory hierarchy
    managed by :meth:`make_path_base` and :meth:`make_path`. ``scheduler_args``
    provide a vehicle for modulating the scheduler behavior, though are not
    strictly required. The ``counters`` and ``jobs`` dictionaries are also
    initialized at instantiation, which are used to internally track scheduler
    components.
    """

    def __init__(
        self,
        root_directory: Optional[str] = './orchestrator_scheduler',
        checkpoint_file: Optional[str] = './orchestrator_checkpoint.json',
        checkpoint_name: Optional[Union[str, None]] = None,
        job_record_file: Optional[Union[str, None]] = None,
        **kwargs,
    ):
        """
        set variables and initialize the recorder

        :param root_directory: name of the directory under which all files and
            subdirectories will sit |default| './orchestrator_scheduler'
        :type root_directory: str
        :param checkpoint_file: name of the checkpoint file to write restart
            information to |default| './orchestrator_checkpoint.json'
        :type checkpoint_file: str
        :param checkpoint_name: name of the restart block for this module in
            the checkpoint file |default| 'scheduler'
        :type checkpoint_name: str
        :param job_record_file: name of the file to save the pickled jobs dict
            |default| './job_record.pkl'
        :type job_record_file: str
        """
        super().__init__()
        self.root_directory = root_directory
        self.checkpoint_file = checkpoint_file
        if checkpoint_name is None:
            checkpoint_name = self.__class__.__name__
        self.checkpoint_name = checkpoint_name
        if job_record_file is None:
            job_record_file = "./" + self.__class__.__name__ +\
                "_job_record.pkl"
        self.job_record_file = job_record_file
        self.counters = {}
        self.jobs = {}
        self.problematic_states = [
            'unknown',
            'error',
            'done_timeout',
            'done_cancelled',
            'done_other',
            'done_unknown',
        ]
        self.new_counters = False
        self.restart_scheduler()

    def make_path_base(
        self,
        module: str,
        path_type: str,
    ) -> Union[str, PathLike]:
        """
        Create the base directory for data and run outputs, excluding counter

        make_path_base generates a hierarchical directory structure within the
        root directory. The path hierarchy is root/module/path_type, where the
        ``module`` is the orchestrator module responsible for the call,
        ``path_type`` is the type of calculation or job within that module.
        This is a shortened version of :meth:`make_path`, omitting the count at
        the end of the directory path. The method returns the path as a string

        :param module: module requesting the path
        :type module: str
        :param path_type: "type" or purpose of the path, i.e. training_data,
            trajectories, ground_truth, etc. This is typically supplied in an
            input file
        :type path_type: str
        :returns: created path name
        :rtype: str
        """
        dir_name = f'{self.root_directory}/{module}/{path_type}'
        system(f'mkdir -p {dir_name} 2> /dev/null')

        return dir_name

    def make_path(self, module: str, path_type: str) -> Union[str, PathLike]:
        """
        Create the directory for data and run outputs to be located

        make_path generates a hierarchical directory structure within the root
        directory. The path hierarchy is root/module/path_type/count, where the
        ``module`` is the orchestrator module responsible for the call,
        ``path_type`` is the type of calculation or job within that module, and
        count is the increment of that specific calculation type. The method
        returns the generated path as a string

        :param module: the module requesting the path
        :type module: str
        :param path_type: "type" of the path, i.e. the purpose
        :type path_type: str
        :returns: created path name
        :rtype: str
        """
        counter_key = module + '.' + path_type
        if self.counters.get(counter_key) is None:
            self.counters[counter_key] = 0

        dir_name = (f'{self.root_directory}/{module}/{path_type}/'
                    f'{self.counters[counter_key]:05d}')
        system(f'mkdir -p {dir_name} 2> /dev/null')

        self.counters[counter_key] += 1
        self.new_counters = True
        self.checkpoint_scheduler()  # necessary for local
        return dir_name

    def get_job_status(self, job_handle: Union[int, str]) -> JobStatus:
        """
        Queries the status of a job handle.

        Returns :class:`~JobStatus`, which has (minimally) attributes: ``path``
        , ``state``, and ``exit_code``, which correspond to the location of the
        job, the job state (i.e. completed), and an exit code that is 0 if
        successful and a flag with information if not.

        :param job_handle: job ID originally returned from :meth:`~submit_job`
        :type job_handle: Union[int, str]
        :returns: job's :class:`~JobStatus`
        :rtype: JobStatus
        """
        job_status = self.jobs.get(job_handle)
        if job_status is None:
            self.logger.info(f'Queried ID {job_handle} does not exist')
        return job_status

    def get_job_path(
        self,
        job_handle: Union[int, str],
    ) -> Union[str, PathLike]:
        """
        returns the path where a specific job was run

        :param job_handle: job ID
        :type job_handle: Union[int, str]
        :returns: path where the job inputs/outputs are stored
        :rtype: str or PathLike
        """
        try:
            path = self.get_job_status(job_handle).path
        except AttributeError:
            path = None
            self.logger.info(
                f'Could not find path for job {job_handle}, return None')
        return path

    def get_attached_metadata(
        self,
        job_handle: Union[int, str],
    ) -> dict[str, Any]:
        """
        returns the metadata associated with a specific job

        :param job_handle: job ID
        :type job_handle: Union[int, str]
        :returns: dict of metadata associated with the job
        :rtype: dict
        """
        try:
            metadata = self.get_job_status(job_handle).metadata
        except AttributeError:
            metadata = {}
        return metadata

    def resolve_calc_paths(
        self,
        paths: list[Union[int, str]],
        allow_paths: bool = True,
    ) -> tuple[list[str], list[dict]]:
        """
        Helper method to resolve calc_ids or paths into data paths and metadata

        This method handles three types of inputs:
        1. Integer calc_ids (e.g., [1, 2, 3])
        2. String calc_ids (e.g., ['f123456789ab'] for Flux scheduler)
        3. Explicit file paths (e.g., ['./path/to/calc', '/abs/path'])

        The method validates calc_ids against the scheduler's ID_TYPE if
        available

        :param paths: List of calc_ids (int or str) or explicit file paths
            (str)
        :type paths: list[Union[int, str]]
        :param allow_paths: If True, allows explicit file paths in addition to
            calc_ids. If False, only calc_ids are accepted and strings that
            cannot be resolved as calc_ids will raise an error. |default| True
        :type allow_paths: bool
        :returns: Tuple of (data_paths, existing_metadata) where data_paths is
            a list of directory paths and existing_metadata is a list of dicts
        :rtype: tuple[list[str], list[dict]]
        :raises UnidentifiedPathError: If paths cannot be resolved as calc_ids
            or valid paths
        :raises TypeError: If calc_ids don't match the scheduler's expected
            ID_TYPE
        """
        if not paths:
            raise ValueError('paths list cannot be empty')

        first_item = paths[0]

        # Get the scheduler's expected ID type if available
        expected_id_type = getattr(self, 'ID_TYPE', None)

        # Case 1: Integer calc_ids
        if isinstance(first_item, int):
            # Validate against scheduler's ID_TYPE if it's set
            if expected_id_type is not None and expected_id_type is not int:
                raise TypeError('This scheduler expects calc_ids of type '
                                f'{expected_id_type.__name__}, '
                                f'but received int. Got: {first_item}')

            data_paths = []
            existing_metadata = []
            for calc_id in paths:
                data_paths.append(self.get_job_path(calc_id))
                existing_metadata.append(self.get_attached_metadata(calc_id))
            return data_paths, existing_metadata

        # Case 2: String inputs (could be paths or string calc_ids)
        if isinstance(first_item, str):
            # Check if it looks like a file path
            is_path = (os.sep in first_item or os.path.isabs(first_item)
                       or first_item.startswith('.'))

            if is_path:
                if not allow_paths:
                    raise UnidentifiedPathError(
                        'Explicit file paths are not allowed in this context. '
                        f'Only calc_ids are accepted. Got: {first_item}')
                return paths, [{} for _ in range(len(paths))]

            # Validate against scheduler's ID_TYPE if it's set
            if expected_id_type is not None and expected_id_type is not str:
                raise TypeError(f'This scheduler expects calc_ids of type '
                                f'{expected_id_type.__name__}, '
                                'but received str that does not appear to be a'
                                f' file path. Got: {first_item}')

            # Try treating as string calc_ids
            data_paths = []
            existing_metadata = []
            failed_indices = []

            for idx, calc_id in enumerate(paths):
                path = self.get_job_path(calc_id)
                if path is None:
                    failed_indices.append(idx)
                else:
                    data_paths.append(path)
                    existing_metadata.append(
                        self.get_attached_metadata(calc_id))

            if failed_indices:
                failed_items = [paths[i] for i in failed_indices]
                raise UnidentifiedPathError(
                    'Could not resolve the following items as calc_ids: '
                    f'{failed_items}. They do not appear to be valid paths or '
                    'calc_ids.')

            return data_paths, existing_metadata

        # Case 3: Unsupported type
        raise TypeError(
            f'Unsupported type for paths: {type(first_item).__name__}. '
            f'Expected int or str.')

    def get_all_statuses(self) -> dict[int, JobStatus]:
        """
        Returns information about all jobs from this Scheduler.

        Returns a dictionary with ``job_handle``: ``status``, where
        ``job_handle`` is returned by :meth:`~submit_job` and ``status`` is a
        :class:`~JobStatus` instance

        :returns: a dictionary of :class:`~JobStatus` objects
        :rtype: dict
        """
        return self.jobs

    def job_done_file_present(self, job_id: int) -> bool:
        """
        check if the job directory contains the "job_done" file

        Use an empty file to give persistent indication if the job completed
        to avoid problems where job statuses are purged by the scheduler after
        a certain amount of time. The job templates will "touch job_done" at
        the end of the script to provide this persistent indicator.

        :param job_id: job ID to check
        :type job_id: int
        :returns: True if file exists, false otherwise
        :rtype: boolean
        """
        job_path = self.get_job_path(job_id)
        return os.path.isfile(f'{job_path}/job_done')

    def save_job_dict(self):
        """
        Serialize the job dictionary for persistant storage

        Write the jobs dict, containing all of the scheduler's JobStatus
        objects to the job_record_file. The record file is overwritten each
        time.
        """
        copied_dict = False
        # if the file has already been written, we want to save the old version
        # in case something goes wrong with the dump
        if os.path.isfile(self.job_record_file):
            copied_dict = True
            system(f'mv {self.job_record_file} old_job_dict.pkl')

        with open(self.job_record_file, 'wb') as fout:
            pickle.dump(self.jobs, fout)

        # if nothing went wrong with dump, can delete the old job dict
        if copied_dict:
            system('rm old_job_dict.pkl')

    def read_job_dict(self):
        """
        Read a serialized job dictionary and set to the internal jobs dict
        """
        try:
            with open(self.job_record_file, 'rb') as fin:
                self.jobs = pickle.load(fin)
            self.logger.info(f'Read jobs dict from {self.job_record_file}')
        except FileNotFoundError:
            self.logger.info(f'{self.job_record_file} does not exist')

    def _replace_paths_in_files(
        self,
        directory: Union[str, PathLike],
        old_path: str,
        new_path: str,
    ):
        """
        Replace old path references with new path in text files.

        Helper method to scan files in the directory and replaces occurrences
        of the old path with the new path. Only processes text files to avoid
        corrupting binary files.

        :param directory: directory containing files to process
        :type directory: str or PathLike
        :param old_path: old path string to replace
        :type old_path: str
        :param new_path: new path string to replace with
        :type new_path: str
        """
        # remove extra . and / to catch abspath references
        old_path = os.path.normpath(old_path)
        new_path = os.path.normpath(new_path)

        # Common text file extensions that might contain paths
        text_extensions = {
            '.txt', '.sh', '.py', '.json', '.yaml', '.yml', '.xml', '.cfg',
            '.conf', '.ini', '.in', '.dat'
        }

        for root, _dirs, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                _, ext = os.path.splitext(filename)

                # Only process text files
                if ext.lower() not in text_extensions:
                    continue

                try:
                    # Read file content
                    with open(
                            file_path,
                            'r',
                            encoding='utf-8',
                            errors='ignore',
                    ) as f:
                        content = f.read()

                    # Check if old path is in the file
                    if old_path in content:
                        # Replace old path with new path
                        new_content = content.replace(old_path, new_path)

                        # Write back
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)

                        self.logger.info(
                            f"Replaced path references in {file_path}")
                except (IOError, UnicodeDecodeError) as e:
                    # Skip files that can't be read or aren't text files
                    self.logger.debug(f"Skipped {file_path}: {e}")

    def restart_job(
        self,
        calc_id: Union[int, str],
        command: Optional[str] = None,
        job_details: Optional[dict[str, Union[float, str]]] = None,
        copy_pattern: Optional[str] = "*",
        exclude_pattern: Optional[str] = None,
    ) -> Union[int, str]:
        """
        Restarts a job with the same or updated parameters.

        This function creates a new job based on an existing job. It:
        1. Gets the original job details from the provided calc_id
        2. Creates a new run directory
        3. Copies input files from original job directory to new directory
        4. Submits a new job with the original or updated job_details

        :param calc_id: ID of the original job to restart
        :type calc_id: int or str (depends on scheduler implementation)
        :param command: New command to run (if None, tries to use original
            command)
        :type command: str, optional
        :param job_details: Updated job details (merged with original)
        :type job_details: dict, optional
        :param copy_pattern: Shell glob pattern for files to copy (default "*")
        :type copy_pattern: str, optional
        :param exclude_pattern: Shell glob pattern for files to exclude (in
            addition to default exclusions)
        :type exclude_pattern: str, optional
        :returns: calc_id of the new job
        :rtype: int
        """
        # Get original job status and validate
        job_status = self.get_job_status(calc_id)
        if job_status is None:
            raise ValueError(f"Job with ID {calc_id} does not exist")

        original_path = job_status.path

        # Check if job is still running
        if hasattr(job_status, 'state') and job_status.state in [
                'running', 'pending', 'dependency', 'submitted', 'completing'
        ]:
            self.logger.warning(
                f"Job {calc_id} is still in state {job_status.state}. "
                "Continuing with restart anyway.")

        # Extract module and path_type from original path
        # Assuming path structure: root_directory/module/path_type/counter
        try:
            # Skip root_directory and get module and path_type
            rel_path = os.path.relpath(original_path, self.root_directory)
            path_parts = rel_path.split('/')
            module = path_parts[0]
            path_type = path_parts[1]
        except (IndexError, ValueError):
            raise ValueError("Cannot extract module and path_type from "
                             f" path: {original_path}")

        # Create new directory
        new_path = self.make_path(module, path_type)

        # If command not provided, try to get from JobStatus
        if command is None:
            if hasattr(job_status, 'command') and job_status.command:
                command = job_status.command
                self.logger.info(
                    f"Retrieved command from JobStatus: {command}")
            else:
                raise RuntimeError('command not found associated with job')

        # Define default exclusion patterns for output files
        default_excludes = [
            "*.log", "*.out", "*.err", "job_done", "slurm*", "batch*", "flux*"
        ]

        # Combine with user-provided exclusions
        all_excludes = set(default_excludes)
        if exclude_pattern:
            if isinstance(exclude_pattern, list):
                all_excludes.update(exclude_pattern)
            else:
                all_excludes.add(exclude_pattern)

        # Copy files from original to new directory
        files_to_copy = set()
        for pattern in [copy_pattern] if isinstance(copy_pattern,
                                                    str) else copy_pattern:
            files_to_copy.update(glob.glob(f"{original_path}/{pattern}"))

        files_to_exclude = set()
        for pattern in all_excludes:
            files_to_exclude.update(glob.glob(f"{original_path}/{pattern}"))

        for src_file in (files_to_copy - files_to_exclude):
            dst_file = os.path.join(new_path, os.path.basename(src_file))
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
                self.logger.info(f"Copied file {src_file} to {dst_file}")
            elif os.path.isdir(src_file):
                shutil.copytree(src_file, dst_file)
                self.logger.info(f"Copied directory {src_file} to {dst_file}")

        # Replace old path references in copied text files
        self._replace_paths_in_files(new_path, original_path, new_path)

        # Get original job details and merge with provided ones
        # Try to get from JobStatus first
        if hasattr(job_status, 'job_details') and job_status.job_details:
            original_job_details = job_status.job_details
            self.logger.info("Retrieved job_details from JobStatus")
        else:
            # Start with empty dict as default
            original_job_details = {}
            self.logger.info('No job_details associated with job')

        # Merge job details
        if job_details is None:
            merged_job_details = original_job_details
        else:
            merged_job_details = {**original_job_details, **job_details}

        # Submit new job
        self.logger.info(
            f"Submitting restarted job from original job ID: {calc_id}")
        new_calc_id = self.submit_job(command, new_path, merged_job_details)
        self.logger.info(f"Restarted job submitted with new ID: {new_calc_id}")

        return new_calc_id

    @abstractmethod
    def checkpoint_scheduler(self):
        """
        checkpoint the scheduler module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        pass

    @abstractmethod
    def restart_scheduler(self):
        """
        restart the scheduler module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        pass

    @abstractmethod
    def block_until_completed(self, calc_ids: list):
        """
        Function for enforcing synchronous execution

        Helper function to run when blocking behavior is desired. Will
        consistently check the queue until the specified job has completed.
        Implementation for synchronous schedulers is just to pass

        :param calc_ids: list of job IDs of the calculations to check for
            completion. Can also pass a single ID.
        :type calc_ids: list
        """
        pass

    @abstractmethod
    def submit_job(
        self,
        command: str,
        run_path: Union[str, PathLike],
        job_details: Optional[dict[str, Union[float, str]]] = None,
    ) -> Union[int, str]:
        """
        Submits a job for running

        submit_job handles job submission for the modules and is the main
        interface for the schedulers to be used. Inputs define the command to
        be executed for the job, location for the run, and details of the job
        resources. ``job_details`` inlcude ``dependencies`` of the job in the
        form of a list of job_ids, if the job is blocking (``synchronus``) or
        not, as a boolean, and an extra dictionary, ``extra_args``, to add
        flexibility for concrete implementations for further parameterizing the
        job. The ``dependencies`` are a list of job IDs which must have a
        successfully completed :class:`~JobStatus` for the present job to run.
        If one of the dependencies returns an error, this job will not run and
        the status will return an error. If synchronous is True, submit_job
        will not return until the job is completed. This may be the only
        behavior of some implementations. Returns a job handle that can be used
        to query status and to retrieve the present job's :class:`~JobStatus`,
        which is typically created by this method.

        :param command: command that defines the job to be executed
        :type command: implementation dependent
        :param run_path: location in the file system where inputs and outputs
            are to be accessed and stored
        :type run_path: str
        :param job_details: optional parameters for running the job, such as
            number of nodes, queue, etc. |default| ``None``
        :type job_details: dict
        :returns: return job ID to query this job status and location
        :rtype: Union[int, str]
        """
        pass


class HPCScheduler(Scheduler, ABC):
    """
    Generic (and abstract class) for schedulers leveraging HPC schedulers

    HPCScheduler defines the shared init args and restart functionality that
    LSF, Slurm, and other HPC schedulers require. It is not instantitated
    directly, but inherited.
    """

    def __init__(
        self,
        queue: str,
        account: str,
        walltime: Optional[Union[int, float]] = 60,
        nodes: Optional[int] = 1,
        tasks: Optional[int] = 1,
        tasks_per_node: Optional[int] = 1,
        qos: Optional[str] = 'normal',
        wait_freq: Optional[int] = 60,
        remote_machine: Optional[str] = None,
        **kwargs,
    ):
        """
        set variables and initialize the recorder

        The provided input arguments set the default parameters for the
        scheduler, but can be overridden by values passed into ``job_details``
        dict provided to :meth:`submit_job`.

        :param queue: default name of the queue to submit to
        :type queue: str
        :param account: default name of the account for the job
        :type account: str
        :param walltime: default walltime for the job in minutes |default| 60
        :type walltime: int or float
        :param nodes: default number of nodes to request |default| 1
        :type nodes: int
        :param tasks: default number of tasks for a job |default| 1
        :type tasks: int
        :param tasks_per_node: default number of tasks per node for a job.
            Will not be used if tasks is explicitly set. |default| 1
        :type tasks_per_node: int
        :param wait_freq: the frequency with which squeue is called to get job
            status updates, in seconds |default| 60
        :type wait_freq: int
        :param remote_machine: name of the machine to ssh to for job submission
            |default| ``None``
        :type remote_machine: str
        :param kwargs: remaining keywords passed to parent: root_directory,
            checkpoint_file, checkpoint_name, and job_record_file
        :type kwargs: dict
        """
        self.default_queue = queue
        self.default_account = account
        # value in minutes
        self.default_walltime = walltime
        self.default_nodes = nodes
        self.default_tasks = tasks
        self.default_tasks_per_node = tasks_per_node
        self.default_qos = qos
        self.unknown_job_id = 1000
        self.new_unknown_id = False
        self.remote_machine = remote_machine

        # should be overridden by child
        self.ID_TYPE = None

        # value in seconds
        self.synch_check_frequency = int(wait_freq)
        super().__init__(**kwargs)

    @staticmethod
    def _format_walltime(
        minutes: Union[float, int],
        include_seconds: bool,
    ) -> str:
        """
        utility function to create a time string based on input minutes

        different schedulers require time specifications with or without
        seconds, so this utility allows an integer input to be properly
        converted into a HH:MM or HH:MM:ss string

        :param minutes: number of minutes (can be fractional) to convert
        :type minutes: float or int
        :param include_seconds: whether to print out the seconds or not
        :type include_seconds: bool
        :returns: the formatted time string
        :rtype: str
        """
        if not include_seconds:
            minutes = round(minutes)
        hours = int(minutes // 60)
        remaining_minutes = int(minutes % 60)
        seconds = round((minutes - int(minutes)) * 60)
        if seconds == 60:
            seconds = 0
            remaining_minutes += 1
            if remaining_minutes == 60:
                remaining_minutes = 0
                hours += 1
        if include_seconds:
            return f'{hours:02}:{remaining_minutes:02}:{seconds:02}'
        else:
            return f'{hours:02}:{remaining_minutes:02}'

    def checkpoint_scheduler(self):
        """
        checkpoint the scheduler module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        self.save_job_dict()
        save_dict = {
            self.checkpoint_name: {
                'unknown_job': self.unknown_job_id,
                'counters': self.counters,
            }
        }
        if self.new_counters or self.new_unknown_id:
            restarter.write_checkpoint_file(self.checkpoint_file, save_dict)
            self.new_counters = False
            self.new_unknown_id = False

    def restart_scheduler(self):
        """
        restart the scheduler module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        # set the jobs dict from the pickle file
        self.read_job_dict()
        # see if any internal variables were checkpointed
        restart_dict = restarter.read_checkpoint_file(
            self.checkpoint_file,
            self.checkpoint_name,
        )
        self.unknown_job_id = restart_dict.get('unknown_job',
                                               self.unknown_job_id)
        self.counters = restart_dict.get('counters', self.counters)

    def block_until_completed(
        self,
        calc_ids: Union[int, str, list[int], list[str]],
    ):
        """
        Function for enforcing synchronous execution

        Helper function to run when blocking behavior is desired. Will
        consistently check the queue until the specified job has completed.

        :param calc_ids: list of job IDs of the calculations to check for
            completion. Can also pass a single ID.
        :type calc_ids: int, str, or list of ints or strings
        """
        if type(calc_ids) is list:
            if all(isinstance(cid, self.ID_TYPE) for cid in calc_ids):
                remaining_jobs = deepcopy(calc_ids)
            else:
                raise TypeError(f'calc_ids must be a list of {self.ID_TYPE}!')
        elif type(calc_ids) is self.ID_TYPE:
            remaining_jobs = [calc_ids]
        else:
            raise TypeError(f'Job IDs must be a single {self.ID_TYPE}, or a '
                            f'list of {self.ID_TYPE}!')

        wait_cycle_counter = 0
        while len(remaining_jobs) > 0:
            job_states = self.update_job_status(remaining_jobs)
            jobs_to_remove = []
            for job_id, job_state in zip(remaining_jobs, job_states):
                if job_state[:4] == 'done':
                    self.logger.info((f'Job {job_id} completed, removing '
                                      f'from waiting list'))
                    jobs_to_remove.append(job_id)
            # remove after iterating the whole list so no IDs are skipped
            for job_id in jobs_to_remove:
                remaining_jobs.remove(job_id)
            if len(remaining_jobs) == 0:
                break
            if wait_cycle_counter % 5 == 0:
                self.logger.info((f'Wait iteration {wait_cycle_counter}, ['
                                  f'cycle time = {self.synch_check_frequency}'
                                  f' s] with {len(remaining_jobs)} jobs left '
                                  f'to complete'))
            wait_cycle_counter += 1
            sleep(self.synch_check_frequency)
        self.logger.info(f'Jobs {calc_ids} have completed, continuing...')

    def generate_batch_file(
        self,
        command: str,
        run_path: Union[str, PathLike],
        job_details: Optional[dict[str, Union[float, str]]] = None,
        extra_args: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Construct a batch file for job submission

        This is a helper funciton for submit_job to call to construct a batch
        file for the proper scheduler.

        :param command: command that defines the job to be executed
        :type command: str
        :param run_path: directory for the job to be executed in
        :type run_path: str
        :param job_details: optional parameters for running the job.
            Parameters specified in this dict are: 'nodes', 'tasks',
            'tasks_per_node', 'queue', 'account', 'walltime', and
            'custom_preamble' |default| ``None``
        :type job_details: dict
        :param extra_args: dictionary of extra args, can include lines to pre-
            or postpend the job command in the batch script (given by the
            'preamble' and 'postamble' keys), additional #SCHEDULER commands
            (given by the 'extra_header' key where the value is a formatted
            string including the #SBATCH/#BSUB keyword) as well as alternative
            batch template location, specified by the 'template' key |default|
            ``None``
        :type extra_args: dict
        :returns: name of the batch file
        :rtype: str
        """
        if job_details is None:
            job_details = {}
        if extra_args is None:
            extra_args = {}
        # default template will be set by the concrete child classes
        file_template = extra_args.get('template', self.default_template)
        batch_file = Templates(file_template, run_path)

        patterns = [
            'nodes',
            'queue',
            'account',
            'walltime',
            'extra_header',
            'preamble',
            'command',
            'postamble',
        ]
        # use the child defined generate_job_preamble
        scheduler_preamble = self.generate_job_preamble(job_details)
        custom_preamble = job_details.get('custom_preamble', None)
        if custom_preamble is not None:
            launch_string = f'{custom_preamble} {command}'
        else:
            # run string defined for specific schedulers
            launch_string = f'{self.run_string} {scheduler_preamble} {command}'
        replacements = [
            job_details.get('nodes', self.default_nodes),
            job_details.get('queue', self.default_queue),
            job_details.get('account', self.default_account),
            self._format_walltime(
                job_details.get('walltime', self.default_walltime),
                self.USE_SEC,
            ),
            extra_args.get('extra_header', ''),
            extra_args.get('preamble', ''),
            launch_string,
            extra_args.get('postamble', ''),
        ]
        file_name = batch_file.replace(patterns, replacements)
        self.logger.info(f'Batch file written to {run_path}/{file_name}')
        return file_name

    def extract_job_id(self, str_output: str):
        """
        Template method for job ID extraction with common error handling.

        Calls scheduler-specific _parse_job_id method and handles common
        error cases with fallback to unknown_job_id.

        :param str_output: Output string from job submission command
        :type str_output: str
        :returns: Extracted job ID
        :rtype: int or str (depends on scheduler)
        :raises JobSubmissionError: If job ID cannot be extracted
        """
        try:
            job_id = self._parse_job_id(str_output)
            self.logger.info(f'Found job ID: {job_id}')
            return job_id
        except (AttributeError, IndexError, ValueError):
            self.logger.info(
                f'Could not extract job ID from output: {str_output.strip()}')
            self.logger.info(f'Using fallback ID: {self.unknown_job_id}')
            job_id = self.unknown_job_id
            self.unknown_job_id += 1
            self.new_unknown_id = True
            # Import here to avoid circular dependency
            from ..utils.exceptions import JobSubmissionError
            raise JobSubmissionError('Could not obtain job ID from submission '
                                     '- cannot continue at this time. Output '
                                     f'was: {str_output}')

    @abstractmethod
    def _parse_job_id(self, str_output: str):
        """
        Parse scheduler-specific output to extract job ID.

        Child classes must implement this to extract the job ID from
        the scheduler's submission output.

        :param str_output: Output string from scheduler submission command
        :type str_output: str
        :returns: Extracted job ID
        :rtype: int or str (depends on scheduler)
        :raises ValueError, IndexError, AttributeError: If parsing fails
        """
        pass

    def update_job_status(self, job_ids: list) -> list[str]:
        """
        Common status update logic with scheduler-specific queries.

        Queries the scheduler for job status updates, parses the results,
        updates internal job state tracking, and raises exceptions for
        problematic job states.

        :param job_ids: List of job IDs to check for status updates
        :type job_ids: list[int] or list[str]
        :returns: List of updated job states
        :rtype: list[str]
        :raises ProblematicJobStateError: If any jobs are in problematic states
        """
        status_changed = False
        problematic_jobs = []

        # Build and execute query (scheduler-specific)
        query_command = self._build_status_query_command(job_ids)
        query_output = sp.run(query_command,
                              capture_output=True,
                              shell=True,
                              encoding='UTF-8')

        updated_states = []
        if query_output.returncode != 0 or query_output.stderr:
            self.logger.info((f'Problem checking for jobs {job_ids}, '
                              f'exit code: {query_output.returncode} '
                              f'stderr: {query_output.stderr.strip()}'))
            # Handle errors (may need scheduler-specific handling)
            updated_states = self._handle_query_error(job_ids, query_output)
            # Check if any states changed
            for job_id, new_state in zip(job_ids, updated_states):
                known_status = self.get_job_status(job_id)
                if new_state != known_status.state:
                    status_changed = True
                if new_state in self.problematic_states:
                    problematic_jobs.append((job_id, new_state))
        else:
            split_output = query_output.stdout.split()
            for job_id in job_ids:
                known_status = self.get_job_status(job_id)
                new_state = self._parse_job_state(
                    job_id, split_output)  # Scheduler-specific

                if new_state != known_status.state:
                    self.logger.info(f'Updating job {job_id} state from '
                                     f'{known_status.state} to {new_state}')
                    known_status.state = new_state
                    status_changed = True

                if new_state in self.problematic_states:
                    problematic_jobs.append((job_id, new_state))

                updated_states.append(known_status.state)

        if status_changed:
            self.checkpoint_scheduler()

        # Raise exception after processing all jobs
        if problematic_jobs:
            job_list = ', '.join(
                [f'{jid} ({state})' for jid, state in problematic_jobs])
            from ..utils.exceptions import ProblematicJobStateError
            raise ProblematicJobStateError(
                f'Orchestrator does not currently have set behavior for '
                f'problematic job states. Check your queue for any remaining '
                f'pending jobs. Problematic jobs: {job_list}')

        return updated_states

    @abstractmethod
    def _build_status_query_command(self, job_ids: list) -> str:
        """
        Build scheduler-specific status query command.

        :param job_ids: List of job IDs to query
        :type job_ids: list[int] or list[str]
        :returns: Command string to query job statuses
        :rtype: str
        """
        pass

    @abstractmethod
    def _parse_job_state(self, job_id, split_output: list) -> str:
        """
        Parse scheduler-specific state from query output.

        Maps scheduler-specific states to common scheduler states:
        'pending', 'dependency', 'running', 'completing', 'done',
        'done_timeout', 'done_cancelled', 'done_other', 'done_unknown',
        'unknown', 'error'

        :param job_id: Job ID to parse state for
        :type job_id: int or str
        :param split_output: Split output from status query command
        :type split_output: list[str]
        :returns: Parsed job state
        :rtype: str
        """
        pass

    def _handle_query_error(
        self,
        job_ids: list,
        query_output: sp.CompletedProcess,
    ) -> list[str]:
        """
        Handle errors from status query command.

        Default implementation returns empty list, but can be overridden
        by child classes for scheduler-specific error handling.

        :param job_ids: List of job IDs that were queried
        :type job_ids: list[int] or list[str]
        :param query_output: Output from failed query command
        :type query_output: subprocess.CompletedProcess
        :returns: List of job states (may be empty or partial)
        :rtype: list[str]
        """
        return []

    def submit_job(
        self,
        command: str,
        run_path: Union[str, PathLike],
        job_details: Optional[dict[str, Union[float, str]]] = None,
    ):
        """
        Common job submission logic with scheduler-specific details.

        Submits a job to the scheduler with the given command and parameters.
        Handles dependency management, synchronous execution, and error cases.

        :param command: Command that defines the job to be executed
        :type command: str
        :param run_path: Directory for the job to be executed in
        :type run_path: str or PathLike
        :param job_details: Specifics for running the job, such as number of
            nodes, queue, dependencies, synchronous flag, etc.
        :type job_details: dict, optional
        :returns: Job ID to query this job status and location
        :rtype: int or str (depends on scheduler)
        """
        from ..utils.exceptions import UnfullfillableDependenciesError
        from ..utils.data_standard import METADATA_KEY

        if job_details is None:
            job_details = {}
            self._log_default_job_details()

        synchronous = job_details.get('synchronous', False)
        dependencies = job_details.get('dependencies', [])
        extra_args = job_details.get('extra_args', {})

        job_can_run = True
        calc_id = -1
        exit_code = 'undefined error'

        if not command:
            job_can_run = False
            exit_code = 'empty command'
            self.logger.info('Job will not run: no command')

        if job_can_run:
            # Generate batch file (already implemented in base class)
            batch_file = self.generate_batch_file(
                command,
                run_path,
                job_details,
                extra_args,
            )

            # Build dependency string (scheduler-specific)
            if dependencies:
                self.logger.info(f'Including dependencies: {dependencies}')
                depend_str = self._build_dependency_string(
                    dependencies, extra_args)
            else:
                depend_str = ''

            # Build and execute submit command (scheduler-specific)
            submit_command = self._build_submit_command(
                run_path, batch_file, depend_str)

            self.logger.info('Spawning job, ID to be defined')
            process_output = sp.run(submit_command,
                                    capture_output=True,
                                    shell=True,
                                    encoding='UTF-8')
            exit_code = process_output.returncode

            # Extract job ID
            if exit_code != 0:
                self.logger.info(f'Bad submission, exit code = {exit_code}')
                calc_id = self.extract_job_id(process_output.stderr)
            else:
                calc_id = self.extract_job_id(process_output.stdout)

            # Create JobStatus
            metadata = extra_args.get(METADATA_KEY, {})
            job_status = JobStatus(
                run_path,
                'submitted',
                exit_code,
                command=command,
                job_details=job_details,
                metadata=metadata,
            )
        else:
            # Handle job that can't run
            calc_id = self.unknown_job_id
            self.unknown_job_id += 1
            self.new_unknown_id = True
            metadata = extra_args.get(METADATA_KEY, {})
            job_status = JobStatus(
                run_path,
                'done_cancelled',
                exit_code,
                command=command,
                job_details=job_details,
                metadata=metadata,
            )
            if dependencies:
                raise UnfullfillableDependenciesError()

        self.jobs[calc_id] = job_status
        self.checkpoint_scheduler()
        if synchronous:
            self.block_until_completed(calc_id)
        return calc_id

    @abstractmethod
    def _log_default_job_details(self):
        """
        Log default job details when none are provided.

        Child classes should log scheduler-specific defaults.
        """
        pass

    @abstractmethod
    def _build_dependency_string(
        self,
        dependencies: list,
        extra_args: dict,
    ) -> str:
        """
        Build scheduler-specific dependency string.

        :param dependencies: List of job IDs that this job depends on
        :type dependencies: list[int] or list[str]
        :param extra_args: Extra arguments that may affect dependency format
        :type extra_args: dict
        :returns: Dependency string for scheduler submission command
        :rtype: str
        """
        pass

    @abstractmethod
    def _build_submit_command(
        self,
        run_path: Union[str, PathLike],
        batch_file: str,
        depend_str: str,
    ) -> str:
        """
        Build scheduler-specific submit command.

        :param run_path: Directory where the job will be executed
        :type run_path: str or PathLike
        :param batch_file: Name of the batch file to submit
        :type batch_file: str
        :param depend_str: Dependency string (may be empty)
        :type depend_str: str
        :returns: Complete submission command
        :rtype: str
        """
        pass
