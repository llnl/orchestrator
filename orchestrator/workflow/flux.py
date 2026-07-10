from abc import ABC
from os import path, PathLike, getcwd
from typing import Optional, Union
from .workflow_base import HPCWorkflow


class FluxWF(HPCWorkflow, ABC):
    """
    Workflow manager for full execution using a batch file and Flux scheduler.

    FluxWF is a fully featured workflow module for submitting jobs to
    the Flux scheduler using batch files while Orchestrator is running on
    another machine. It can handle asynchronous job submission but still
    provides the option for blocking (synchronous) behavior. Responsibilities
    include directory creation, job creation, and job status checking.
    """

    def __init__(
        self,
        default_submit_command: str = 'batch',
        default_template: Optional[str] = None,
        **kwargs,
    ):
        """
        set variables and initialize the recorder

        :param default_submit_command: command to use for job submission
            |default| ``batch``
        :type default_submit_command: str
        :param default_template: path to the template file to use for
            submission scripts. If none provided, uses the default template
            present in ./default_templates |default| ``None``
        :type default_template: str
        :param kwargs: remaining parameters passed to parent for init. Keys
            include: queue, account, walltime, nodes, tasks, tasks_per_node,
            wait_freq, remote_machine, root_directory, checkpoint_file,
            checkpoint_name, and job_record_file
        :type kwargs: dict
        """
        super().__init__(**kwargs)
        self.default_submit_command = default_submit_command
        if default_template is None:
            source_file_location = path.dirname(path.abspath(__file__))
            self.default_template = (f'{source_file_location}/'
                                     'default_templates/flux.sh')
        else:
            self.default_template = default_template
        # determines print format of walltime strings
        self.USE_SEC = False
        self.ID_TYPE = str
        self.run_string = 'flux run'

    def _parse_job_id(self, str_output: str) -> str:
        """
        Parse Flux-specific output to extract job ID.

        :param str_output: Output string from flux submit
        :type str_output: str
        :returns: Flux job ID
        :rtype: str
        :raises ValueError: If job ID format is invalid
        """
        strip_split_line = str_output.strip().split()
        if len(strip_split_line) == 1:
            flux_id = strip_split_line[0]
            if len(flux_id) == 12 and flux_id[0] == 'f':
                return flux_id
            else:
                raise ValueError(f'Flux ID [{flux_id}] not in expected format')
        else:
            raise ValueError(
                f'Returned output [{str_output}] not in expected format')

    def generate_job_preamble(
        self,
        job_details: dict[str, Union[float, str]],
    ) -> str:
        """
        Set Flux arguments from job_details or from defaults.

        :param job_details: dict passed through :meth:`~submit_job` including
            any desired alterations from the workflow defaults
        :type job_details: dict
        :returns: populated preamble string
        :rtype: str
        """
        node_val = job_details.get('nodes', self.default_nodes)
        task_val = job_details.get('tasks', self.default_tasks)
        tasks_per_node_val = job_details.get('tasks_per_node',
                                             self.default_tasks_per_node)

        if tasks_per_node_val > 1:
            if task_val > 1:
                self.logger.info((f'Warning: tasks and tasks-per-node are '
                                  f'both specified. Using tasks = {task_val}'))
            else:
                # only tasks_per_node specified, but not supported by flux
                # convert to task_val
                task_val = tasks_per_node_val * node_val

        # task_val > 1, set by tasks_per_node_val, or both are defaults (1)
        # in all cases, use -n with task_val
        job_arg_string = (f'-N{node_val} -n{task_val}')

        return job_arg_string

    def _build_status_query_command(self, job_ids: list[str]) -> str:
        """
        Build Flux-specific status query command.

        :param job_ids: List of Flux job IDs to query
        :type job_ids: list[str]
        :returns: Command string to query job statuses
        :rtype: str
        """
        job_str = ' '.join(job_ids)
        if self.remote_machine is None:
            return f'flux jobs --format "{{id}} {{status}}" {job_str}'
        else:
            return (f"ssh {self.remote_machine} '"
                    f'flux jobs --format "{{id}} {{status}}" {job_str}\'')

    def _parse_job_state(self, job_id: str, split_output: list) -> str:
        """
        Parse Flux-specific state from query output.

        :param job_id: Flux job ID to parse state for
        :type job_id: str
        :param split_output: Split output from flux jobs command
        :type split_output: list[str]
        :returns: Parsed job state
        :rtype: str
        """
        try:
            flux_id_index = split_output.index(job_id)
            flux_state = split_output[flux_id_index + 1]

            # Map Flux states to workflow states
            if flux_state in ['PRIORITY', 'SCHED']:
                return 'pending'
            elif flux_state == 'DEPEND':
                return 'dependency'
            elif flux_state in ['RUN', 'CLEANUP']:
                return 'running'
            elif flux_state == 'COMPLETED':
                return 'done'
            elif flux_state == 'FAILED':
                return 'done_failed'
            elif flux_state == 'CANCELED':
                return 'done_cancelled'
            elif flux_state == 'TIMEOUT':
                return 'done_timeout'
            else:
                return 'unknown'
        except ValueError:
            # flux id not in list, so state unknown
            self.logger.info((f'Cannot find {job_id} with flux jobs, '
                              f'set state to unknown'))
            return 'unknown'
        except Exception:
            self.logger.info((f'Job {job_id} state parsing had an '
                              f' unknown error, set state to "error"'))
            return 'error'

    def _log_default_job_details(self):
        """Log default Flux job details when none are provided."""
        self.logger.info(f'No job details specified, will use defaults:\n'
                         f'nnodes = {self.default_nodes}, walltime = '
                         f'{self.default_walltime}, queue = '
                         f'{self.default_queue}, flux submit command = '
                         f'{self.default_submit_command}')

    def _build_dependency_string(
        self,
        dependencies: list,
        extra_args: dict,
    ) -> str:
        """
        Build Flux-specific dependency string.

        :param dependencies: List of Flux job IDs that this job depends on
        :type dependencies: list[str]
        :param extra_args: Extra arguments, may contain 'after' key
        :type extra_args: dict
        :returns: Dependency string for flux submission command
        :rtype: str
        """
        after_type = extra_args.get('after', 'afterok')
        # format the list to remove [] and spaces between commas
        depend_list = [f'--dependency={after_type}:{d}' for d in dependencies]
        return ' ' + ' '.join(depend_list)

    def _build_submit_command(
        self,
        run_path: Union[str, PathLike],
        batch_file: str,
        depend_str: str,
    ) -> str:
        """
        Build Flux-specific submit command.

        :param run_path: Directory where the job will be executed
        :type run_path: str or PathLike
        :param batch_file: Name of the batch file to submit
        :type batch_file: str
        :param depend_str: Dependency string (may be empty)
        :type depend_str: str
        :returns: Complete submission command
        :rtype: str
        """
        # Get submit command from job_details or use default
        # no job_details here so pass value via _current_submit_command
        submit_str = getattr(self, '_current_submit_command',
                             self.default_submit_command)

        if self.remote_machine is None:
            return (f'cd {run_path}; flux {submit_str}'
                    f'{depend_str} {batch_file}')
        else:
            cwd = getcwd()
            return (f'ssh {self.remote_machine} '
                    f'"source /etc/profile; cd {cwd}/{run_path};'
                    f' flux {submit_str}{depend_str} {batch_file}"')

    def submit_job(
        self,
        command: str,
        run_path: Union[str, PathLike],
        job_details: Optional[dict[str, Union[float, str]]] = None,
    ) -> str:
        """
        Submits a job for running using a submission script and Flux.

        See parent class for full documentation. Flux-specific behavior
        includes support for custom submit commands via the 'submit_command'
        key in job_details.

        :param command: command that defines the job to be executed
        :type command: str
        :param run_path: directory for the job to be executed in
        :type run_path: str
        :param job_details: specifics for running the job
        :type job_details: dict
        :returns: return job ID to query this job status and location
        :rtype: str
        """
        # Store submit command for _build_submit_command to use
        if job_details is not None:
            self._current_submit_command = job_details.get(
                'submit_command', self.default_submit_command)
        else:
            self._current_submit_command = self.default_submit_command

        # Call parent implementation
        return super().submit_job(command, run_path, job_details)

    @staticmethod
    def _format_walltime(
        minutes: Union[float, int],
        include_seconds: bool,
    ) -> str:
        """
        utility function to create a time string based on input minutes

        The flux scheduler will generically work with only a minute specifier,
        and does not support mixed units.

        :param minutes: number of minutes (can be fractional) to convert
        :type minutes: float or int
        :param include_seconds: whether to print out the seconds or not. FSD
            supports fractional minutes. This argument is not used.
        :type include_seconds: bool
        :returns: the formatted time string
        :rtype: str
        """
        return f'{minutes:02}m'
