from abc import ABC
from os import path, PathLike, getcwd
from typing import Optional, Union
from .workflow_base import HPCWorkflow


class LSFWF(HPCWorkflow, ABC):
    """
    Workflow manager for full execution using a batch file and LSF scheduler

    LSFWF is a fully featured workflow module for submitting jobs to
    the LSF scheduler using batch files while Orchestrator is running on an LSF
    or different machine. It can handle asynchronus job submission but still
    provides the option for blocking (synchronous) behavior. Responsibilities
    include directory creation, job creation, job status checking.
    """

    def __init__(
        self,
        default_template: Optional[str] = None,
        **kwargs,
    ):
        """
        set variables and initialize the recorder

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
        if default_template is None:
            source_file_location = path.dirname(path.abspath(__file__))
            self.default_template = (f'{source_file_location}/'
                                     'default_templates/lsf.sh')
        else:
            self.default_template = default_template
        # determines print format of walltime strings
        self.USE_SEC = False
        self.ID_TYPE = int
        # best guesses for default paths
        lrun_path_df = '/usr/tcetmp/bin/lrun'
        jsrun_path_df = '/usr/tcetmp/bin/jsrun'
        lsf_profile_path_df = '/opt/ibm/spectrumcomputing/lsf/conf/profile.lsf'
        if self.remote_machine is None:
            self.run_string = 'lrun'
            self.jsrun_string = 'jsrun'
            self.lsf_profile = ''
        else:
            self.run_string = kwargs.get('lrun_path', lrun_path_df)
            self.jsrun_string = kwargs.get('jsrun_path', jsrun_path_df)
            self.lsf_profile = kwargs.get('lsf_profile_path',
                                          lsf_profile_path_df)

    def _parse_job_id(self, str_output: str) -> int:
        """
        Parse LSF-specific output to extract job ID.

        :param str_output: Output string from bsub
        :type str_output: str
        :returns: LSF job ID
        :rtype: int
        :raises ValueError: If job ID format is invalid
        """
        split_output = str_output.split()
        # check if expected output format is present
        if split_output[0] == 'Job' and split_output[2] == 'is':
            # output from bsub: "Job <12345> is ..."
            return int(split_output[1][1:-1])
        else:
            raise ValueError(
                f'Output string is unexpected format: {str_output.strip()}')

    def generate_job_preamble(
        self,
        job_details: dict[str, Union[float, str]],
    ) -> str:
        """
        Set LSF arguments from job_details or from defaults defined by the WF

        This is a helper function for constructing the preamble of the lrun
        command. Values set are nodes, tasks (optional).

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

        if task_val > 1:
            if tasks_per_node_val > 1:
                self.logger.info((f'Warning: tasks and tasks-per-node are '
                                  f'both specified. Using tasks = {task_val}'))
            job_arg_string = (f'-N{node_val} -n{task_val}')
        else:
            # either tasks_per_node_val > 1, or both tasks-per-node and tasks
            # = 1. In both cases, desireable to use tasks-per-node so if nodes
            # > 1, can benefit from the distributed memory setup
            job_arg_string = (f'-N{node_val} -T{tasks_per_node_val}')

        if 'jsrun' in job_arg_string and not job_arg_string.startswith('/'):
            job_arg_string = job_arg_string.replace('jsrun', self.jsrun_string)

        return job_arg_string

    def _build_status_query_command(self, job_ids: list[int]) -> str:
        """
        Build LSF-specific status query command.

        :param job_ids: List of LSF job IDs to query
        :type job_ids: list[int]
        :returns: Command string to query job statuses
        :rtype: str
        """
        job_str = ' '.join([str(x) for x in job_ids])
        # note that this command may need to be modified
        # for different compute systems
        if self.remote_machine is None:
            return (f'bquery -o "id stat pendstate dependency exit_reason" '
                    f'{job_str}')
        else:
            return (f"ssh {self.remote_machine} '"
                    f"source {self.lsf_profile}; "
                    f'bquery -o "id stat pendstate dependency exit_reason" '
                    f"{job_str}'")

    def _parse_job_state(self, job_id: int, split_output: list) -> str:
        """
        Parse LSF-specific state from query output.

        :param job_id: LSF job ID to parse state for
        :type job_id: int
        :param split_output: Split output from bquery command
        :type split_output: list[str]
        :returns: Parsed job state
        :rtype: str
        """
        try:
            lsf_str_index = split_output.index(str(job_id))
            lsf_state = split_output[lsf_str_index + 1]
            if lsf_state == 'DONE':
                return 'done'
            elif lsf_state == 'RUN':
                return 'running'
            elif lsf_state == 'PEND':
                pend_state = split_output[lsf_str_index + 2]
                if pend_state == 'IPEND':
                    dependency = split_output[lsf_str_index + 3]
                    if dependency == '-':
                        return 'pending_blocked'
                    else:
                        return 'dependency'
                else:
                    return 'pending'
            elif lsf_state == 'EXIT':
                exit_reason = split_output[lsf_str_index + 4][:-1]
                kill_word = split_output[lsf_str_index + 6]
                if exit_reason == 'TERM_RUNLIMIT':
                    return 'done_timeout'
                elif exit_reason == 'TERM_OWNER' and kill_word == 'killed':
                    return 'done_cancelled'
                else:
                    return 'done_other'
            else:
                return 'unknown'
        except ValueError:
            # LSF id not in list, so state unknown
            self.logger.info((f'Cannot find {job_id} with bquery, '
                              f'set state to unknown'))
            return 'unknown'
        except Exception:
            self.logger.info((f'Job {job_id} state parsing had an '
                              f' unknown error, set state to "error"'))
            return 'error'

    def _log_default_job_details(self):
        """Log default LSF job details when none are provided."""
        self.logger.info((f'No job details specified, will use defaults:\n'
                          f'  nnodes = {self.default_nodes}, G = '
                          f'{self.default_account}, W = '
                          f'{self.default_walltime}, q = '
                          f'{self.default_queue}'))

    def _build_dependency_string(
        self,
        dependencies: list,
        extra_args: dict,
    ) -> str:
        """
        Build LSF-specific dependency string.

        :param dependencies: List of LSF job IDs that this job depends on
        :type dependencies: list[int]
        :param extra_args: Extra arguments, may contain 'after' key
        :type extra_args: dict
        :returns: Dependency string for bsub command
        :rtype: str
        """
        # after type for LSF should be exit (= afterany) or done (= afterok)
        after_type = extra_args.get('after', 'exit')
        # build up the depend string
        depend_str = f'-w "{after_type}({dependencies[0]})'
        if len(dependencies) > 1:
            for remaining_id in dependencies[1:]:
                depend_str += f' && {after_type}({remaining_id})'
        depend_str += '"'
        return depend_str

    def _build_submit_command(
        self,
        run_path: Union[str, PathLike],
        batch_file: str,
        depend_str: str,
    ) -> str:
        """
        Build LSF-specific submit command.

        :param run_path: Directory where the job will be executed
        :type run_path: str or PathLike
        :param batch_file: Name of the batch file to submit
        :type batch_file: str
        :param depend_str: Dependency string (may be empty)
        :type depend_str: str
        :returns: Complete submission command
        :rtype: str
        """
        if self.remote_machine is None:
            return f'cd {run_path}; bsub {depend_str} {batch_file}'
        else:
            cwd = getcwd()
            # note that this command may need to be modified
            # for different compute systems
            return (f'ssh {self.remote_machine} "'
                    f'cd {cwd}/{run_path}; '
                    f'source {self.lsf_profile}; '
                    f'bsub {depend_str} {batch_file}"')
