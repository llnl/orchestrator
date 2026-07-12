from abc import ABC
from os import path, PathLike, getcwd
import subprocess as sp
from typing import Optional, Union
from .scheduler_base import HPCScheduler


class SlurmScheduler(HPCScheduler, ABC):
    """
    Scheduler manager for full execution using a batch file and slurm scheduler

    SlurmScheduler is a fully featured scheduler module for submitting jobs to
    the slurm scheduler using batch files. It can handle asynchronus job
    submission but still provides the option for blocking (synchronous)
    behavior. Responsibilities include directory creation, job creation, job
    status checking.
    """

    def __init__(self, default_template: Optional[str] = None, **kwargs):
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
                                     'default_templates/slurm.sh')
        else:
            self.default_template = default_template
        # determines print format of walltime strings
        self.USE_SEC = True
        self.ID_TYPE = int
        self.run_string = 'srun'

    def _parse_job_id(self, str_output: str) -> int:
        """
        Parse Slurm-specific output to extract job ID.

        :param str_output: Output string from sbatch or srun
        :type str_output: str
        :returns: Slurm job ID
        :rtype: int
        :raises ValueError: If job ID format is invalid
        """
        split_output = str_output.split()
        # check if expected output format is present
        if split_output[0] == 'Submitted' and split_output[2] == 'job':
            # output from sbatch: "Submitted batch job 12345"
            return int(split_output[3])
        elif split_output[0] == 'srun:' and split_output[1] == 'job':
            # output from srun: "srun: job 12345 ..."
            return int(split_output[2])
        else:
            raise ValueError(
                f'Output string is unexpected format: {str_output.strip()}')

    def generate_job_preamble(
        self,
        job_details: dict[str, Union[float, str]],
    ) -> str:
        """
        Set slurm args from job_details or defaults defined by the Scheduler

        This is a helper function for constructing the preamble of the srun
        command. Values set are nodes, tasks (optional)

        :param job_details: dict passed through :meth:`~submit_job` including
            any desired alterations from the scheduler defaults
        :type job_details: dict
        :returns: populated preable string
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
            job_arg_string = (f'-N {node_val} -n {task_val}')
        else:
            # either tasks_per_node_val > 1, or both tasks-per-node and tasks
            # = 1. In both cases, desireable to use tasks-per-node so if nodes
            # > 1, can benefit from the distributed memory setup
            job_arg_string = (
                f'-N {node_val} --ntasks-per-node {tasks_per_node_val}')
        return job_arg_string

    def check_completed_job_status(self, slurm_id: int) -> str:
        """
        Use scontrol to extract the status of a completed job

        :param slurm_id: job ID of the job to query
        :type slurm_id: int
        :returns: job status
        :rtype: str
        """
        if self.remote_machine is None:
            control_command = f'scontrol show job {slurm_id}'
        else:
            control_command = (f"ssh {self.remote_machine} 'source /etc/"
                               f"profile; scontrol show job {slurm_id}'")

        control_output = sp.run(control_command,
                                capture_output=True,
                                shell=True,
                                encoding='UTF-8')
        if control_output.returncode != 0:
            # can't find job
            if self.job_done_file_present(slurm_id):
                self.logger.info('scontrol query return code nonzero, but '
                                 'job_done file present')
                state = 'done'
            else:
                state = 'done_unknown'
        else:
            # will get full report, JobState at 10 after =
            job_state = control_output.stdout.split()[10].split('=')[1]
            if job_state == 'COMPLETED':
                state = 'done'
            elif job_state == 'TIMEOUT':
                state = 'done_timeout'
            elif job_state == 'CANCELLED':
                state = 'done_cancelled'
            else:
                state = 'done_other'
        return state

    def _build_status_query_command(self, job_ids: list[int]) -> str:
        """
        Build Slurm-specific status query command.

        :param job_ids: List of Slurm job IDs to query
        :type job_ids: list[int]
        :returns: Command string to query job statuses
        :rtype: str
        """
        job_str = f'{job_ids[0]}'
        if len(job_ids) > 1:
            for remaining_id in job_ids[1:]:
                job_str += f',{remaining_id}'

        if self.remote_machine is None:
            return f'squeue -j {job_str} -o "%i %t %R"'
        else:
            return (f"ssh {self.remote_machine} 'source /etc/profile;"
                    f' squeue -j {job_str} -o "%i %t %R"\'')

    def _parse_job_state(self, job_id: int, split_output: list) -> str:
        """
        Parse Slurm-specific state from query output.

        :param job_id: Slurm job ID to parse state for
        :type job_id: int
        :param split_output: Split output from squeue command
        :type split_output: list[str]
        :returns: Parsed job state
        :rtype: str
        """
        try:
            slurm_str_index = split_output.index(str(job_id))
            slurm_state = split_output[slurm_str_index + 1]
            if slurm_state == 'CG':
                return 'completing'
            elif slurm_state == 'R':
                return 'running'
            elif slurm_state == 'PD':
                reason = split_output[slurm_str_index + 2][1:-1]
                if reason == 'Dependency':
                    return 'dependency'
                else:
                    return 'pending'
            else:
                return 'unknown'
        except ValueError:
            # slurm id not in list, so query job status with scontrol
            known_status = self.get_job_status(job_id)
            if known_status.state[:4] != 'done':
                # completed job state has not been assigned
                return self.check_completed_job_status(job_id)
            else:
                # scontrol already called, no changes
                return known_status.state
        except Exception:
            self.logger.info((f'Job {job_id} state parsing had an '
                              f' unknown error, set state to "error"'))
            return 'error'

    def _handle_query_error(
        self,
        job_ids: list[int],
        query_output: sp.CompletedProcess,
    ) -> list[str]:
        """
        Handle Slurm-specific query errors.

        Slurm removes jobs from the queue after they complete, so we need
        special handling to check for job_done files.

        :param job_ids: List of Slurm job IDs that were queried
        :type job_ids: list[int]
        :param query_output: Output from failed query command
        :type query_output: subprocess.CompletedProcess
        :returns: List of job states
        :rtype: list[str]
        """
        updated_states = []
        # possible that the job is just not in database anymore
        if query_output.stderr and 'Invalid job id' in query_output.stderr:
            status_changed = False
            problematic_jobs = []
            for job_id in job_ids:
                known_status = self.get_job_status(job_id)
                if self.job_done_file_present(job_id):
                    new_state = 'done'
                    self.logger.info((f'Updating job {job_id} state '
                                      f'from {known_status.state} to '
                                      f'{new_state} based on job_done file'))
                    known_status.state = new_state
                    status_changed = True
                else:
                    new_state = 'error'
                    self.logger.info((f'Updating job {job_id} state '
                                      f'from {known_status.state} to '
                                      f'{new_state}'))
                    known_status.state = new_state
                    status_changed = True
                    # collect for after all updates
                    problematic_jobs.append((job_id, new_state))
                updated_states.append(known_status.state)

            if status_changed:
                self.checkpoint_scheduler()
        return updated_states

    def _log_default_job_details(self):
        """Log default Slurm job details when none are provided."""
        self.logger.info((f'No job details specified, will use defaults:\n'
                          f'  N = {self.default_nodes}, A = '
                          f'{self.default_account}, t = '
                          f'{self.default_walltime}, p = '
                          f'{self.default_queue}'))

    def _build_dependency_string(
        self,
        dependencies: list,
        extra_args: dict,
    ) -> str:
        """
        Build Slurm-specific dependency string.

        :param dependencies: List of Slurm job IDs that this job depends on
        :type dependencies: list[int]
        :param extra_args: Extra arguments, may contain 'after' key
        :type extra_args: dict
        :returns: Dependency string for sbatch command
        :rtype: str
        """
        after_type = extra_args.get('after', 'afterany')
        # format the list to remove [] and spaces between commas
        no_space_list = ''.join(str(dependencies)[1:-1].split())
        # swap commas for : to separate job ids
        no_space_list = no_space_list.replace(',', ':')
        return f'-d {after_type}:{no_space_list}'

    def _build_submit_command(
        self,
        run_path: Union[str, PathLike],
        batch_file: str,
        depend_str: str,
    ) -> str:
        """
        Build Slurm-specific submit command.

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
            return f'cd {run_path}; sbatch {depend_str} {batch_file}'
        else:
            cwd = getcwd()
            return (f'ssh {self.remote_machine} "source /etc/profile; '
                    f'cd {cwd}/{run_path}; '
                    f'sbatch {depend_str} {batch_file}"')
