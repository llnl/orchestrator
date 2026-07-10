import math
from os import PathLike
from abc import ABC
from copy import deepcopy
from time import sleep
from typing import Union
from .workflow_base import HPCWorkflow, JobStatus
from ..utils.data_standard import METADATA_KEY

from aiida import load_profile
from aiida.orm import QueryBuilder, ProcessNode, load_node, CalcJobNode
from aiida.engine import submit
from aiida.engine.processes.builder import ProcessBuilder
from aiida.manage.configuration import get_config
from aiida.engine.daemon.client import get_daemon_client, DaemonClient


class AiidaWF(HPCWorkflow, ABC):
    """
    Workflow class for using AiiDA

    Note that ``kwargs`` set the default parameters for the workflow,
    but can be overridden by values passed into ``job_details`` provided to
    :meth:`submit_job`.
    """

    def __init__(self, **kwargs: dict):
        """
        Initialize the values needed for the AiiDA WF.

        :param kwargs: arguments to control workflow behavior, keys
            may include root_directory, checkpoint_file, checkpoint_name,
            and job_record. These will be defaulted to
            './orchestrator_workflow' and './orchestrator_checkpoint.json',
            'workflow', and './job_record.pkl', respectively. AiiDA
            additionally uses 'queue' -> 'pbatch', 'account' -> 'iap',
            'walltime' -> '1:00', 'nodes' -> 1, 'tasks' -> 1, and
            'tasks_per_node' -> 1
        :type kwargs: dict
        """
        super().__init__(**kwargs)
        self.check_daemon_status()
        self.ID_TYPE = int

    def check_daemon_status(self):
        """
        Check the status of the daemon used for AiiDA. If the daemon is not
        currently running, will attempt to start the daemon.
        """

        client = self.get_aiida_daemon_client()

        try:
            # Check that the daemon has not become stale.
            if client._is_pid_file_stale:
                # If it is stale, stop the daemon and start again.
                client._clean_potentially_stale_pid_file()
                client.start_daemon()
            elif not client.is_daemon_running:
                self.logger.info('No AiiDA daemon was found to be running. '
                                 'Starting the daemon now.')
                client.start_daemon()
                pid = client.get_daemon_pid()
                self.logger.info(f'AiiDA daemon started ({pid=}).')
        except RuntimeError:
            msg = ('The daemon is not running and was unable to be started. '
                   'Fix the daemon and attempt the job after.')
            self.logger.warning(msg)
            raise RuntimeError(msg)

    def check_daemon_workload(self, workload_target: float = 0.9):
        """
        Will check the workload of the daemon and set to the designated
        workload target.

        :param workload_target: Target value to set the workload. Is used to
            decided the number of daemon workers needed. Default value is set
            to 90%.
        """

        client = self.get_aiida_daemon_client()
        active_workers = client.get_numprocesses()['numprocesses']

        config = get_config()

        slots_per_worker = config.get_option('daemon.worker_process_slots',
                                             scope=self.default_profile_name)
        ap = (QueryBuilder().append(ProcessNode,
                                    filters={
                                        'attributes.process_state': {
                                            'in':
                                            ('created', 'waiting', 'running')
                                        }
                                    }).count())

        needed_workers = math.ceil(ap / (workload_target * slots_per_worker))

        # Always keep one worker running.
        if needed_workers == 0 and active_workers == 1:
            needed_workers = 1

        diff = needed_workers - active_workers

        if diff > 0:
            self.logger.info(
                f'Increasing the number of AiiDA daemons by {diff}.')
            client.increase_workers(diff)
        elif diff < 0:
            client.decrease_workers(abs(diff))
            self.logger.info(
                f'Decreasing the number of AiiDA daemons by {abs(diff)}.')

    def get_aiida_daemon_client(self) -> DaemonClient:
        """
        Retrieve the daemon client used within in AiiDA.
        """

        config = get_config()
        self.default_profile_name = config.default_profile_name

        client = get_daemon_client(self.default_profile_name)

        return client

    def checkpoint_workflow(self):
        """
        Checkpoint the workflow module into the checkpoint file.

        Save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities.
        """
        self.save_job_dict()

    def restart_workflow(self):
        """
        Restart the workflow module from the checkpoint file.

        Check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so.
        """
        # set the jobs dict from the pickle file
        self.read_job_dict()

    def update_job_status(self, pks: list[int]) -> list[str]:
        """
        Query the scheduler and extract the job_status.

        This helper function uses the AiiDA QueryBuilder to check updates
        about a job's progress, modifying the corresponding job_status
        object. Status options are: 'CREATED', 'EXCEPTED', 'FINISHED',
        'KILLED', 'RUNNING', and 'WAITING'. The current status is returned
        for convenience.

        :param pks: list of AiiDA PKs of the jobs to check for completion
        :returns: list of job states
        """

        # Load the AiiDA profile
        load_profile()

        # Create an instance of the QueryBuilder
        qb = QueryBuilder()

        # Query for the pks from the list
        qb.append(ProcessNode,
                  filters={'id': {
                      'in': pks
                  }},
                  project=['attributes.process_state', 'attributes.exit_code'])
        statuses = qb.all()  # Returns a list of tuples [(state, exit_code)]

        status_changed = False
        updated_states = []
        for pk, status in zip(pks, statuses):
            # Get the previous job state
            known_status = self.get_job_status(pk)
            if status[0] != known_status.state:
                self.logger.info((f'Updating job {pk} state from '
                                  f'{known_status.state} to {status[0]}'))
                known_status.state = status[0]
                status_changed = True
            updated_states.append(status)
        if status_changed:
            self.checkpoint_workflow()

        return updated_states

    def block_until_completed(self, pks: Union[int, list[int]]):
        """
        Function will periodically check on the job status in AiiDA. The time
        between checks is based on the wait_freq variable in the workflow. The
        default value is 60 seconds.

        :param pks: list of AiiDA PKs of the jobs to check for
            completion. Can also pass a single ID.
        """

        if type(pks) is list:
            remaining_jobs = deepcopy(pks)
        elif type(pks) is int:
            remaining_jobs = [pks]
        else:
            raise TypeError('AiiDA PKs must be a single int or a list!')

        wait_cycle_counter = 0
        while len(remaining_jobs) > 0:
            job_states = self.update_job_status(remaining_jobs)
            jobs_to_remove = []
            for pk, job_state in zip(remaining_jobs, job_states):
                if job_state[0] == 'finished':
                    self.logger.info((f'AiiDA Job <{pk}> completed, removing '
                                      f'from waiting list'))
                    jobs_to_remove.append(pk)

            # remove after iterating the whole list so no IDs are skipped
            for pk in jobs_to_remove:
                remaining_jobs.remove(pk)

            # Check the daemon workload
            self.check_daemon_workload()

            if len(remaining_jobs) == 0:
                break

            if wait_cycle_counter % 5 == 0:
                self.logger.info((f'Wait iteration {wait_cycle_counter}, ['
                                  f'cycle time = {self.synch_check_frequency}'
                                  f' s] with {len(remaining_jobs)} jobs left '
                                  f'to complete'))

            wait_cycle_counter += 1
            sleep(self.synch_check_frequency)

        self.logger.info(f'Jobs {pks} have completed, continuing...')

    def submit_job(self, builder: ProcessBuilder, job_details: dict) -> int:
        """
        Submits a job to AiiDA.

        submit_job handles job submission to AiiDA for the Oracle calculations.
        As most information should have already been defined in the `computer`
        and `code` items of AiiDA, this function will primarily set things such
        as details about the job's resources (``job_details``). Note that while
        default job resources (nodes, account, walltime, etc.) are present,
        they can be overridden by providing these keywords in the
        ``job_details`` dict for any specific calculation. Creates the
        :class:`~.workflow_base.JobStatus` for this job, where the job state is
        initially 'CREATED' and can be updated to 'EXCEPTED', 'FINISHED',
        'KILLED', 'RUNNING', or 'WAITING'. The 'FINISHED' state means the
        calculation has completed, but can be decorated with suffixes that add
        more information if the job didn't successfully complete (i.e.
        'FINISHED_TIMEOUT'). Status checks are performed by
        :meth:`~update_job_status`. Returns the AiiDA pk, which can be used to
        retrieve the present job's :class:`~.workflow_base.JobStatus`.

        :param builder: AiiDA builder object containing the required
            information to submit the calculation.
        :param job_details: specifics for running the job, such as
            number of nodes, queue, etc., as well as optional dependency list,
            if the job should be synchronous or asychronous, and any other
            optional arguments, such as pre- or postambles |default| ``None``
        :returns: return job ID to query this job status and location
        """

        self.logger.info('Spawning job, ID to be defined')

        extra_args = job_details.get('extra_args', {})

        # Load AiiDA profile
        load_profile()

        if job_details is None:
            job_details = {}
            self.logger.info((f'No job details specified, will use defaults:\n'
                              f'  N = {self.default_nodes},\n'
                              f'  A = {self.default_account},\n'
                              f'  t = {self.default_walltime},\n'
                              f'  p = {self.default_queue}\n'))

        calc = submit(builder)
        pk = calc.pk
        self.logger.info(f'Spawning AiiDA job with PK=<{pk}>')

        # Convert builder to string representation for storage
        command_str = str(builder) if builder else None
        metadata = extra_args.get(METADATA_KEY, {})
        job_status = JobStatus(
            f'{self.root_directory}/{pk}',
            'created',
            0,
            command=command_str,
            job_details=job_details,
            metadata=metadata,
        )
        self.jobs[pk] = job_status

        self.checkpoint_workflow()

        return pk

    @staticmethod
    def get_job_path(pk: int) -> Union[str, PathLike]:
        """
        Given the parent pk value that an Oracle returns, will get the absolute
        path on the remote server.

        :param pk: AiiDA PK
        :returns: Path on the remote server where the calculation occurred.
        """

        parent = load_node(pk)

        path = None
        for node in parent.called_descendants:
            if isinstance(node, CalcJobNode):
                path = node.get_remote_workdir()

        if not path:
            raise ValueError(
                f'The provided pk <{pk}> does not appear to have any '
                'CalcJobNode types associated with the descendants. '
                'Make sure the job finished completely.')

        return path
