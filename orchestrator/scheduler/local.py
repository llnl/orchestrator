import shutil
from os import system, PathLike
from typing import Optional, Union
from .scheduler_base import Scheduler, JobStatus
from ..utils.restart import restarter
from ..utils.data_standard import METADATA_KEY


class LocalScheduler(Scheduler):
    """
    Scheduler manager for execution in the local environment (i.e. login node)

    Responsibilities include directory creation, job creation, job status
    checking. Can run with mpi tasks if tasks are set in job_details, but no
    default parallel behavior is set.
    """

    def __init__(self, **kwargs):
        """
        set variables and initialize the recorder

        :param kwargs: parameters passed to parent for init. Keys include:
            root_directory, checkpoint_file, checkpoint_name, and
            job_record_file
        :type kwargs:
        """
        self.current_job_id = 0
        self.ID_TYPE = int  # LocalScheduler uses integer job IDs
        super().__init__(**kwargs)

    def checkpoint_scheduler(self):
        """
        checkpoint the scheduler module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        self.save_job_dict()
        save_dict = {
            self.checkpoint_name: {
                'current_job': self.current_job_id,
                'counters': self.counters,
            }
        }
        restarter.write_checkpoint_file(self.checkpoint_file, save_dict)

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
        self.current_job_id = restart_dict.get('current_job',
                                               self.current_job_id)
        self.counters = restart_dict.get('counters', self.counters)

    def block_until_completed(self, calc_ids: Union[list, int]):
        """
        Function for enforcing synchronous execution

        Implementation is just to pass since submit_job won't return until the
        job is completed.

        :param calc_ids: list of IDs of the jobs to check for
            completion. Can also pass a single ID.
        :type calc_ids: int or list
        """
        pass

    def submit_job(
        self,
        command: str,
        run_path: Union[str, PathLike],
        job_details: Optional[dict[str, Union[float, str]]] = None,
    ) -> int:
        """
        Submits a job for running.

        submit_job handles job submission for the modules and is the main
        interface for the schedulers to be used. For the
        :class:`LocalScheduler`
        implementation, this method uses ``os.system`` to execute the command
        on via command line interface. Inputs define the command to be executed
        for the job, location for the run, and details of the job.
        ``job_details`` inlcude ``dependencies`` of the job but no other keys.
        The ``dependencies`` are a list of job IDs which must have a
        successfully completed :class:`~JobStatus` for the present job to run.
        If one of the dependencies returns an error, this job will not run and
        the status will return an error. Creates the
        :class:`~.scheduler_base.JobStatus` for this job, where the job state
        is always 'done' since jobs are run instantly on the command line.
        Returns a job handle (ID) that can be used to query status and to
        retrieve the present job's :class:`~.scheduler_base.JobStatus`.

        :param command: command that defines the job to be executed
        :type command: implementation dependent
        :param run_path: location in the file system where inputs and outputs
            are to be accessed and stored
        :type run_path: str
        :param job_details: optional parameters for running the job, such as
            number of nodes, queue, etc. |default| ``None``
        :type job_details: dict
        :returns: return job ID to query this job status and location
        :rtype: int
        """
        if job_details is None:
            job_details = {}
        calc_id = self.current_job_id
        self.current_job_id += 1

        # check inputs and provide information to user
        synchronous = job_details.get('synchronous', True)
        dependencies = job_details.get('dependencies', [])
        tasks = job_details.get('tasks')
        extra_args = job_details.get('extra_args', {})

        if not synchronous:
            self.logger.info(
                ('LocalScheduler cannot be run asynchronously, running '
                 'in a blocking manner'))
        if extra_args:
            self.logger.info(
                'Note that extra_args are not read by LocalScheduler')

        job_can_run = True
        exit_code = 'undefined error'
        if dependencies:
            self.logger.info(f'Checking dependencies for: {calc_id}')
            for depend_id in dependencies:
                job_status = self.get_job_status(depend_id)
                if job_status.state != 'done' or job_status.exit_code != 0:
                    job_can_run = False
                    exit_code = 'dependency not satisfied'
                    self.logger.info((f'[{depend_id}]: state = '
                                      f'{job_status.state}, exit_code = '
                                      f'{job_status.exit_code}'))
                    break
        if command is None or command == '':
            job_can_run = False
            exit_code = 'empty command'

        if tasks is not None:
            mpi = None
            mpi_list = ['mpirun', 'mpiexec', 'srun']
            for mpi_exec in mpi_list:
                if shutil.which(mpi_exec):
                    mpi = mpi_exec
            if mpi:
                command = f'{mpi} -n {tasks} {command}'
            elif mpi is None:
                raise RuntimeError('`tasks` was set in job_details but could '
                                   f'not find {mpi_list} in the environment.')

        if job_can_run:
            self.logger.info(f'Spawning job with ID: {calc_id}')
            exit_code = system(
                f'(cd {run_path}; {command} 2>> local_scheduler_stdout.log)')
            self.logger.info(f'Job {calc_id} execution completed')
            state = 'done'
        else:
            state = 'done_cancelled'

        metadata = extra_args.get(METADATA_KEY, {})
        job_status = JobStatus(
            run_path,
            state,
            exit_code,
            command=command,
            job_details=job_details,
            metadata=metadata,
        )
        self.jobs[calc_id] = job_status
        self.save_job_dict()
        self.checkpoint_scheduler()
        return calc_id
