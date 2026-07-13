import json
import shutil
from pathlib import Path
from time import sleep
from orchestrator.utils.setup_input import init_and_validate_module_type


def check_slurm_available() -> bool:
    """Check if Slurm scheduler is available."""
    return (shutil.which('sbatch') is not None
            and shutil.which('squeue') is not None)


def check_flux_available() -> bool:
    """Check if Flux scheduler is available."""
    return shutil.which('flux') is not None


def check_lsf_available() -> bool:
    """Check if LSF scheduler is available."""
    return (shutil.which('bsub') is not None
            and shutil.which('bquery') is not None)


def _run_scheduler_integration_tests(
    scheduler,
    scheduler_name: str,
    validate_job_id_fn=None,
) -> None:
    """
    Shared integration test logic for HPC schedulers.

    :param scheduler: Initialized scheduler instance
    :param scheduler_name: Name of scheduler (lowercase, for paths/messages)
    :param validate_job_id_fn: Optional function to validate job ID format
    """
    # Test 1: Submit a simple job
    test_dir_1 = scheduler.make_path(f'real_{scheduler_name}', 'simple_job')
    output_filename = f'{scheduler_name}_output.txt'
    job_id_1 = scheduler.submit_job(
        command=(f'echo "{scheduler_name.capitalize()} integration test" > '
                 f'{output_filename} && sleep 2'),
        run_path=test_dir_1,
        job_details={'synchronous': True},
    )

    # Verify job completed
    job_status_1 = scheduler.get_job_status(job_id_1)
    assert job_status_1 is not None, "Job status should exist"
    assert job_status_1.state == 'done', (
        f"Job should be done, got {job_status_1.state}")

    output_file = Path(test_dir_1) / output_filename
    assert output_file.exists(), "Output file should exist"

    # Run scheduler-specific job ID validation if provided
    if validate_job_id_fn is not None:
        validate_job_id_fn(job_id_1)

    # Test 2: Submit job with dependency
    test_dir_2 = scheduler.make_path(f'real_{scheduler_name}', 'dependent_job')
    job_id_2 = scheduler.submit_job(
        command='echo "Dependent job" > dependent_output.txt',
        run_path=test_dir_2,
        job_details={
            'dependencies': [job_id_1],
            'synchronous': True
        },
    )

    job_status_2 = scheduler.get_job_status(job_id_2)
    assert job_status_2.state == 'done', "Dependent job should complete"

    # Test 3: Test async submission and status checking
    test_dir_3 = scheduler.make_path(f'real_{scheduler_name}', 'async_job')
    job_id_3 = scheduler.submit_job(
        command='echo "Async job" > async_output.txt && sleep 5',
        run_path=test_dir_3,
        job_details={'synchronous': False},
    )

    # Manually wait and check status
    max_wait_time = 300  # 5 minutes max
    waited = 0
    while waited < max_wait_time:
        status_list = scheduler.update_job_status([job_id_3])
        if status_list[0][:4] == 'done':
            break
        sleep(10)
        waited += 10

    final_status = scheduler.get_job_status(job_id_3)
    assert final_status.state == 'done', (
        "Async job should eventually complete")

    # Test 4: Test restart_job functionality
    test_dir_4 = scheduler.make_path(f'real_{scheduler_name}', 'restart_test')

    # Create input file
    input_path = Path(test_dir_4) / 'input.dat'
    with open(input_path, 'w') as f:
        f.write(f'Original path: {test_dir_4}\n')

    # Submit original job
    job_id_4 = scheduler.submit_job(
        command='cat input.dat > restart_output.txt',
        run_path=test_dir_4,
        job_details={
            'walltime': 5,
            'synchronous': True
        },
    )

    assert scheduler.get_job_status(job_id_4).state == 'done', (
        "Original job should complete")

    # Restart with longer walltime
    restarted_job_id = scheduler.restart_job(
        calc_id=job_id_4,
        job_details={'walltime': 6},
    )

    # Wait for restarted job
    scheduler.block_until_completed(restarted_job_id)

    restarted_status = scheduler.get_job_status(restarted_job_id)
    assert restarted_status.state == 'done', ("Restarted job should complete")
    assert restarted_status.job_details['walltime'] == 6, (
        "Walltime should be updated")

    # Verify files were copied and paths updated
    new_path = restarted_status.path
    new_input = Path(new_path) / 'input.dat'
    assert new_input.exists(), "Input file should be copied"

    with open(new_input, 'r') as f:
        content = f.read()
        assert new_path in content, ("Path should be updated in copied file")


def _validate_flux_job_id(job_id: str) -> None:
    """Validate Flux job ID format (12 chars starting with 'f')."""
    assert isinstance(job_id, str), "Flux job ID should be string"
    assert len(job_id) == 12, (
        f"Flux job ID should be 12 chars, got {len(job_id)}")
    assert job_id[0] == 'f', (
        f"Flux job ID should start with 'f', got {job_id[0]}")


def scheduler_real_slurm_submit_test(input_file: str) -> bool:
    """
    Integration test for SlurmScheduler with real Slurm scheduler.

    Submits actual jobs to Slurm and monitors their lifecycle.
    Only runs if Slurm is available.

    :param input_file: Path to JSON configuration file
    :type input_file: str
    :returns: True if test completed successfully
    :rtype: bool
    """
    if not check_slurm_available():
        print("Slurm not available, skipping real Slurm test")
        return False

    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    scheduler = init_and_validate_module_type('scheduler', test_inputs)
    _run_scheduler_integration_tests(scheduler, 'slurm')

    print('Real Slurm integration test completed successfully!')
    return True


def scheduler_real_flux_submit_test(input_file: str) -> bool:
    """
    Integration test for FluxScheduler with real Flux scheduler.

    Submits actual jobs to Flux and monitors their lifecycle.
    Only runs if Flux is available.

    :param input_file: Path to JSON configuration file
    :type input_file: str
    :returns: True if test completed successfully
    :rtype: bool
    """
    if not check_flux_available():
        print("Flux not available, skipping real Flux test")
        return False

    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    scheduler = init_and_validate_module_type('scheduler', test_inputs)
    _run_scheduler_integration_tests(
        scheduler,
        'flux',
        validate_job_id_fn=_validate_flux_job_id,
    )

    print('Real Flux integration test completed successfully!')
    return True


def scheduler_real_lsf_submit_test(input_file: str) -> bool:
    """
    Integration test for LSFScheduler with real LSF scheduler.

    Submits actual jobs to LSF and monitors their lifecycle.
    Only runs if LSF is available.

    :param input_file: Path to JSON configuration file
    :type input_file: str
    :returns: True if test completed successfully
    :rtype: bool
    """
    if not check_lsf_available():
        print("LSF not available, skipping real LSF test")
        return False

    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    scheduler = init_and_validate_module_type('scheduler', test_inputs)
    _run_scheduler_integration_tests(scheduler, 'lsf')

    print('Real LSF integration test completed successfully!')
    return True
