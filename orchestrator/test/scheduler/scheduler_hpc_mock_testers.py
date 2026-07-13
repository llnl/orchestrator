import json
from pathlib import Path
from unittest import mock
from orchestrator.utils.setup_input import init_and_validate_module_type


def scheduler_mock_slurm_submit_test(input_file: str) -> bool:
    """
    Test SlurmScheduler job submission with mocked subprocess calls.

    Tests job ID extraction, batch file generation, and state parsing
    without requiring actual Slurm scheduler.

    :param input_file: Path to JSON configuration file
    :type input_file: str
    :returns: True if test completed successfully
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    scheduler = init_and_validate_module_type('scheduler', test_inputs)

    # Mock subprocess.run for sbatch submission
    with mock.patch('subprocess.run') as mock_run:
        # Mock successful job submission
        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Submitted batch job 12345"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Submit job
        test_dir = scheduler.make_path('mock_slurm', 'test_job')
        job_id = scheduler.submit_job(
            command='echo "Slurm test" > output.txt',
            run_path=test_dir,
        )

        # Verify job ID was extracted correctly
        assert job_id == 12345, f"Expected job ID 12345, got {job_id}"

        # Verify batch file was created
        batch_files = list(Path(test_dir).glob('*.sh'))
        assert len(batch_files) > 0, "Batch file should be created"

        # Verify batch file contains expected content
        with open(batch_files[0], 'r') as f:
            content = f.read()
            assert 'srun' in content, (
                "Batch file should contain srun command")
            assert 'echo "Slurm test" > output.txt' in content, (
                "Command should be in batch file")

    # Test job ID parsing from different formats
    scheduler_obj = scheduler

    # Test normal output
    job_id_1 = scheduler_obj._parse_job_id("Submitted batch job 54321")
    assert job_id_1 == 54321, "Should parse normal sbatch output"

    # Test srun output
    job_id_2 = scheduler_obj._parse_job_id(
        "srun: job 98765 queued and waiting")
    assert job_id_2 == 98765, "Should parse srun output"

    # Test state parsing
    split_output = ['12345', 'R', '54321', 'PD', '98765', 'CG']

    state_1 = scheduler_obj._parse_job_state(12345, split_output)
    assert state_1 == 'running', f"Expected 'running', got {state_1}"

    state_2 = scheduler_obj._parse_job_state(54321, split_output)
    assert state_2 == 'pending', f"Expected 'pending', got {state_2}"

    state_3 = scheduler_obj._parse_job_state(98765, split_output)
    assert state_3 == 'completing', f"Expected 'completing', got {state_3}"

    print('Mock Slurm test completed successfully!')
    return True


def scheduler_mock_flux_submit_test(input_file: str) -> bool:
    """
    Test FluxScheduler job submission with mocked subprocess calls.

    Tests Flux job ID format validation, batch file generation, and state
    parsing without requiring actual Flux scheduler.

    :param input_file: Path to JSON configuration file
    :type input_file: str
    :returns: True if test completed successfully
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    scheduler = init_and_validate_module_type('scheduler', test_inputs)

    # Mock subprocess.run for flux batch submission
    with mock.patch('subprocess.run') as mock_run:
        # Mock successful job submission
        # Flux ID format: 12 chars starting with 'f'
        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = "f1234567890a"  # Valid Flux job ID
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Submit job
        test_dir = scheduler.make_path('mock_flux', 'test_job')
        job_id = scheduler.submit_job(
            command='echo "Flux test" > output.txt',
            run_path=test_dir,
        )

        # Verify job ID was extracted correctly
        assert job_id == "f1234567890a", (
            f"Expected job ID 'f1234567890a', got {job_id}")

        # Verify batch file was created
        batch_files = list(Path(test_dir).glob('*.sh'))
        assert len(batch_files) > 0, "Batch file should be created"

        # Verify batch file contains expected content
        with open(batch_files[0], 'r') as f:
            content = f.read()
            assert 'flux run' in content, (
                "Batch file should contain flux run command")
            assert 'echo "Flux test" > output.txt' in content, (
                "Command should be in batch file")

    # Test job ID parsing
    scheduler_obj = scheduler

    # Valid Flux ID
    job_id_1 = scheduler_obj._parse_job_id("f1a2b3c4d5e6")
    assert job_id_1 == "f1a2b3c4d5e6", ("Should parse valid Flux job ID")

    # Test invalid formats should raise ValueError
    try:
        scheduler_obj._parse_job_id("invalid_id")
        assert False, "Should raise ValueError for invalid ID"
    except ValueError:
        pass  # Expected

    try:
        scheduler_obj._parse_job_id("f123")  # Too short
        assert False, "Should raise ValueError for short ID"
    except ValueError:
        pass  # Expected

    # Test state parsing
    split_output = [
        'f1234567890a', 'RUN', 'f1234567890b', 'SCHED', 'f1234567890c',
        'COMPLETED'
    ]

    state_1 = scheduler_obj._parse_job_state('f1234567890a', split_output)
    assert state_1 == 'running', f"Expected 'running', got {state_1}"

    state_2 = scheduler_obj._parse_job_state('f1234567890b', split_output)
    assert state_2 == 'pending', f"Expected 'pending', got {state_2}"

    state_3 = scheduler_obj._parse_job_state('f1234567890c', split_output)
    assert state_3 == 'done', f"Expected 'done', got {state_3}"

    # Test dependency string construction
    dependencies = ['f1234567890a', 'f1234567890b']
    depend_str = scheduler_obj._build_dependency_string(dependencies, {})
    assert '--dependency=afterok:f1234567890a' in depend_str, (
        "Should contain first dependency")
    assert '--dependency=afterok:f1234567890b' in depend_str, (
        "Should contain second dependency")

    print('Mock Flux test completed successfully!')
    return True


def scheduler_mock_lsf_submit_test(input_file: str) -> bool:
    """
    Test LSFScheduler job submission with mocked subprocess calls.

    Tests LSF job ID extraction, batch file generation, and state parsing
    without requiring actual LSF scheduler.

    :param input_file: Path to JSON configuration file
    :type input_file: str
    :returns: True if test completed successfully
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    scheduler = init_and_validate_module_type('scheduler', test_inputs)

    # Mock subprocess.run for bsub submission
    with mock.patch('subprocess.run') as mock_run:
        # Mock successful job submission
        # LSF format: "Job <12345> is submitted..."
        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "Job <67890> is submitted to default queue <normal>.")
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Submit job
        test_dir = scheduler.make_path('mock_lsf', 'test_job')
        job_id = scheduler.submit_job(
            command='echo "LSF test" > output.txt',
            run_path=test_dir,
        )

        # Verify job ID was extracted correctly
        assert job_id == 67890, f"Expected job ID 67890, got {job_id}"

        # Verify batch file was created
        batch_files = list(Path(test_dir).glob('*.sh'))
        assert len(batch_files) > 0, "Batch file should be created"

        # Verify batch file contains expected content
        with open(batch_files[0], 'r') as f:
            content = f.read()
            assert 'lrun' in content or 'jsrun' in content, (
                "Batch file should contain lrun/jsrun command")
            assert 'echo "LSF test" > output.txt' in content, (
                "Command should be in batch file")

    # Test job ID parsing
    scheduler_obj = scheduler

    # Valid LSF output
    job_id_1 = scheduler_obj._parse_job_id(
        "Job <12345> is submitted to queue.")
    assert job_id_1 == 12345, "Should parse valid LSF job ID"

    # Test invalid format should raise ValueError
    try:
        scheduler_obj._parse_job_id("Invalid output")
        assert False, "Should raise ValueError for invalid format"
    except ValueError:
        pass  # Expected

    # Test state parsing
    # bquery output format: "id stat pendstate dependency exit_reason"
    split_output = [
        '12345', 'RUN', '-', '-', '-', '54321', 'PEND', 'some_reason', '-', '-'
    ]

    state_1 = scheduler_obj._parse_job_state(12345, split_output)
    assert state_1 == 'running', f"Expected 'running', got {state_1}"

    state_2 = scheduler_obj._parse_job_state(54321, split_output)
    assert state_2 == 'pending', f"Expected 'pending', got {state_2}"

    print('Mock LSF test completed successfully!')
    return True


def scheduler_mock_slurm_error_handling_test(input_file: str) -> bool:
    """
    Test error handling with mocked failures.

    Tests failed submissions, problematic job states, and exception
    handling.

    :param input_file: Path to JSON configuration file
    :type input_file: str
    :returns: True if test completed successfully
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    scheduler = init_and_validate_module_type('scheduler', test_inputs)

    # Test 1: Mock failed job submission (non-zero returncode)
    with mock.patch('subprocess.run') as mock_run:
        mock_result = mock.Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "sbatch: error: Unable to allocate resources"
        mock_run.return_value = mock_result

        from orchestrator.utils.exceptions import JobSubmissionError

        try:
            test_dir = scheduler.make_path('mock_error', 'failed_submit')
            job_id = scheduler.submit_job(
                command='echo "This will fail"',
                run_path=test_dir,
            )
            # Job ID should be the fallback unknown_job_id
            assert job_id >= 1000, ("Failed submission should use fallback ID")
        except JobSubmissionError:
            # Expected behavior - submission failed
            pass

    # Test 2: Test problematic job states
    from orchestrator.utils.exceptions import ProblematicJobStateError

    with mock.patch('subprocess.run') as mock_run:
        # First mock submission success
        mock_submit = mock.Mock()
        mock_submit.returncode = 0
        mock_submit.stdout = "Submitted batch job 99999"
        mock_submit.stderr = ""

        # Then mock status query showing TIMEOUT state
        # first, job is not present in query
        mock_status = mock.Mock()
        mock_status.returncode = 0
        mock_status.stdout = ""
        mock_status.stderr = ""

        # then, job is present in scontrol
        mock_parse = mock.Mock()
        mock_parse.returncode = 0
        mock_parse.stdout = ("col0 col1 col2 col3 col4 col5 col6 col7 col8 "
                             "col9 val=TIMEOUT")
        mock_parse.stderr = ""

        mock_run.side_effect = [mock_submit, mock_status, mock_parse]

        test_dir = scheduler.make_path('mock_error', 'timeout_job')
        job_id = scheduler.submit_job(
            command='echo "Timeout test"',
            run_path=test_dir,
        )

        try:
            # Try to update job status
            # Should raise ProblematicJobStateError
            scheduler.update_job_status([job_id])
            assert False, (
                "Should raise ProblematicJobStateError for TIMEOUT state")
        except ProblematicJobStateError as e:
            assert str(job_id) in str(e), ("Error should mention the job ID")
            assert 'done_timeout' in str(e).lower() or (
                'timeout' in str(e).lower()), ("Error should mention timeout")

    print('Mock error handling test completed successfully!')
    return True


def scheduler_mock_batch_file_generation_test(input_file: str) -> bool:
    """
    Test batch file generation for all schedulers.

    Tests walltime formatting, preamble construction, and template
    substitution.

    :param input_file: Path to JSON configuration file
    :type input_file: str
    :returns: True if test completed successfully
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    # Test for each scheduler type
    for scheduler_config in ['slurm_config', 'flux_config', 'lsf_config']:
        if scheduler_config not in test_inputs:
            continue

        config = test_inputs[scheduler_config]
        scheduler = init_and_validate_module_type('scheduler', config)

        # Mock subprocess to avoid actual submissions
        with mock.patch('subprocess.run') as mock_run:
            mock_result = mock.Mock()
            mock_result.returncode = 0

            # Set appropriate stdout based on scheduler type
            if 'Slurm' in scheduler.__class__.__name__:
                mock_result.stdout = "Submitted batch job 11111"
            elif 'Flux' in scheduler.__class__.__name__:
                mock_result.stdout = "f123456789ab"
            elif 'LSF' in scheduler.__class__.__name__:
                mock_result.stdout = "Job <22222> is submitted to queue."

            mock_result.stderr = ""
            mock_run.return_value = mock_result

            # Test with various job details
            test_dir = scheduler.make_path('batch_test',
                                           scheduler.__class__.__name__)

            job_details = {
                'nodes': 2,
                'tasks': 4,
                'walltime': 120,  # 2 hours in minutes
                'queue': 'debug',
                'account': 'test_account',
            }

            scheduler.submit_job(
                command='echo "Batch file test" > test.txt',
                run_path=test_dir,
                job_details=job_details,
            )

            # Find and read the batch file
            batch_files = list(Path(test_dir).glob('*.sh'))
            assert len(batch_files) > 0, (
                f"Batch file should be created for {scheduler_config}")

            with open(batch_files[0], 'r') as f:
                content = f.read()

                # Check common elements
                assert job_details['queue'] in content, (
                    "Queue should be in batch file")
                assert job_details['account'] in content, (
                    "Account should be in batch file")
                assert 'echo "Batch file test" > test.txt' in content, (
                    "Command should be in batch file")

                # Check scheduler-specific elements
                if 'Slurm' in scheduler.__class__.__name__:
                    assert '#SBATCH' in content, (
                        "Should have SBATCH directives")
                    assert '02:00:00' in content, (
                        "Should have formatted walltime with seconds")
                    assert 'srun' in content, ("Should have srun command")

                elif 'Flux' in scheduler.__class__.__name__:
                    assert 'flux run' in content, (
                        "Should have flux run command")
                    # Flux uses minutes
                    assert '120m' in content or '02:00' in content, (
                        "Should have walltime")

                elif 'LSF' in scheduler.__class__.__name__:
                    assert '#BSUB' in content, ("Should have BSUB directives")
                    assert 'lrun' in content or 'jsrun' in content, (
                        "Should have lrun/jsrun command")

        # Test walltime formatting
        if hasattr(scheduler, '_format_walltime'):
            # Test different walltime values
            if 'Slurm' in scheduler.__class__.__name__:
                # Slurm uses HH:MM:SS format
                formatted = scheduler._format_walltime(90,
                                                       include_seconds=True)
                assert formatted == '01:30:00', (
                    f"Expected '01:30:00', got {formatted}")

                formatted2 = scheduler._format_walltime(150.5,
                                                        include_seconds=True)
                assert formatted2 == '02:30:30', (
                    f"Expected '02:30:30', got {formatted2}")

            elif 'Flux' in scheduler.__class__.__name__:
                # Flux uses minutes
                formatted = scheduler._format_walltime(90,
                                                       include_seconds=False)
                assert formatted == '90m', (f"Expected '90m', got {formatted}")

            elif 'LSF' in scheduler.__class__.__name__:
                # LSF uses HH:MM format
                formatted = scheduler._format_walltime(90,
                                                       include_seconds=False)
                assert formatted == '01:30', (
                    f"Expected '01:30', got {formatted}")

    print('Mock batch file generation test completed successfully!')
    return True
