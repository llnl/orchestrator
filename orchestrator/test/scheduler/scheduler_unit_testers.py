import json
from pathlib import Path
from orchestrator.utils.setup_input import init_and_validate_module_type


def scheduler_local_submit_test(input_file: str) -> bool:
    """
    Test LocalScheduler job submission functionality.

    Tests basic job submission, execution, status tracking, and dependencies.

    :param input_file: Path to JSON configuration file
    :type input_file: str
    :returns: True if test completed successfully
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    scheduler = init_and_validate_module_type('scheduler', test_inputs)

    # Test 1: Submit a simple job
    test_dir = scheduler.make_path('test', 'simple_job')
    job_id_1 = scheduler.submit_job(
        command='echo "Hello World" > output.txt',
        run_path=test_dir,
    )

    # Verify job completed
    job_status_1 = scheduler.get_job_status(job_id_1)
    assert job_status_1 is not None, "Job status should not be None"
    expected_state = 'done'
    assert job_status_1.state == expected_state, (
        f"Job state should be 'done', got {job_status_1.state}")
    assert job_status_1.exit_code == 0, (
        f"Exit code should be 0, got {job_status_1.exit_code}")
    assert job_status_1.path == test_dir, "Job path should match"

    # Verify output file was created
    output_file = Path(test_dir) / 'output.txt'
    assert output_file.exists(), "Output file should exist"
    with open(output_file, 'r') as f:
        content = f.read().strip()
        assert content == "Hello World", (
            f"Expected 'Hello World', got '{content}'")

    # Test 2: Submit job with command that fails
    test_dir_2 = scheduler.make_path('test', 'failing_job')
    job_id_2 = scheduler.submit_job(
        command='exit 1',
        run_path=test_dir_2,
    )

    job_status_2 = scheduler.get_job_status(job_id_2)
    assert job_status_2.state == 'done', ("Failed job should still be 'done'")
    assert job_status_2.exit_code != 0, (
        "Exit code should be non-zero for failed job")

    # Test 3: Submit job with dependencies
    test_dir_3 = scheduler.make_path('test', 'dependency_job')
    scheduler.submit_job(
        command='echo "Dependent job" > dependent.txt',
        run_path=test_dir_3,
        job_details={'dependencies': [job_id_1]},
    )

    job_status_3 = scheduler.get_job_status(job_id_1)
    assert job_status_3.state == 'done', "Dependent job should complete"
    assert job_status_3.exit_code == 0, "Dependent job should succeed"

    # Test 4: Submit job with failed dependency
    test_dir_4 = scheduler.make_path('test', 'failed_dependency_job')
    job_id_4 = scheduler.submit_job(
        command='echo "Should not run" > should_not_exist.txt',
        run_path=test_dir_4,
        job_details={'dependencies': [job_id_2]},
    )

    job_status_4 = scheduler.get_job_status(job_id_4)
    expected_cancelled = 'done_cancelled'
    assert job_status_4.state == expected_cancelled, (
        "Job with failed dependency should be cancelled")
    no_run_file = Path(test_dir_4) / 'should_not_exist.txt'
    assert not no_run_file.exists(), (
        "Job with failed dependency should not create output")

    print('LocalScheduler submit test completed successfully!')
    return True


def scheduler_path_management_test(input_file: str) -> bool:
    """
    Test path management functionality (make_path, make_path_base).

    Tests directory creation, counter incrementation, and path hierarchy.

    :param input_file: Path to JSON configuration file
    :type input_file: str
    :returns: True if test completed successfully
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    scheduler = init_and_validate_module_type('scheduler', test_inputs)

    # Test 1: make_path_base creates directory without counter
    base_path = scheduler.make_path_base('test_module', 'test_type')
    expected_base = (f"{scheduler.root_directory}/test_module/test_type")
    assert base_path == expected_base, (
        f"Expected {expected_base}, got {base_path}")
    assert Path(base_path).exists(), "Base path directory should exist"

    # Test 2: make_path creates directories with counters
    path_1 = scheduler.make_path('module1', 'type1')
    expected_1 = f"{scheduler.root_directory}/module1/type1/00000"
    assert path_1 == expected_1, f"Expected {expected_1}, got {path_1}"
    assert Path(path_1).exists(), "First path should exist"

    path_2 = scheduler.make_path('module1', 'type1')
    expected_2 = f"{scheduler.root_directory}/module1/type1/00001"
    assert path_2 == expected_2, f"Expected {expected_2}, got {path_2}"
    assert Path(path_2).exists(), "Second path should exist"

    # Test 3: Different module/type combinations have separate counters
    path_3 = scheduler.make_path('module2', 'type1')
    expected_3 = f"{scheduler.root_directory}/module2/type1/00000"
    assert path_3 == expected_3, ("Different module should start at counter 0")

    path_4 = scheduler.make_path('module1', 'type2')
    expected_4 = f"{scheduler.root_directory}/module1/type2/00000"
    assert path_4 == expected_4, ("Different type should start at counter 0")

    # Test 4: Counter continues incrementing
    path_5 = scheduler.make_path('module1', 'type1')
    expected_5 = f"{scheduler.root_directory}/module1/type1/00002"
    assert path_5 == expected_5, (
        f"Counter should continue, expected {expected_5}, got {path_5}")

    # Test 5: Verify counters are tracked internally
    counter_key_1 = 'module1.type1'
    assert counter_key_1 in scheduler.counters, ("Counter key should exist")
    assert scheduler.counters[counter_key_1] == 3, (
        "Counter should be 3 after 3 calls")

    print('Path management test completed successfully!')
    return True


def scheduler_checkpoint_restart_test(input_file: str) -> bool:
    """
    Test checkpoint and restart functionality.

    Tests state serialization, job dict persistence, and state restoration.

    :param input_file: Path to JSON configuration file
    :type input_file: str
    :returns: True if test completed successfully
    :rtype: bool
    """
    from orchestrator.utils.data_standard import METADATA_KEY
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    # Create first scheduler instance and submit jobs
    scheduler1 = init_and_validate_module_type('scheduler', test_inputs)

    # Submit multiple jobs to build up state
    test_dir_1 = scheduler1.make_path('checkpoint_test', 'job_batch')
    scheduler1.submit_job(
        command='echo "Job 1" > job1.txt',
        run_path=test_dir_1,
    )

    test_dir_2 = scheduler1.make_path('checkpoint_test', 'job_batch')
    job_id_2 = scheduler1.submit_job(
        command='echo "Job 2" > job2.txt',
        run_path=test_dir_2,
        job_details={'extra_args': {
            METADATA_KEY: {
                'test_key': 'test_value'
            }
        }},
    )

    test_dir_3 = scheduler1.make_path('checkpoint_test', 'other_type')
    scheduler1.submit_job(
        command='echo "Job 3" > job3.txt',
        run_path=test_dir_3,
    )

    # Capture state before restart
    original_counters = scheduler1.counters.copy()
    original_jobs = scheduler1.get_all_statuses()
    checkpoint_file = scheduler1.checkpoint_file
    job_record_file = scheduler1.job_record_file

    # Verify checkpoint files were created
    assert Path(checkpoint_file).exists(), ("Checkpoint file should exist")
    assert Path(job_record_file).exists(), ("Job record file should exist")

    # Create new scheduler instance (simulating restart)
    scheduler2 = init_and_validate_module_type('scheduler', test_inputs)

    # Verify counters were restored
    assert scheduler2.counters == original_counters, (
        "Counters should be restored")

    # Verify jobs were restored
    restored_jobs = scheduler2.get_all_statuses()
    assert len(restored_jobs) == len(original_jobs), (
        "All jobs should be restored")

    for job_id in original_jobs:
        assert job_id in restored_jobs, (f"Job {job_id} should be restored")
        orig_status = original_jobs[job_id]
        rest_status = restored_jobs[job_id]
        assert orig_status.path == rest_status.path, ("Job path should match")
        assert orig_status.state == rest_status.state, (
            "Job state should match")
        assert orig_status.exit_code == rest_status.exit_code, (
            "Exit code should match")

    # Verify metadata was preserved
    job_2_status = scheduler2.get_job_status(job_id_2)
    assert 'test_key' in job_2_status.metadata, (
        "Metadata should be preserved")
    assert job_2_status.metadata['test_key'] == 'test_value', (
        "Metadata value should match")

    # Test that new scheduler can continue from restored state
    test_dir_4 = scheduler2.make_path('checkpoint_test', 'job_batch')
    expected_counter = original_counters['checkpoint_test.job_batch']
    expected_path = (f"{scheduler2.root_directory}/checkpoint_test/job_batch/"
                     f"{expected_counter:05d}")
    assert test_dir_4 == expected_path, (
        "New paths should continue from restored counter")

    print('Checkpoint/restart test completed successfully!')
    return True


def scheduler_job_status_test(input_file: str) -> bool:
    """
    Test job status tracking functionality.

    Tests get_job_status, get_job_path, get_attached_metadata,
    get_all_statuses.

    :param input_file: Path to JSON configuration file
    :type input_file: str
    :returns: True if test completed successfully
    :rtype: bool
    """
    from orchestrator.utils.data_standard import METADATA_KEY
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    scheduler = init_and_validate_module_type('scheduler', test_inputs)

    # Submit jobs with different metadata
    metadata_1 = {'key1': 'value1', 'description': 'Test job 1'}
    test_dir_1 = scheduler.make_path('status_test', 'tracked_jobs')
    job_id_1 = scheduler.submit_job(
        command='echo "Status test 1" > status1.txt',
        run_path=test_dir_1,
        job_details={'extra_args': {
            METADATA_KEY: metadata_1,
        }},
    )

    metadata_2 = {'key2': 'value2', 'description': 'Test job 2'}
    test_dir_2 = scheduler.make_path('status_test', 'tracked_jobs')
    job_id_2 = scheduler.submit_job(
        command='echo "Status test 2" > status2.txt',
        run_path=test_dir_2,
        job_details={'extra_args': {
            METADATA_KEY: metadata_2,
        }},
    )

    # Test get_job_status
    status_1 = scheduler.get_job_status(job_id_1)
    assert status_1 is not None, "Should retrieve job status"
    assert status_1.path == test_dir_1, "Path should match"
    assert status_1.state == 'done', "State should be 'done'"
    assert status_1.exit_code == 0, "Exit code should be 0"
    expected_cmd = 'echo "Status test 1" > status1.txt'
    assert status_1.command == expected_cmd, "Command should match"

    # Test get_job_path
    retrieved_path_1 = scheduler.get_job_path(job_id_1)
    assert retrieved_path_1 == test_dir_1, (
        "get_job_path should return correct path")

    retrieved_path_2 = scheduler.get_job_path(job_id_2)
    assert retrieved_path_2 == test_dir_2, (
        "get_job_path should return correct path")

    # Test get_attached_metadata
    retrieved_metadata_1 = scheduler.get_attached_metadata(job_id_1)
    assert retrieved_metadata_1 == metadata_1, "Metadata should match"

    retrieved_metadata_2 = scheduler.get_attached_metadata(job_id_2)
    assert retrieved_metadata_2 == metadata_2, "Metadata should match"

    # Test get_all_statuses
    all_statuses = scheduler.get_all_statuses()
    assert len(all_statuses) >= 2, "Should have at least 2 jobs"
    assert job_id_1 in all_statuses, ("Job 1 should be in all statuses")
    assert job_id_2 in all_statuses, ("Job 2 should be in all statuses")

    # Test querying non-existent job
    fake_job_id = 99999
    fake_status = scheduler.get_job_status(fake_job_id)
    assert fake_status is None, ("Non-existent job should return None")

    fake_path = scheduler.get_job_path(fake_job_id)
    assert fake_path is None, ("Non-existent job path should return None")

    fake_metadata = scheduler.get_attached_metadata(fake_job_id)
    assert fake_metadata == {}, (
        "Non-existent job metadata should return empty dict")

    print('Job status test completed successfully!')
    return True


def scheduler_restart_job_test(input_file: str) -> bool:
    """
    Test job restart functionality.

    Tests file copying, path replacement, job detail merging, and
    resubmission.

    :param input_file: Path to JSON configuration file
    :type input_file: str
    :returns: True if test completed successfully
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    scheduler = init_and_validate_module_type('scheduler', test_inputs)

    # Create and submit initial job with input files
    test_dir = scheduler.make_path('restart_test', 'initial_run')

    # Create input files
    input_file_path = Path(test_dir) / 'input.txt'
    with open(input_file_path, 'w') as f:
        f.write(f'Input data from {test_dir}\n')

    config_file_path = Path(test_dir) / 'config.json'
    with open(config_file_path, 'w') as f:
        json.dump({'path': test_dir, 'setting': 'value1'}, f)

    # Submit initial job
    job_id_1 = scheduler.submit_job(
        command='cat input.txt > output.txt',
        run_path=test_dir,
        job_details={
            'nodes': 1,
            'synchronous': False,
        },
    )

    # Verify initial job completed
    status_1 = scheduler.get_job_status(job_id_1)
    assert status_1.state == 'done', "Initial job should complete"
    assert (Path(test_dir) / 'output.txt').exists(), ("Output should exist")

    # Restart the job with modified job_details
    new_job_id = scheduler.restart_job(
        calc_id=job_id_1,
        job_details={'synchronous': True},  # Changed from asynch to synch
    )

    # Verify new job was created
    assert new_job_id != job_id_1, "New job ID should be different"

    new_status = scheduler.get_job_status(new_job_id)
    assert new_status is not None, "New job status should exist"

    new_path = new_status.path
    assert new_path != test_dir, "New job path should be different"
    assert Path(new_path).exists(), "New job directory should exist"

    # Verify files were copied
    new_input_file = Path(new_path) / 'input.txt'
    assert new_input_file.exists(), "Input file should be copied"

    new_config_file = Path(new_path) / 'config.json'
    assert new_config_file.exists(), "Config file should be copied"

    # Verify path was updated in config.json
    with open(new_config_file, 'r') as f:
        new_config = json.load(f)
        assert new_config['path'] == new_path, (
            "Path in config should be updated")
        assert new_config['setting'] == 'value1', (
            "Other settings should be preserved")

    # Verify input file content has updated path reference
    with open(new_input_file, 'r') as f:
        content = f.read()
        assert new_path in content, ("Path in input file should be updated")
        assert test_dir not in content, "Old path should be replaced"

    # Verify job details were merged
    assert new_status.job_details.get('synchronous') is True, (
        "Synchronous should be updated to True")
    assert new_status.job_details.get('nodes') == 1, (
        "Nodes should be preserved from original")

    # Verify job ran successfully
    assert new_status.state == 'done', "Restarted job should complete"
    new_output = Path(new_path) / 'output.txt'
    assert new_output.exists(), "Restarted job should create output"

    print('Restart job test completed successfully!')
    return True
