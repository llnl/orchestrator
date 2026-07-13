#!/usr/bin/env python3
"""Driver script for scheduler module tests."""

from orchestrator.test.scheduler.scheduler_unit_testers import (
    scheduler_local_submit_test,
    scheduler_path_management_test,
    scheduler_checkpoint_restart_test,
    scheduler_job_status_test,
    scheduler_restart_job_test,
)
from orchestrator.test.scheduler.scheduler_hpc_mock_testers import (
    scheduler_mock_slurm_submit_test,
    scheduler_mock_flux_submit_test,
    scheduler_mock_lsf_submit_test,
    scheduler_mock_slurm_error_handling_test,
    scheduler_mock_batch_file_generation_test,
)
from orchestrator.test.scheduler.scheduler_hpc_real_testers import (
    scheduler_real_slurm_submit_test,
    scheduler_real_flux_submit_test,
    scheduler_real_lsf_submit_test,
    check_slurm_available,
    check_flux_available,
    check_lsf_available,
)
import pytest
import shutil
import os
import sys


def clean_test_dirs():
    """Clean up test artifacts."""
    delete_paths = ["./scheduler/", "./orchestrator_scheduler/"]
    delete_files = [
        "orchestrator_checkpoint.json",
        "LocalScheduler_job_record.pkl",
        "SlurmScheduler_job_record.pkl",
        "FluxScheduler_job_record.pkl",
        "LSFScheduler_job_record.pkl",
    ]
    for file in delete_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass
    for loc in delete_paths:
        try:
            shutil.rmtree(loc)
        except FileNotFoundError:
            pass


def run_test(test_func, input_file, test_name):
    """Run a test function and handle exceptions gracefully."""
    try:
        result = test_func(input_file)
        if result:
            print(f"{test_name} passed")
        else:
            print(f"{test_name} failed")
        return result
    except Exception as e:
        print(f"{test_name} raised exception: {e}")
        return False


test_dir = os.path.dirname(os.path.abspath(__file__))
original_dir = os.getcwd()
print(f"Test directory: {test_dir}")
print(f"Working directory: {os.getcwd()}")

print("\n" + "=" * 60)
print("Cleaning up previous test artifacts")
print("=" * 60)
clean_test_dirs()

# Run local scheduler tests (all reuse same config)
print("\n" + "=" * 60)
print("Running Local Scheduler Tests")
print("=" * 60)
local_config = os.path.join(test_dir, 'test_inputs/01_local.json')
local_tests_ran = [False] * 5
local_tests_ran[0] = run_test(scheduler_local_submit_test, local_config,
                              "Local scheduler submit test")
local_tests_ran[1] = run_test(scheduler_path_management_test, local_config,
                              "Path management test")
local_tests_ran[2] = run_test(scheduler_checkpoint_restart_test, local_config,
                              "Checkpoint/restart test")
local_tests_ran[3] = run_test(scheduler_job_status_test, local_config,
                              "Job status test")
local_tests_ran[4] = run_test(scheduler_restart_job_test, local_config,
                              "Restart job test")

# Run mock HPC tests
print("\n" + "=" * 60)
print("Running Mock HPC Scheduler Tests")
print("=" * 60)
mock_tests_ran = [False] * 5
mock_tests_ran[0] = run_test(
    scheduler_mock_slurm_submit_test,
    os.path.join(test_dir, 'test_inputs/02_mock_slurm.json'),
    "Mock Slurm test")
mock_tests_ran[1] = run_test(
    scheduler_mock_flux_submit_test,
    os.path.join(test_dir, 'test_inputs/03_mock_flux.json'), "Mock Flux test")
mock_tests_ran[2] = run_test(
    scheduler_mock_lsf_submit_test,
    os.path.join(test_dir, 'test_inputs/04_mock_lsf.json'), "Mock LSF test")
# Reuse slurm config for error handling
mock_tests_ran[3] = run_test(
    scheduler_mock_slurm_error_handling_test,
    os.path.join(test_dir, 'test_inputs/02_mock_slurm.json'),
    "Mock error handling test")
mock_tests_ran[4] = run_test(
    scheduler_mock_batch_file_generation_test,
    os.path.join(test_dir, 'test_inputs/05_mock_batch.json'),
    "Mock batch file generation test")

# Run real HPC tests if available
print("\n" + "=" * 60)
print("Running Real HPC Scheduler Tests (if available)")
print("=" * 60)
real_tests_ran = [False] * 3

if check_slurm_available():
    print("Slurm detected, running Slurm integration tests...")
    real_tests_ran[0] = run_test(
        scheduler_real_slurm_submit_test,
        os.path.join(test_dir, 'test_inputs/06_real_slurm.json'),
        "Real Slurm integration test")
else:
    print("Slurm not available, skipping Slurm integration tests")

if check_flux_available():
    print("Flux detected, running Flux integration tests...")
    real_tests_ran[1] = run_test(
        scheduler_real_flux_submit_test,
        os.path.join(test_dir, 'test_inputs/07_real_flux.json'),
        "Real Flux integration test")
else:
    print("Flux not available, skipping Flux integration tests")

if check_lsf_available():
    print("LSF detected, running LSF integration tests...")
    real_tests_ran[2] = run_test(
        scheduler_real_lsf_submit_test,
        os.path.join(test_dir, 'test_inputs/08_real_lsf.json'),
        "Real LSF integration test")
else:
    print("LSF not available, skipping LSF integration tests")

# Summary and pytest validation
test_strings = []
local_validation_tests = [
    'test_local_scheduler_submit[LocalScheduler]',
    'test_scheduler_paths[0]',
    'test_scheduler_checkpoint[0]',
    'test_scheduler_job_status[LocalScheduler]',
    'test_scheduler_restart_job[LocalScheduler]',
]

print("\n" + "=" * 60)
print("Test Execution Summary")
print("=" * 60)
print(f"Local scheduler tests: {sum(local_tests_ran)}/{len(local_tests_ran)}")
print(f"Mock HPC tests: {sum(mock_tests_ran)}/{len(mock_tests_ran)}")
print(f"Real HPC tests: {sum(real_tests_ran)}/{len(real_tests_ran)}")

for ran, test in zip(local_tests_ran, local_validation_tests):
    if ran:
        test_strings.append(f'test_scheduler.py::{test}')

if any(mock_tests_ran[:3]):
    for scheduler in ['Slurm', 'Flux', 'LSF']:
        test_strings.append(
            f'test_scheduler.py::test_mock_hpc_submit[{scheduler}]')

if len(test_strings) > 0:
    print("\n" + "=" * 60)
    print("Running pytest validation")
    print("=" * 60)
    os.chdir(test_dir)
    exit_code = pytest.main(['-v', '--tb=short'] + test_strings)
    os.chdir(original_dir)
    clean_test_dirs()
    sys.exit(exit_code)
else:
    print('\nNo tests ran successfully')
    clean_test_dirs()
    sys.exit(1)
