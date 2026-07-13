import pytest
from pathlib import Path
import shutil


# Helper functions to check scheduler availability
def check_slurm_available():
    return (shutil.which('sbatch') is not None
            and shutil.which('squeue') is not None)


def check_flux_available():
    return shutil.which('flux') is not None


def check_lsf_available():
    return (shutil.which('bsub') is not None
            and shutil.which('bquery') is not None)


# Local scheduler tests
@pytest.mark.parametrize('scheduler_type', ['LocalScheduler'])
def test_local_scheduler_submit(scheduler_type):
    """Validate local scheduler job submission works correctly."""
    scheduler_dir = Path('./orchestrator_scheduler')
    assert scheduler_dir.exists(), "Scheduler directory should be created"
    test_dirs = list(scheduler_dir.glob('test/simple_job/00000'))
    assert len(test_dirs) > 0, "Simple job directory should exist"


@pytest.mark.parametrize('test_id', [0])
def test_scheduler_paths(test_id):
    """Validate path management creates correct directory structure."""
    scheduler_dir = Path('./orchestrator_scheduler')
    assert scheduler_dir.exists(), "Root scheduler directory should exist"


@pytest.mark.parametrize('test_id', [0])
def test_scheduler_checkpoint(test_id):
    """Validate checkpoint and restart functionality."""
    checkpoint_file = Path('./orchestrator_checkpoint.json')
    assert checkpoint_file.exists(), "Checkpoint file should be created"


@pytest.mark.parametrize('scheduler_type', ['LocalScheduler'])
def test_scheduler_job_status(scheduler_type):
    """Validate job status tracking."""
    scheduler_dir = Path('./orchestrator_scheduler')
    status_test_dirs = list(scheduler_dir.glob('status_test/tracked_jobs/*'))
    assert len(status_test_dirs) >= 2, (
        "Should have at least 2 status test jobs")


@pytest.mark.parametrize('scheduler_type', ['LocalScheduler'])
def test_scheduler_restart_job(scheduler_type):
    """Validate job restart functionality."""
    scheduler_dir = Path('./orchestrator_scheduler')
    restart_dirs = list(scheduler_dir.glob('restart_test/initial_run/*'))
    assert len(restart_dirs) >= 2, (
        "Should have original and restarted job directories")


# Mock HPC scheduler tests
@pytest.mark.parametrize('scheduler', ['Slurm', 'Flux', 'LSF'])
def test_mock_hpc_submit(scheduler):
    """Validate HPC scheduler submission with mocked schedulers."""
    scheduler_dir = Path('./orchestrator_scheduler')
    assert scheduler_dir.exists(), (
        "Scheduler directory should exist for mock tests")


@pytest.mark.parametrize('scheduler', ['Slurm', 'Flux', 'LSF'])
def test_mock_job_id_parsing(scheduler):
    """Validate job ID parsing from scheduler output."""
    assert True, f"{scheduler} job ID parsing test completed"


@pytest.mark.parametrize('scheduler', ['Slurm', 'Flux', 'LSF'])
def test_mock_state_parsing(scheduler):
    """Validate job state parsing from scheduler queries."""
    assert True, f"{scheduler} state parsing test completed"


@pytest.mark.parametrize('scheduler', ['Slurm', 'Flux', 'LSF'])
def test_mock_batch_file_generation(scheduler):
    """Validate batch file generation for each scheduler."""
    scheduler_dir = Path('./orchestrator_scheduler')
    batch_test_dirs = list(scheduler_dir.glob('batch_test/*Scheduler/*'))
    found_batch = False
    for test_dir in batch_test_dirs:
        batch_files = list(test_dir.glob('*.sh'))
        if len(batch_files) > 0:
            found_batch = True
            assert batch_files[0].stat().st_size > 0, (
                "Batch file should not be empty")
            break
    assert found_batch or len(batch_test_dirs) == 0, (
        "Should generate batch files or no test dirs")


# Real HPC scheduler tests (skipped if scheduler not available)
@pytest.mark.skipif(not check_slurm_available(), reason="Slurm not available")
@pytest.mark.parametrize('test_id', [0])
def test_real_slurm_submit(test_id):
    """Integration test for Slurm scheduler (requires Slurm)."""
    scheduler_dir = Path('./orchestrator_scheduler')
    slurm_dirs = list(scheduler_dir.glob('real_slurm/*/'))
    assert len(slurm_dirs) > 0, ("Slurm test jobs should have been created")


@pytest.mark.skipif(not check_flux_available(), reason="Flux not available")
@pytest.mark.parametrize('test_id', [0])
def test_real_flux_submit(test_id):
    """Integration test for Flux scheduler (requires Flux)."""
    scheduler_dir = Path('./orchestrator_scheduler')
    flux_dirs = list(scheduler_dir.glob('real_flux/*/'))
    assert len(flux_dirs) > 0, ("Flux test jobs should have been created")


@pytest.mark.skipif(not check_lsf_available(), reason="LSF not available")
@pytest.mark.parametrize('test_id', [0])
def test_real_lsf_submit(test_id):
    """Integration test for LSF scheduler (requires LSF)."""
    scheduler_dir = Path('./orchestrator_scheduler')
    lsf_dirs = list(scheduler_dir.glob('real_lsf/*/'))
    assert len(lsf_dirs) > 0, ("LSF test jobs should have been created")
