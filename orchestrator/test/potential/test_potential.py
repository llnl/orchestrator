import pytest
from pathlib import Path
from orchestrator.test.reference_output.compare_outputs import compare_outputs

ref_dir = 'TEST_PATH/reference_output/'
test_dir = 'INSTALL_PATH/'


@pytest.mark.parametrize('job_id', [3, 8])
def test_fitsnap(job_id):

    if job_id == 3:
        dir_num = 0
    elif job_id == 8:
        dir_num = 2

    test_file = (f'{test_dir}/potential/potential/SNAPPotential/training/'
                 f'0000{dir_num}/snap_potential.md')
    ref_file = (f'{ref_dir}/potential/03_unit_test/00000/'
                f'snap_potential.md')
    compare_outputs(ref_file, test_file)
    test_file = (f'{test_dir}/potential/potential/SNAPPotential/training/'
                 f'0000{dir_num}/snap_potential.snapcoeff')
    ref_file = (f'{ref_dir}/potential/03_unit_test/00000/'
                f'snap_potential.snapcoeff')
    compare_outputs(ref_file, test_file, skip_lines=1)


@pytest.mark.parametrize('job_id', [4, 9])
def test_fitsnap_weighted(job_id):

    if job_id == 4:
        dir_num = 1
    elif job_id == 9:
        dir_num = 3

    ref_file = (f'{ref_dir}/potential/04_unit_test/00000/'
                f'snap_potential.md')
    test_file = (f'{test_dir}/potential/potential/SNAPPotential/training/'
                 f'0000{dir_num}/snap_potential.md')
    compare_outputs(ref_file, test_file)
    ref_file = (f'{ref_dir}/potential/04_unit_test/00000/'
                f'snap_potential.snapcoeff')
    test_file = (f'{test_dir}/potential/potential/SNAPPotential/training/'
                 f'0000{dir_num}/snap_potential.snapcoeff')
    compare_outputs(ref_file, test_file, skip_lines=1)


@pytest.mark.parametrize('job_id', [0, 1, 2, 3, 4, 5])
def test_potential_did_not_crash(job_id):
    # Doesn't need to do anything, we're just checking that the constructor
    # didn't crash
    pass


@pytest.mark.parametrize('job_id', [5])
def test_nequip_train(job_id):
    """Check if the best.ckpt was generated in the Hydra output directory."""

    # Define the root output directory
    output_root = Path(f"{test_dir}/potential/potential"
                       "/NequIPAllegroPotential/training/00000/outputs")

    # 1. Look for the 'best.ckpt' file using glob.
    # The pattern matches: any_date / any_time / logger_name / best.ckpt
    checkpoint_pattern = "**/nequip_trainer_logger/best.ckpt"

    # Get all matches
    checkpoints = list(output_root.glob(checkpoint_pattern))

    # 2. Assertions
    assert len(checkpoints) > 0, f"No best.ckpt found in {output_root}"

    # Optional: If you want to ensure you are looking at the newest one
    # or a specific job, you can further filter the 'checkpoints' list.
    target_checkpoint = checkpoints[-1]

    assert target_checkpoint.exists()
    assert target_checkpoint.is_file()


@pytest.mark.parametrize('job_id', [6])
def test_nequip_train_zero_weights(job_id):
    """Check if the logged force loss is zero, since the test should have been
    run with zero weights on all atoms."""

    # Define the root output directory
    output_root = Path(f"{test_dir}/potential/potential"
                       "/NequIPAllegroPotential/training/00001/outputs")

    # 1. Look for the 'best.ckpt' file using glob.
    # The pattern matches: any_date / any_time / logger_name / best.ckpt
    checkpoint_pattern = "**/nequip_trainer_logger/best.ckpt"

    # Get all matches
    checkpoints = list(output_root.glob(checkpoint_pattern))

    # 2. Assertions
    assert len(checkpoints) > 0, f"No best.ckpt found in {output_root}"

    # Optional: If you want to ensure you are looking at the newest one
    # or a specific job, you can further filter the 'checkpoints' list.
    target_checkpoint = checkpoints[-1]

    assert target_checkpoint.exists()
    assert target_checkpoint.is_file()

    # Additionally check if error logs were generated
    metrics_pattern = "**/nequip_trainer_logger/version_0/metrics.csv"
    metric_logs = list(output_root.glob(metrics_pattern))
    assert len(metric_logs) > 0, f"No metrics.csv found in {output_root}"

    import pandas as pd
    metrics = pd.read_csv(metric_logs[0])

    assert metrics['train_loss_epoch/forces_weighted_loss'].max() == 0


@pytest.mark.parametrize('job_id', [7])
def test_nequip_train_error_logging(job_id):
    """Check if the error logs were generated in the Hydra output directory."""

    # Define the root output directory
    output_root = Path(f"{test_dir}/potential/potential"
                       "/NequIPAllegroPotential/training/00002/outputs")

    # 1. Look for the 'best.ckpt' file using glob.
    # The pattern matches: any_date / any_time / logger_name / best.ckpt
    checkpoint_pattern = "**/nequip_trainer_logger/best.ckpt"

    # Get all matches
    checkpoints = list(output_root.glob(checkpoint_pattern))

    # 2. Assertions
    assert len(checkpoints) > 0, f"No best.ckpt found in {output_root}"

    # Optional: If you want to ensure you are looking at the newest one
    # or a specific job, you can further filter the 'checkpoints' list.
    target_checkpoint = checkpoints[-1]

    assert target_checkpoint.exists()
    assert target_checkpoint.is_file()

    # Additionally check if error logs were generated
    log_pattern = "**/nequip_trainer_logger/train_force_errors_epoch_0.npy"

    # Get all matches
    error_logs = list(output_root.glob(log_pattern))

    assert len(error_logs) > 0, f"No error logs found in {output_root}"
