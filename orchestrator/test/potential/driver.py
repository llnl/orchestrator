from orchestrator.test.potential.potential_unit_testers import (
    potential_instantiate_does_not_crash_test,
    potential_save_without_training,
    potential_train_test,
    potential_submit_train_test,
    potential_train_test_zero_per_atom_weights,
    potential_kimkit_combined_test,
)
import os
import pytest
import shutil
import glob


def clean_test_dirs():
    # delete lingering files and directories after tests complete
    delete_paths = [
        "./potential/",
    ]
    delete_files = ["orchestrator_checkpoint.json"]
    glob_target_dirs = [
        "SW_StillingerWeber_1985_Si__MO_*",
        "DNN_Potential_Orchestrator_Generated_Si__MO_*",
        "Test_Model_Si__MO_*",
        "*.log",
        "*.pkl",
    ]

    for file in delete_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass

    for loc in delete_paths:
        shutil.rmtree(loc, ignore_errors=True)

    for target_dir in glob_target_dirs:
        for f in glob.glob(target_dir):
            shutil.rmtree(f, ignore_errors=True)


clean_test_dirs()

# SNAP tests
tests_ran = [False] * 13
tests_ran[0], _ = potential_instantiate_does_not_crash_test(
    'test_inputs/10_fitsnap_from_scratch_input.json')
tests_ran[1], _ = potential_save_without_training(
    'test_inputs/10_fitsnap_from_scratch_input.json')
tests_ran[2], _ = potential_train_test(
    'test_inputs/03_fitsnap_test_input.json')
tests_ran[3], _ = potential_train_test(
    'test_inputs/04_fitsnap_test_per_atom_weights_input.json')
tests_ran[4], _ = potential_submit_train_test(
    'test_inputs/08_fitsnap_submit_test_input.json')
tests_ran[5], _ = potential_submit_train_test(
    'test_inputs/09_fitsnap_submit_test_per_atom_weights_input.json')
tests_ran[6], _ = potential_kimkit_combined_test(
    'test_inputs/11_fitsnap_from_files.json')

# Nequip tests
tests_ran[7], _ = potential_instantiate_does_not_crash_test(
    'test_inputs/01_nequip_from_scratch.json')
tests_ran[8], _ = potential_save_without_training(
    'test_inputs/01_nequip_from_scratch.json')
tests_ran[9], _ = potential_train_test('test_inputs/02_nequip_train.json')
tests_ran[10], _ = potential_train_test_zero_per_atom_weights(
    'test_inputs/02_nequip_train.json')
tests_ran[11], _ = potential_train_test(
    'test_inputs/07_nequip_error_logging.json')
tests_ran[12], _ = potential_kimkit_combined_test(
    'test_inputs/06_nequip_kim_tests.json')

# now validate the tests:
validation_tests = [
    'test_potential_did_not_crash[0]',
    'test_potential_did_not_crash[1]',
    'test_fitsnap[3]',
    'test_fitsnap_weighted[4]',
    'test_fitsnap[8]',
    'test_fitsnap_weighted[9]',
    'test_potential_did_not_crash[2]',
    'test_potential_did_not_crash[3]',
    'test_potential_did_not_crash[4]',
    'test_nequip_train[5]',
    'test_nequip_train_zero_weights[6]',
    'test_nequip_train_error_logging[7]',
    'test_potential_did_not_crash[5]',
]
test_strings = []
if (len(tests_ran) != len(validation_tests)):
    print('WARNING!! Test and validation lists are different lengths!')
    print('Did you add a test without adding a validation for it?')
    print('Or did you comment out a validation for a skipped test? \
        (Don\'t do that)')
for ran, test in zip(tests_ran, validation_tests):
    if ran:
        test_strings.append(f'test_potential.py::{test}')
    else:
        print(f'{test} was not run / did not complete, omitting from pytest')
if len(test_strings) > 0:
    _ = pytest.main(['-v'] + test_strings)
else:
    print('No tests ran successfully, skipping pytest validation')

# # kimkit tests
# kimkit_fitsnap_test_ran = False

# kimkit_fitsnap_test_ran = potential_kimkit_combined_test(
#     'test_inputs/12_fitsnap_potential_test_input.json')

# if kimkit_fitsnap_test_ran:
#     print('Kimkit/FitSnap test completed, check std output for confirmation')
# else:
#     print('There was a problem running the kimkit/FitSnap integration test')
