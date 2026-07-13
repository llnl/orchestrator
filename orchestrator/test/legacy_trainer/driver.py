from orchestrator.test.trainer.trainer_potential_unit_testers import (
    trainer_potential_combined_test,
    trainer_potential_scheduler_test,
    potential_kimkit_combined_test,
    potential_kim_api_integration_test,
)
import os
import pytest
import shutil
import glob


def clean_test_dirs():
    # delete lingering files and directories after tests complete
    delete_paths = [
        "./trainer/",
        "./submit_trainer/",
    ]
    delete_files = ["job_record.pkl", "orchestrator_checkpoint.json"]
    glob_target_dirs = [
        "SW_StillingerWeber_1985_Si__MO_*",
        "DNN_Potential_Orchestrator_Generated_Si__MO_*", "Test_Model_Si__MO_*",
        "*.log"
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

# First, the potential module base tests
tests_ran = [False] * 10
tests_ran[0] = trainer_potential_combined_test(
    'test_inputs/01_dnn_test_input.json')[0]
tests_ran[1] = trainer_potential_combined_test(
    'test_inputs/02_kim_test_input.json')[0]
tests_ran[2] = trainer_potential_combined_test(
    'test_inputs/03_fitsnap_test_input.json')[0]
tests_ran[3] = trainer_potential_combined_test(
    'test_inputs/04_fitsnap_test_per_atom_weights_input.json')
tests_ran[4] = trainer_potential_combined_test(
    'test_inputs/05_chimes_test_input.json')
tests_ran[5] = trainer_potential_combined_test(
    'test_inputs/06_chimes_test_per_atom_weights_input.json')

tests_ran[6] = trainer_potential_scheduler_test(
    'test_inputs/07_dnn_submit_test_input.json')
tests_ran[7] = trainer_potential_scheduler_test(
    'test_inputs/08_fitsnap_submit_test_input.json')
tests_ran[8] = trainer_potential_scheduler_test(
    'test_inputs/09_fitsnap_submit_test_per_atom_weights_input.json')
tests_ran[9] = trainer_potential_scheduler_test(
    'test_inputs/10_chimes_submit_test_input.json')

# now validate the tests:
validation_tests = [
    'test_dnn_trainer_bp_potential[1]',
    'test_parametric_trainer_sw_potential',
    'test_fitsnap_trainer[3]',
    'test_fitsnap_weighted_trainer[4]',
    'test_chimes_trainer[5]',
    'test_chimes_weighted_trainer[6]',
    'test_dnn_trainer_bp_potential[7]',
    'test_fitsnap_trainer[8]',
    'test_fitsnap_weighted_trainer[9]',
    'test_chimes_trainer[10]',
]
test_strings = []
if (len(tests_ran) != len(validation_tests)):
    print('WARNING!! Test and validation lists are different lengths!')
    print('Did you add a test without adding a validation for it?')
    print('Or did you comment out a validation for a skipped test? \
        (Don\'t do that)')
for ran, test in zip(tests_ran, validation_tests):
    if ran:
        test_strings.append(f'test_trainer.py::{test}')
    else:
        print(f'{test} was not run / did not complete, omitting from pytest')
if len(test_strings) > 0:
    _ = pytest.main(['-v'] + test_strings)
else:
    print('No tests ran successfully, skipping pytest validation')

# kimkit tests
kimkit_test_ran = False
kimkit_fitsnap_test_ran = False
kimkit_dnn_test_ran = False

kimkit_test_ran = potential_kimkit_combined_test(
    'test_inputs/11_kim_potential_test_input.json')
kimkit_fitsnap_test_ran = potential_kimkit_combined_test(
    'test_inputs/12_fitsnap_potential_test_input.json')
kimkit_dnn_test_ran = potential_kimkit_combined_test(
    'test_inputs/13_dnn_potential_test_input.json')

if kimkit_test_ran:
    print('Kimkit test completed, check std output for confirmation')
else:
    print('There was a problem running the kimkit integration test')
if kimkit_fitsnap_test_ran:
    print('Kimkit/FitSnap test completed, check std output for confirmation')
else:
    print('There was a problem running the kimkit/FitSnap integration test')
if kimkit_dnn_test_ran:
    print('KIMKit/DNN test completed, check std output for confirmation')
else:
    print('There was a problem running the KIMKit/DNN integration test')

# Finally, test integrtation with the KIM_API
potential_kim_api_test_ran = False
potential_kim_api_fitsnap_test_ran = False
potential_kim_api_chimes_test_ran = False

potential_kim_api_test_ran = potential_kim_api_integration_test(
    'test_inputs/14_kim_api_test_input.json')
potential_kim_api_fitsnap_test_ran = potential_kim_api_integration_test(
    'test_inputs/15_kim_api_fitsnap_test_input.json')
potential_kim_api_chimes_test_ran = potential_kim_api_integration_test(
    'test_inputs/16_kim_api_chimes_test_input.json')

if potential_kim_api_test_ran:
    print(
        'Potential/KIM_API test completed, check std output for confirmation')
else:
    print('There was a problem running the Potential/KIM_API integration test')
if potential_kim_api_fitsnap_test_ran:
    print('Potential/KIM_API FitSnap test completed, check std output for '
          'confirmation')
else:
    print('There was a problem running the Potential/KIM_API FitSnap '
          'integration test')
if potential_kim_api_chimes_test_ran:
    print('Potential/KIM_API ChIMES test completed, check std output for '
          'confirmation')
else:
    print('There was a problem running the Potential/KIM_API ChIMES '
          'integration test')
