# Make sure to change paths here and input_params.json file
# based on the location of your files

# Simply run by "python test_melting.py" on the command line
# on Lassen after activating your Orchestrator environment

# Warning: This test is using "fake" lammps input files
# (much shorter runs for various steps) for a fast evaluation
# of the scheduler for melting point calculations
from orchestrator.test.target_property.target_property_unit_testers import (
    target_property_unit_test, sample_config_unit_test)
import pytest

tests_ran = [False] * 7

tests_ran[0] = target_property_unit_test(
    'test_inputs/simple_elastic_input.json')
tests_ran[1] = target_property_unit_test(
    'test_inputs/potential_elastic_input.json')
tests_ran[2] = target_property_unit_test(
    'test_inputs/copper_standardpress_melting_input.json')
tests_ran[3] = target_property_unit_test(
    'test_inputs/copper_standardpress_melting_input_kim_api.json')
tests_ran[4] = target_property_unit_test(
    'test_inputs/copper_standardpress_melting_ssh2flux_input.json')
tests_ran[5] = sample_config_unit_test()
tests_ran[6] = target_property_unit_test(
    'test_inputs/mixture_properties_default_template.json')

# now validate the tests:
validation_tests = [
    'test_elastic_constants[simple_elastic_output.dat]',
    'test_elastic_constants[potential_elastic_output.dat]',
    'test_melting_point',
    'test_melting_point',
    'test_melting_point',
    'test_config_sampling',
    'test_simple_property',
]
test_strings = []
for ran, test in zip(tests_ran, validation_tests):
    if ran:
        test_strings.append(f'test_target_property.py::{test}')
    else:
        print(f'{test} was not run or did not complete, omitting from pytest')
if len(test_strings) > 0:
    _ = pytest.main(['-v'] + test_strings)
else:
    print('No tests ran successfully, skipping pytest validation')

kimrun_tests_ran = [False] * 4

kimrun_tests_ran[0] = target_property_unit_test(
    'test_inputs/kimkit_kimrun_input.json')
kimrun_tests_ran[1] = target_property_unit_test(
    'test_inputs/existing_potential_kimrun_input.json')
kimrun_tests_ran[2] = target_property_unit_test(
    'test_inputs/potential_kimrun_input.json')
kimrun_tests_ran[3] = target_property_unit_test(
    'test_inputs/kimkit_kimrun_slurm_input.json')

# now validate the tests:
kimrun_validation_tests = [
    'test_kimrun[kimkit_kimrun_output.dat]',
    'test_kimrun[existing_potential_kimrun_output.dat]',
    'test_kimrun[potential_kimrun_output.dat]',
    'test_kimrun[kimkit_kimrun_slurm_output.dat]',
]
kimrun_test_strings = []
for ran, test in zip(kimrun_tests_ran, kimrun_validation_tests):
    if ran:
        kimrun_test_strings.append(f'test_target_property.py::{test}')
    else:
        print(f'{test} was not run or did not complete, omitting from pytest')
if len(kimrun_test_strings) > 0:
    _ = pytest.main(['-v'] + kimrun_test_strings)
else:
    print('No KIMRun tests ran successfully, skipping pytest validation')
