import pytest
from orchestrator.test.reference_output.compare_outputs import compare_outputs

ref_dir = 'TEST_PATH/reference_output'
test_dir = 'INSTALL_PATH'


def test_simple_property():
    """
    Test the simple property target property module

    In this test the  simple property calculations are run from the provided
    input file. The calculated simple properties are saved to
    mixture_properties_default_template.dat and is the basis for comparison
    in this test.
    """
    ref_file = (f'{ref_dir}/target_property/'
                'mixture_properties_default_template.dat')
    test_file = (f'{test_dir}/target_property/'
                 'mixture_properties_default_template.dat')
    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_path', [
    'copper_standardpress_melting_output.dat',
    'copper_standardpress_melting_output_kim_api.dat',
    'copper_standardpress_melting_ssh2flux_output.dat',
])
def test_melting_point(job_path):
    """
    Test the melting point target property module

    In this test the melting point calculations are run from the provided
    input file. The calculated melting point is saved to melting_temp.dat
    and is the basis for comparison in this test. This test uses copper
    standard pressure input file
    """

    ref_file = (f'{ref_dir}/target_property/{job_path}')
    test_file = (f'{test_dir}/target_property/{job_path}')
    compare_outputs(ref_file, test_file)


def test_config_sampling():

    ref_file = (f'{ref_dir}/target_property/'
                'sampled_configs_npt.xyz')
    test_file = (f'{test_dir}/target_property/'
                 'sampled_configs_npt.xyz')
    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_path', [
    'simple_elastic_output.dat',
    'potential_elastic_output.dat',
])
def test_elastic_constants(job_path):
    """
    Test the elastic constants target property module

    In this test the elastic constant calculations are run from the provided
    input file. The calculated constants are saved to C_tensor.dat
    and is the basis for comparison in this test. This test uses tantalum
    """
    ref_file = (f'{ref_dir}/target_property/{job_path}')
    test_file = (f'{test_dir}/target_property/{job_path}')
    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_path', [
    'kimkit_kimrun_output.dat',
    'existing_potential_kimrun_output.dat',
    'potential_kimrun_output.dat',
    'kimkit_kimrun_slurm_output.dat',
])
def test_kimrun(job_path):
    """
    Test the KIMRun target property module

    In this test some simple KIM tests are run against
    a LAMMPS Simulator model for W and
    the Stillinger Weber PM for Si and compared against
    known good results.
    """
    ref_file = (f'{ref_dir}/target_property/{job_path}')
    test_file = (f'{test_dir}/target_property/{job_path}')
    compare_outputs(ref_file, test_file)
