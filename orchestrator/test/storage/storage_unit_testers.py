import json
from ase import Atoms
from getpass import getuser
from orchestrator.utils.input_output import ase_glob_read
from orchestrator.utils.setup_input import init_and_validate_module_type
from orchestrator.utils.data_standard import (
    ENERGY_KEY,
    FORCES_KEY,
)
from orchestrator.utils.isinstance import isinstance_no_import
from orchestrator.utils.exceptions import DuplicateDatasetNameError


def storage_unit_test(input_file: str) -> bool:
    """
    basic test of the storage module

    :param input_file: input file path with requisite module blocks.
        storage is required, auxiliary keys are dataset_name and
        init_config_path
    :type input_file: str
    :returns: boolean flag that the function completed execution. Does not
        necessarily indicate a correct output, but is used to determine if the
        pytest should be run
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    storage = init_and_validate_module_type('storage', test_inputs)
    if isinstance_no_import(storage, 'ColabfitStorage'):
        # best practice to explicitly set property map
        storage.set_property_map({
            'energy_field': ENERGY_KEY,
            'force_field': FORCES_KEY
        })

    dataset_name = test_inputs.get('dataset_name')
    dataset_name += f'_{getuser()}'
    init_config_path = test_inputs.get('init_config_path')

    # read in all the data
    configs = ase_glob_read(init_config_path)
    cleaned_configs = []
    for config in configs:
        # copy structural information about the config
        clean_config = Atoms(symbols=config.symbols,
                             cell=config.cell,
                             pbc=config.pbc,
                             positions=config.positions)
        # set the energy
        clean_config.info[ENERGY_KEY] = config.info[ENERGY_KEY]
        # set the forces
        clean_config.set_array(FORCES_KEY, config.arrays[FORCES_KEY])
        cleaned_configs.append(clean_config)

    new_data = ase_glob_read('./test_inputs')
    new_cleaned_configs = []
    for config in new_data:
        # copy structural information about the config
        clean_config = Atoms(symbols=config.symbols,
                             cell=config.cell,
                             pbc=config.pbc,
                             positions=config.positions)
        # set the energy
        clean_config.info[ENERGY_KEY] = config.get_potential_energy()
        # set the forces
        clean_config.set_array(FORCES_KEY, config.get_forces())
        new_cleaned_configs.append(clean_config)

    try:
        # initialize database
        try:
            if isinstance_no_import(storage, 'ColabfitStorage'):
                init_handle = storage.new_dataset(dataset_name,
                                                  cleaned_configs,
                                                  strict=True)
            else:
                init_handle = storage.new_dataset(dataset_name,
                                                  cleaned_configs)
            print(f'Initialized dataset: {init_handle}')
        except DuplicateDatasetNameError as e:
            # catch if cleanup didn't happen previously
            print('Dataset already exists, deleting and trying again')
            problem_name = e.args[0].split()[-1]
            storage.delete_dataset(problem_name)
            if isinstance_no_import(storage, 'ColabfitStorage'):
                init_handle = storage.new_dataset(dataset_name,
                                                  cleaned_configs,
                                                  strict=True)
            else:
                init_handle = storage.new_dataset(dataset_name,
                                                  cleaned_configs)
            print(f'Initialized dataset: {init_handle}')

        dataset = storage.get_data(init_handle)
        init_len = len(dataset)
        print(f'Number of configurations in the original dataset: {init_len}')

        if isinstance_no_import(storage, 'ColabfitStorage'):
            new_handle = storage.add_data(init_handle, new_cleaned_configs)
        else:
            new_handle = storage.add_data(init_handle, new_cleaned_configs)
        print(f'Added to dataset: {new_handle}')
        dataset = storage.get_data(new_handle)
        added_len = len(dataset)
        print(f'Number of configurations in the new dataset: {added_len}')
    except Exception as e:
        raise e
    try:
        if isinstance_no_import(storage, 'ColabfitStorage'):
            queried_output = storage.list_data(dataset_handle=new_handle,
                                               capture_output=True)
            assert isinstance(queried_output[0], dict), "DB Query Fail"
    except Exception as e:
        raise e
    try:
        if isinstance_no_import(storage, 'ColabfitStorage'):
            ex_a = b'----\n{\'id\': \'DS\',\n \'n\': \'my_n\'}'
            ex_b = (b'----\n{\'d\': \'DS\',\n \'n\': \'my_n\','
                    + b'\n \'desc\': \'{"n": "my_n"\'}')
            ex_c = (b'----\n{\'d\': \'DS\',\n \'n\': \'my_n\','
                    + b'\n \'desc\': \'{\\\\"n\\\\": \\\\"my_n\\\\"\'}')
            for ex in [ex_a, ex_b, ex_c]:
                cleaned = storage._clean_output_dictionary_binary(ex)
                assert isinstance(json.loads(cleaned),
                                  dict), f"Clean json fail for {ex}"
    except Exception as e:
        raise e
    else:
        if init_len == 9 and added_len == 10:
            test_name = input_file.split('/')[-1]
            print(f'storage unit test for {test_name} was succesful!')
    finally:
        # always clean up afterwards
        try:
            storage.delete_dataset(new_handle)
            if init_handle != new_handle:
                storage.delete_dataset(init_handle)
        except Exception as e:
            print(f'{e}---No dataset handles saved to delete')

    return True
