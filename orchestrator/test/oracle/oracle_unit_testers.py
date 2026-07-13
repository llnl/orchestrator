import json
import os
from ase.io import read
from orchestrator.utils.setup_input import init_and_validate_module_type
from orchestrator.utils.input_output import safe_write


def oracle_unit_test(input_file: str) -> bool:
    """
    basic test of the oracle module

    :param input_file: input file path with requisite module blocks.
        oracle is required, scheduler and storage are optional. If
        oracle key isn't present the whole input is taken as the
        oracle args
    :type input_file: str
    :returns: boolean flag that the function completed execution. Does not
        necessarily indicate a correct output, but is used to determine if the
        pytest should be run
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        all_inputs = json.load(fin)

    # check if input has sections, if not oracle is the only input
    oracle_inputs = all_inputs.get('oracle', all_inputs)
    # create the modules
    oracle = init_and_validate_module_type('oracle', oracle_inputs, True)
    scheduler = init_and_validate_module_type('scheduler', all_inputs)
    if scheduler is None:
        scheduler = oracle.default_scheduler
    storage = init_and_validate_module_type('storage', all_inputs)

    # read with ASE?
    configs = []
    for root, _, files in os.walk(oracle_inputs.get('data_path'),
                                  followlinks=True):
        for i, f in enumerate(files):
            if i < 3:
                configs.append(read(f'{root}/{f}'))
    # sort as storage.colabfit does
    if storage is not None:
        storage.set_default_property_map()
        configs = storage.sort_configurations(configs)

    calc_ids = oracle.run(
        oracle_inputs.get('path_type'),
        oracle_inputs.get('extra_input_args', {}),
        configs,
        scheduler=scheduler,
        job_details=oracle_inputs.get('job_details', {}),
    )
    scheduler.block_until_completed(calc_ids)

    if storage is not None:
        new_handle = oracle.save_labeled_configs(
            calc_ids,
            storage=storage,
            dataset_name='oracle_unit_test',
            scheduler=scheduler,
        )
        # TODO - update test checking to extract from storage?
        print(f'Labeled configurations saved to {new_handle}')
        saved_data = storage.get_data(new_handle)
        for config, calc_id in zip(saved_data, calc_ids):
            # remove po/ds-id as it will always vary
            # remove metadata as it depends on user
            config.info.pop('ds-id')
            config.info.pop('po-id')
            save_path = scheduler.get_job_path(calc_id)
            if save_path is None:
                save_path = scheduler.make_path(oracle.__class__.__name__,
                                                oracle_inputs.get('path_type'))
            safe_write(f'{save_path}/saved_config.extxyz', config)
        storage.delete_dataset(new_handle)
    else:
        for calc_id in calc_ids:
            run_path = scheduler.get_job_path(calc_id)
            config = oracle.parse_for_storage(run_path, calc_id, scheduler)
            # remove metadata as it is user-dependent
            _ = config.info.pop('_metadata')
            safe_write(f'{run_path}/saved_config.extxyz', config)

    return True
