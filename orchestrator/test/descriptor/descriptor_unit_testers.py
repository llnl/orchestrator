import json
import os
from ase.io import read
from orchestrator.utils.input_output import ase_glob_read, safe_write
from orchestrator.utils.setup_input import init_and_validate_module_type
from orchestrator.utils.data_standard import METADATA_KEY, STRESS_KEY
from orchestrator.utils.isinstance import isinstance_no_import


def descriptor_unit_test(input_file: str) -> bool:
    """
    basic test of the descriptor module

    :param input_file: input file path with requisite module blocks.
        descriptors is required, scheduler and storage are optional. If
        descriptors key isn't present the whole input is taken as the
        descriptor args
    :type input_file: str
    :returns: boolean flag that the function completed execution. Does not
        necessarily indicate a correct output, but is used to determine if the
        pytest should be run
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        all_inputs = json.load(fin)

    # check if input has sections, if not descriptor is the only input
    descriptor_inputs = all_inputs.get('descriptors', all_inputs)
    storage_inputs = all_inputs.get('storage', None)
    # create the modules
    descriptor = init_and_validate_module_type('descriptor', all_inputs)
    scheduler = init_and_validate_module_type('scheduler', all_inputs)
    if scheduler is None:
        scheduler = descriptor.default_scheduler
    storage = init_and_validate_module_type('storage', all_inputs)
    if isinstance_no_import(storage, 'ColabfitStorage'):
        storage.set_default_property_map()

    init_config_path = os.path.abspath(all_inputs.get('init_config_path'))
    configs = ase_glob_read(init_config_path)

    calc_ids = descriptor.run(
        path_type=descriptor_inputs['path_type'],
        compute_args=descriptor_inputs.get('compute_args', {}),
        configs=configs,
        scheduler=scheduler,
        job_details=descriptor_inputs.get('job_details', {}),
        batch_size=all_inputs.get('batch_size', 1),
    )
    scheduler.block_until_completed(calc_ids)

    if storage is not None:

        if storage_inputs['storage_type'] == 'COLABFIT':
            # Have to add in an ACSF property, if it doesn't exist yet
            property_def = descriptor.get_colabfit_property_definition()
            existing_properties = [
                _['property-name'] for _ in storage.get_property_definitions()
            ]
            if property_def['property-name'] not in existing_properties:
                storage.define_new_properties(property_def)

            storage.add_property_mapping(
                new_property_name=descriptor.OUTPUT_KEY,
                new_map=descriptor.get_colabfit_property_map())

            # Stress is not available in sample_configs
            if 'cauchy-stress' in storage.property_map:
                del storage.property_map['cauchy-stress']
            if STRESS_KEY in storage.property_map:
                del storage.property_map[STRESS_KEY]

        # Delete any old data, if it exists
        if storage_inputs['storage_type'] == 'COLABFIT':
            # Make sure the descriptors can be extracted properly
            try:
                existing_ds = storage._get_id_from_name(
                    storage_inputs['storage_args']['dataset_name'])
                storage.delete_dataset(existing_ds)

            except Exception:
                pass  # doesn't exist, so no need to delete
        else:
            storage.delete_dataset(
                storage_inputs['storage_args']['dataset_name'])

        new_handle = descriptor.save_labeled_configs(
            calc_ids,
            dataset_name=storage_inputs['storage_args']['dataset_name'],
            scheduler=scheduler,
            storage=storage,
        )

        saved_data = storage.get_data(new_handle)

        assert len(saved_data) == len(configs), "Storage returned " + \
            "incorrect number of configurations. " + \
            f"Expected {len(configs)}, but got {len(saved_data)}"

        key_name = descriptor.OUTPUT_KEY + '_descriptors'

        i = 0  # for handling batched results
        for calc_id in calc_ids:
            save_path = scheduler.get_job_path(calc_id)

            # check how many results there should be
            dummy_configs = read(os.path.join(save_path,
                                              descriptor.atoms_file_name),
                                 index=':')  # always returns a list

            n = len(dummy_configs)
            batched_results = []  # for writing out the batched results
            for j in range(n):
                atoms = saved_data[i]
                assert ((key_name in atoms.arrays) or (key_name in atoms.info)
                        ), f"{key_name} key not found in .arrays or .info"

                # to make sure reference data matching works
                if METADATA_KEY in atoms.info:
                    del atoms.info[METADATA_KEY]
                if "ds-id" in atoms.info:
                    del atoms.info['ds-id']
                if "po-id" in atoms.info:
                    del atoms.info['po-id']
                if "co-id" in atoms.info:
                    del atoms.info['co-id']

                batched_results.append(atoms)
                i += 1
            safe_write(f'{save_path}/saved_config.extxyz', batched_results)

        storage.delete_dataset(new_handle)
    return True
