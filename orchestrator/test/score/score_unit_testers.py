import json
import numpy as np
import os
from orchestrator.utils.data_standard import (
    METADATA_KEY,
    ENERGY_KEY,
    FORCES_KEY,
)
from orchestrator.utils.input_output import safe_write, safe_read
from orchestrator.utils.setup_input import init_and_validate_module_type


def atom_level_score_unit_test(input_file: str) -> bool:
    """
    Basic test of atom-level scores.

    :param input_file: input file path with requisite module blocks.
        score is required, scheduler and storage are optional. If score key
        isn't present the whole input is taken as the score args
    :type input_file: str
    :returns: boolean flag that the function completed execution. Does not
        necessarily indicate a correct output, but is used to determine if the
        pytest should be run
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        all_inputs = json.load(fin)

    score_inputs = all_inputs.get('score')
    storage_inputs = all_inputs.get('storage')

    # create the modules
    score = init_and_validate_module_type('score', all_inputs)
    scheduler = init_and_validate_module_type('scheduler', all_inputs)
    if scheduler is None:
        scheduler = score.default_scheduler
    storage = init_and_validate_module_type('storage', all_inputs)

    query_configs = safe_read(score_inputs['query_configs_path'], index=':')
    for atoms in query_configs:
        # Reference data uses these keys
        atoms.info[ENERGY_KEY] = atoms.info['Energy']
        atoms.arrays[FORCES_KEY] = atoms.arrays['force']

    calc_ids = score.run(
        path_type=score_inputs['path_type'],
        compute_args=score_inputs.get('compute_args', {}),
        configs=query_configs,
        scheduler=scheduler,
        job_details=score_inputs.get('job_details', {}),
        batch_size=all_inputs.get('batch_size', 1),
    )
    scheduler.block_until_completed(calc_ids)

    if storage is not None:

        if storage_inputs['storage_type'] == 'COLABFIT':
            storage.set_default_property_map()
            property_def = score.get_colabfit_property_definition()
            existing_properties = [
                _['property-name'] for _ in storage.get_property_definitions()
            ]
            if property_def['property-name'] not in existing_properties:
                storage.define_new_properties(property_def)

            storage.add_property_mapping(
                new_property_name=score.OUTPUT_KEY.replace("_", "-"),
                new_map=score.get_colabfit_property_map())

            # stress not used in this reference data
            del storage.property_map['cauchy-stress']

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

        new_handle = score.save_labeled_configs(
            calc_ids,
            dataset_name=storage_inputs['storage_args']['dataset_name'],
            scheduler=scheduler,
            storage=storage,
            cleanup=False,
        )
        saved_data = storage.get_data(new_handle)

        assert len(saved_data) == len(query_configs), "Storage returned " + \
            "incorrect number of configurations. " + \
            f"Expected {len(query_configs)}, but got {len(saved_data)}"

        if storage_inputs['storage_type'] == 'COLABFIT':
            # ColabFit slugifies the keys
            key_name = score.OUTPUT_KEY.replace('-', '_') + '_score'
        else:
            key_name = score.OUTPUT_KEY + '_score'

        i = 0  # for handling batched results
        for calc_id in calc_ids:
            save_path = scheduler.get_job_path(calc_id)

            # check how many results there should be
            dummy_configs = safe_read(os.path.join(save_path,
                                                   score.data_file_name),
                                      index=':')  # always returns a list

            n = len(dummy_configs)
            assert n > 0, 'No results found at "{save_path}"'

            batched_results = []  # for writing out the batched results
            for _ in range(n):
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
                if 'ltau_forces_uq_train_descriptors' in atoms.info:
                    del atoms.info['ltau_forces_uq_train_descriptors']
                if 'ltau_forces_uq_error_trajectory' in atoms.info:
                    del atoms.info['ltau_forces_uq_error_trajectory']
                if 'quests_delta_entropy_reference_set' in atoms.info:
                    del atoms.info['quests_delta_entropy_reference_set']

                batched_results.append(atoms)
                i += 1
            safe_write(f'{save_path}/saved_config.extxyz', batched_results)

        storage.delete_dataset(new_handle)

    return True


def atom_level_score_unit_test_with_in_memory_arrays(input_file: str) -> bool:
    """
    Test of the score module specific for handling in memory array input.

    :param input_file: input file path with requisite module blocks.
        score is required, scheduler and storage are optional. If
        score key isn't present the whole input is taken as the
        score args
    :type input_file: str
    :returns: boolean flag that the function completed execution. Does not
        necessarily indicate a correct output, but is used to determine if the
        pytest should be run
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        all_inputs = json.load(fin)

    score_inputs = all_inputs.get('score')
    storage_inputs = all_inputs.get('storage')

    # convert score_args to in-memory arrays
    for key in ['train_descriptors', 'error_pdfs', 'reference_set']:
        try:
            # first, check if it's in int_args
            score_inputs['score_args'][key] = np.load(
                score_inputs['score_args'][key])
        except KeyError:
            try:
                # instead, see if it's in compute_args
                score_inputs['compute_args'][key] = np.load(
                    score_inputs['compute_args'][key])
            except KeyError:
                # key may not exist for all tests
                pass

    # create the modules
    score = init_and_validate_module_type('score', all_inputs)
    scheduler = init_and_validate_module_type('scheduler', all_inputs)
    if scheduler is None:
        scheduler = score.default_scheduler
    storage = init_and_validate_module_type('storage', all_inputs)

    query_configs = safe_read(score_inputs['query_configs_path'], index=':')

    calc_ids = score.run(
        path_type=score_inputs['path_type'],
        compute_args=score_inputs.get('compute_args', {}),
        configs=query_configs,
        scheduler=scheduler,
        job_details=score_inputs.get('job_details', {}),
        batch_size=all_inputs.get('batch_size', 1),
    )
    scheduler.block_until_completed(calc_ids)

    if storage is not None:

        if storage_inputs['storage_type'] == 'COLABFIT':
            storage.set_default_property_map()
            property_def = score.get_colabfit_property_definition()
            existing_properties = [
                _['property-name'] for _ in storage.get_property_definitions()
            ]
            if property_def['property-name'] not in existing_properties:
                storage.define_new_properties(property_def)

            storage.add_property_mapping(
                new_property_name=score.OUTPUT_KEY.replace("_", "-"),
                new_map=score.get_colabfit_property_map())

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

        new_handle = score.save_labeled_configs(
            calc_ids,
            dataset_name=storage_inputs['storage_args']['dataset_name'],
            scheduler=scheduler,
            storage=storage,
            cleanup=False,
        )
        saved_data = storage.get_data(new_handle)

        assert len(saved_data) == len(query_configs), "Storage returned " + \
            "incorrect number of configurations. " + \
            f"Expected {len(query_configs)}, but got {len(saved_data)}"

        key_name = score.OUTPUT_KEY + '_score'

        i = 0  # for handling batched results
        for calc_id in calc_ids:
            save_path = scheduler.get_job_path(calc_id)

            # check how many results there should be
            dummy_configs = safe_read(os.path.join(save_path,
                                                   score.data_file_name),
                                      index=':')  # always returns a list

            n = len(dummy_configs)
            assert n > 0, 'No results found at "{save_path}"'

            batched_results = []  # for writing out the batched results
            for _ in range(n):
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
                if 'ltau_forces_uq_train_descriptors' in atoms.info:
                    del atoms.info['ltau_forces_uq_train_descriptors']
                if 'ltau_forces_uq_error_trajectory' in atoms.info:
                    del atoms.info['ltau_forces_uq_error_trajectory']
                if 'quests_delta_entropy_reference_set' in atoms.info:
                    del atoms.info['quests_delta_entropy_reference_set']

                batched_results.append(atoms)
                i += 1
            safe_write(f'{save_path}/saved_config.extxyz', batched_results)

        storage.delete_dataset(new_handle)

    return True


def dataset_level_score_unit_test(input_file: str) -> bool:
    """
    Basic test of dataset-level score.

    :param input_file: input file path with requisite module blocks.
        score is required, scheduler and storage are optional. If
        score key isn't present the whole input is taken as the
        score args
    :type input_file: str
    :returns: boolean flag that the function completed execution. Does not
        necessarily indicate a correct output, but is used to determine if the
        pytest should be run
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        all_inputs = json.load(fin)

    # check if input has sections, if not score is the only input
    score_inputs = all_inputs.get('score')

    # create the modules
    score = init_and_validate_module_type('score', all_inputs)
    scheduler = init_and_validate_module_type('scheduler', all_inputs)
    if scheduler is None:
        scheduler = score.default_scheduler
    storage = init_and_validate_module_type('storage', all_inputs)

    dataset = safe_read(score_inputs['dataset_path'], index=':')

    calc_ids = score.run(
        path_type=score_inputs['path_type'],
        compute_args=score_inputs.get('compute_args', {}),
        configs=dataset,
        scheduler=scheduler,
        job_details=score_inputs.get('job_details', {}),
        batch_size=all_inputs.get('batch_size', 1),
    )
    scheduler.block_until_completed(calc_ids)

    if storage is not None:
        raise NotImplementedError(
            "Storage of dataset-level scores is not supported yet")

    # output JSON files copied manually to reference data

    return True


def model_level_score_unit_test(input_file: str) -> bool:
    """
    Basic test of model-level scores.

    This function is mostly the same as `dataset_level_score_unit_test`, with
    slight modification in some arguments names and reading dataset.

    :param input_file: input file path with requisite module blocks.
        score is required, scheduler and storage are optional. If
        score key isn't present the whole input is taken as the
        score args
    :type input_file: str
    :returns: boolean flag that the function completed execution. Does not
        necessarily indicate a correct output, but is used to determine if the
        pytest should be run
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        all_inputs = json.load(fin)

    # check if input has sections, if not score is the only input
    score_inputs = all_inputs.get('score', all_inputs)
    scheduler_inputs = all_inputs.get('scheduler', None)
    storage_inputs = all_inputs.get('storage', None)

    # create the modules
    score = init_and_validate_module_type('score',
                                          score_inputs,
                                          single_input_dict=True)
    if scheduler_inputs:
        scheduler = init_and_validate_module_type('scheduler',
                                                  scheduler_inputs,
                                                  single_input_dict=True)
    else:
        scheduler = score.default_scheduler
    if storage_inputs:
        storage = init_and_validate_module_type('storage',
                                                storage_inputs,
                                                single_input_dict=True)
    else:
        storage = None

    dataset = score.read_data(score_inputs['dataset_path'])

    calc_ids = score.run(
        score_inputs['path_type'],
        dataset,
        score_inputs.get('compute_args', {}),
        scheduler,
        score_inputs.get('job_details', {}),
        all_inputs.get('batch_size', 1),
    )

    scheduler.block_until_completed(calc_ids)

    if storage is not None:
        raise NotImplementedError(
            "Storage of model-level scores is not supported yet")

    # output JSON files copied manually to reference data

    return True
