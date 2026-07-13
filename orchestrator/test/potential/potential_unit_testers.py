import json
import os
from dataclasses import asdict, is_dataclass
from orchestrator.utils.setup_input import init_and_validate_module_type


def potential_instantiate_does_not_crash_test(input_file):

    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    potential = init_and_validate_module_type('potential', test_inputs)

    if is_dataclass(potential.hyperparameters):
        hparams = asdict(potential.hyperparameters)
    elif isinstance(potential.hyperparameters, dict):
        hparams = potential.hyperparameters
    else:
        raise TypeError('hyperparameters are of unexpected type')
    # Ensure that hyperparameters were set correctly
    for k, v in hparams.items():
        if k in test_inputs['potential']['potential_args']:
            assert v == test_inputs['potential']['potential_args'][k]

    return True, potential


def potential_save_without_training(input_file):

    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    potential = init_and_validate_module_type('potential', test_inputs)

    import pytest

    # Assert that saving the file without training raises a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        potential.save_potential('new_model', makedirs=True)

    return True, potential


def potential_train_test(input_file: str) -> bool:
    """
    basic test of the trainer and potential modules

    :param input_file: input file path with requisite module blocks.
        potential, trainer, and storage are required
    :type input_file: str
    :returns: boolean flag that the function completed execution. Does not
        necessarily indicate a correct output, but is used to determine if the
        pytest should be run
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    potential = init_and_validate_module_type('potential', test_inputs)
    storage = init_and_validate_module_type('storage', test_inputs)
    scheduler = init_and_validate_module_type('scheduler', test_inputs)

    # Add some keys if they don't exist so we can do **test_inputs
    test_inputs['training_args']['job_details'] = test_inputs[
        'training_args'].get('job_details', {})
    per_atom_weights = test_inputs['training_args'].get('per_atom_weights')
    if type(per_atom_weights) is str:
        if not os.path.isabs(per_atom_weights):
            cwd = os.path.abspath('.')
            abs_per_atom_weights = f'{cwd}/{per_atom_weights}'
            # overwrite with the abs path
            test_inputs['training_args'][
                'per_atom_weights'] = abs_per_atom_weights

    _, _ = potential.train(
        test_inputs['dataset_handle'],
        storage,
        scheduler,
        **test_inputs['training_args'],
    )
    return True, potential


def potential_train_test_zero_per_atom_weights(input_file: str) -> bool:
    """
    Same as the typical training unit test, but using zero-weights to ensure
    atom-level weighting is working correctly

    :param input_file: input file path with requisite module blocks.
        potential, trainer, and storage are required
    :type input_file: str
    :returns: boolean flag that the function completed execution. Does not
        necessarily indicate a correct output, but is used to determine if the
        pytest should be run
    :rtype: bool
    """
    import os

    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    potential = init_and_validate_module_type('potential', test_inputs)
    storage = init_and_validate_module_type('storage', test_inputs)
    scheduler = init_and_validate_module_type('scheduler', test_inputs)

    # delete the temp dir in case previous runs
    if test_inputs.get('path_type') is not None:
        try:
            os.rmdir(test_inputs.get('path_type'))
        except FileNotFoundError:
            pass

    # storage module needs to already possess the dataset... TODO
    # could add here as part of unit test, but probably better to keep separate

    # Add some keys if they don't exist so we can do **test_inputs
    test_inputs['training_args']['job_details'] = test_inputs[
        'training_args'].get('job_details', {})

    per_atom_weights = test_inputs.get('per_atom_weights', False)

    dataset = storage.get_data(test_inputs['dataset_handle'])

    import numpy as np
    natoms = [len(atoms) for atoms in dataset]
    per_atom_weights = np.zeros(sum(natoms))
    per_atom_weights = np.array_split(per_atom_weights, np.cumsum(natoms)[:-1])

    test_inputs['training_args']['per_atom_weights'] = per_atom_weights

    _, training_loss = potential.train(test_inputs['dataset_handle'], storage,
                                       scheduler,
                                       **test_inputs['training_args'])
    return True, potential


def potential_submit_train_test(input_file: str) -> bool:
    """
    test of the submit_train functionality of trainer

    :param input_file: input file path with requisite module blocks.
        potential, trainer, storage, and scheduler are required
    :type input_file: str
    :returns: boolean flag that the function completed execution. Does not
        necessarily indicate a correct output, but is used to determine if the
        pytest should be run
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    potential = init_and_validate_module_type('potential', test_inputs)
    storage = init_and_validate_module_type('storage', test_inputs)
    scheduler = init_and_validate_module_type('scheduler', test_inputs)

    # Add some keys if they don't exist so we can do **test_inputs
    test_inputs['training_args']['job_details'] = test_inputs[
        'training_args'].get('job_details', {})

    per_atom_weights = test_inputs['training_args'].get('per_atom_weights')
    if type(per_atom_weights) is str:
        if not os.path.isabs(per_atom_weights):
            cwd = os.path.abspath('.')
            abs_per_atom_weights = f'{cwd}/{per_atom_weights}'
            # overwrite with the abs path
            test_inputs['training_args'][
                'per_atom_weights'] = abs_per_atom_weights

    calc_id = potential.submit_train(test_inputs['dataset_handle'], storage,
                                     scheduler, **test_inputs['training_args'])
    print(f'Training job submitted as {calc_id}')

    potential.load_from_submitted_training(calc_id, scheduler)

    return True, potential


def potential_kimkit_combined_test(input_file: str) -> bool:
    """
    test of kimkit storing functionality of Potentials

    :param input_file: path to input file for the given potential type
    :type potential_type: str
    :returns: boolean flag that the function completed execution. Does not
        necessarily indicate a correct output, but is used to determine if the
        pytest should be run
    :rtype: bool
    """

    from orchestrator.potential.kim_mixins import (enumerate_kim_repository,
                                                   delete_potential_from_kim)

    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    # print("Instantiating from existing files")
    potential1 = init_and_validate_module_type('potential', test_inputs)

    first_kim_id = potential1.save_potential_to_kim(
        'A dummy model generated during testing.')

    # print(f"Saved instantiated model under new KIM ID: {first_kim_id}")
    assert first_kim_id is not None

    # print("Loading existing KIM model")
    potential_args = {
        "potential_type": test_inputs['potential']['potential_type'],
        "potential_args": {
            "initialize_from": "kim_id",
            "kim_id": first_kim_id,
        }
    }
    potential2 = init_and_validate_module_type('potential',
                                               potential_args,
                                               single_input_dict=True)

    # Make sure they have the same kim id and hyperparameters
    assert potential2.kim_id == potential1.kim_id

    if is_dataclass(potential1.hyperparameters):
        hparams1 = asdict(potential1.hyperparameters)
        hparams2 = asdict(potential2.hyperparameters)
    elif isinstance(potential1.hyperparameters, dict):
        hparams1 = potential1.hyperparameters
        hparams2 = potential2.hyperparameters
    else:
        raise TypeError('hyperparameters are of unexpected type')

    for k, v in hparams2.items():
        assert hparams1[k] == v

    # print("Saving to see if version update works")
    versioned_kim_id = potential2.save_potential_to_kim(
        'Testing version updates.')

    # Make sure the IDs are the same, but the version has been incremented
    assert versioned_kim_id[:-1] == first_kim_id[:-1]
    assert int(versioned_kim_id[-1]) == int(first_kim_id[-1]) + 1

    # print("Re-loading and saving again to see if forking works")
    potential_args = {
        "potential_type": test_inputs['potential']['potential_type'],
        "potential_args": {
            "initialize_from": "kim_id",
            "kim_id": first_kim_id,
        }
    }
    potential3 = init_and_validate_module_type('potential',
                                               potential_args,
                                               single_input_dict=True)
    forked_kim_id = potential3.save_potential_to_kim(
        'Testing forking updates.')

    assert forked_kim_id[:-1] != first_kim_id[:-1]

    # print("Checking repository status")
    repository_contents = enumerate_kim_repository()

    assert first_kim_id in repository_contents
    assert versioned_kim_id in repository_contents
    assert forked_kim_id in repository_contents

    # print("Deleting stuff generated by this test")
    delete_potential_from_kim(versioned_kim_id)
    delete_potential_from_kim(forked_kim_id)

    # Note: you have to be careful about deletion order because kimkit won't
    # let you delete things that have existing newer/forked versions
    delete_potential_from_kim(first_kim_id)

    # print("Confirming that we deleted things properly")
    repository_contents = enumerate_kim_repository()

    assert first_kim_id not in repository_contents
    assert versioned_kim_id not in repository_contents
    assert forked_kim_id not in repository_contents

    return True, potential1
