import json
from numpy import loadtxt
from orchestrator.utils.setup_input import init_and_validate_module_type


def trainer_potential_combined_test(input_file: str) -> bool:
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
    import os

    if '/' in input_file:
        input_directory = '/'.join(input_file.split('/')[:-1])
    else:
        input_directory = '.'
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    potential = init_and_validate_module_type('potential', test_inputs)
    trainer = init_and_validate_module_type('trainer', test_inputs)
    storage = init_and_validate_module_type('storage', test_inputs)

    # delete the temp dir in case previous runs
    try:
        os.rmdir(test_inputs.get('path_type'))
    except FileNotFoundError:
        pass

    potential.build_potential()
    # storage module needs to already possess the dataset... TODO
    # could add here as part of unit test, but probably better to keep separate

    eweight = test_inputs['trainer']['trainer_args'].get('eweight', 1.0)
    fweight = test_inputs['trainer']['trainer_args'].get('fweight', 1.0)
    vweight = test_inputs['trainer']['trainer_args'].get('vweight', 1.0)

    per_atom_weights = test_inputs['trainer'].get('per_atom_weights', False)
    if type(per_atom_weights) is str:
        per_atom_weights = loadtxt(input_directory + '/' + per_atom_weights)

    # # change kim_id to avoid conflicts with system potentials
    # if potential.kim_id == "SW_StillingerWeber_1985_Si__MO_405512056662_006":
    if potential.model_type != 'dnn':
        potential.generate_new_kim_id(id_prefix="Test_KIMPotential")
    else:
        potential.generate_new_kim_id(id_prefix="Test_KIMPotential")

    _, training_loss = trainer.train(
        test_inputs.get('path_type'),
        potential,
        storage,
        test_inputs['dataset_handle'],
        eweight=eweight,
        fweight=fweight,
        vweight=vweight,
        per_atom_weights=per_atom_weights,
        upload_to_kimkit=False,
    )

    return True, potential


def trainer_potential_scheduler_test(input_file: str) -> bool:
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
    if '/' in input_file:
        input_directory = '/'.join(input_file.split('/')[:-1])
    else:
        input_directory = '.'
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    potential = init_and_validate_module_type('potential', test_inputs)
    trainer = init_and_validate_module_type('trainer', test_inputs)
    storage = init_and_validate_module_type('storage', test_inputs)
    scheduler = init_and_validate_module_type('scheduler', test_inputs)

    potential.build_potential()

    eweight = test_inputs['trainer']['trainer_args'].get('eweight', 1.0)
    fweight = test_inputs['trainer']['trainer_args'].get('fweight', 1.0)
    vweight = test_inputs['trainer']['trainer_args'].get('vweight', 1.0)

    per_atom_weights = test_inputs['trainer'].get('per_atom_weights', False)
    if type(per_atom_weights) is str:
        per_atom_weights = loadtxt(input_directory + '/' + per_atom_weights)

    calc_id = trainer.submit_train(
        test_inputs.get('path_type'),
        potential,
        storage,
        test_inputs['dataset_handle'],
        scheduler,
        test_inputs.get('job_details', {}),
        eweight=eweight,
        fweight=fweight,
        vweight=vweight,
        per_atom_weights=per_atom_weights,
        upload_to_kimkit=False,
    )
    print(f'Training job submitted as {calc_id}')

    trainer.load_from_submitted_training(calc_id, potential, scheduler)
    return True


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
    try:
        import tarfile
        import os
        import tempfile
        from kimkit import models
        from kimkit.src import config as cf
    except ModuleNotFoundError:
        print("It doesn't look like kimkit and/or kliff is installed in your "
              'environment, skipping the potential_kimkit_integration_test!')

    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    has_driver = test_inputs.get("has_driver")
    previous_driver_kimcode = test_inputs.get("previous_driver_kimcode")
    driver_test_kimcode = test_inputs.get("driver_test_kimcode")
    example_kimcode_prefix = test_inputs.get("example_kimcode_prefix")
    driver_md = test_inputs.get("driver_md")
    trainer_test_input_file = test_inputs.get("trainer_test_input_file")

    here = os.path.dirname(os.path.realpath(__file__))
    # move up one level
    here = os.path.dirname(here)
    with tempfile.TemporaryDirectory() as temporary_directory:
        print(f'Files stored in temporary directory {temporary_directory}')

        delete_files = []

        try:
            if has_driver:
                driver_tarfile = os.path.join(here, "shared_inputs/potential/",
                                              previous_driver_kimcode + ".txz")
                driver_tar = tarfile.open(driver_tarfile)

                try:
                    # import the driver needed for the model
                    models.import_item(
                        tarfile_obj=driver_tar,
                        metadata_dict=driver_md,
                        previous_item_name=previous_driver_kimcode)

                except cf.KimCodeAlreadyInUseError:
                    models.delete(driver_test_kimcode)
                    models.import_item(
                        tarfile_obj=driver_tar,
                        metadata_dict=driver_md,
                        previous_item_name=previous_driver_kimcode)

            # now begin the actual test of the potential module
            # (or at least the parts of it that interface with kimkit)

            __, potential = trainer_potential_combined_test(
                trainer_test_input_file)

            # these potentials don't already have a kimcode
            # supply human-readable prefix and one will be assigned
            # try:
            if potential.model_type == "fitsnap":
                potential.generate_new_kim_id(id_prefix=example_kimcode_prefix)
                try:
                    example_kimcode = potential.save_potential_files(
                        kim_id=potential.kim_id,
                        model_driver=driver_test_kimcode)

                except cf.KimCodeAlreadyInUseError:
                    models.delete(example_kimcode)

                    example_kimcode = potential.save_potential_files(
                        kim_id=potential.kim_id,
                        model_driver=driver_test_kimcode)
            # except AttributeError:
            elif potential.model_type == "dnn":
                potential.model_driver = 'no-driver'
                potential.generate_new_kim_id(id_prefix=example_kimcode_prefix)
                try:
                    example_kimcode = potential.save_potential_files(
                        model_name_prefix=example_kimcode_prefix)

                except cf.KimCodeAlreadyInUseError:
                    models.delete(example_kimcode)

                    example_kimcode = potential.save_potential_files(
                        model_name_prefix=example_kimcode_prefix)

            else:
                potential.generate_new_kim_id(id_prefix=example_kimcode_prefix)
                try:

                    example_kimcode = potential.save_potential_files(
                        model_name_prefix=example_kimcode_prefix,
                        model_driver=driver_test_kimcode)

                except cf.KimCodeAlreadyInUseError:
                    models.delete(example_kimcode)

                    example_kimcode = potential.save_potential_files(
                        model_name_prefix=example_kimcode_prefix,
                        model_driver=driver_test_kimcode)

            potential.get_potential_files(destination_path=".",
                                          kim_id=example_kimcode)

            # delete the model/driver after the test suceeds (hopefully)
            models.delete(example_kimcode)
            if driver_test_kimcode != "no-driver":
                models.delete(driver_test_kimcode)

        except Exception as e:
            try:
                models.delete(example_kimcode)
            except UnboundLocalError:
                pass
            if driver_test_kimcode != "no-driver":
                models.delete(driver_test_kimcode)
            raise e

        else:
            print("potential/kimkit integration test passed.")

        finally:

            for file in delete_files:
                os.remove(file)

    return True


def potential_kim_api_integration_test(input_file: str) -> bool:
    """
    Test of the potential module's integration with the KIM_API

    Test attempts to train a new KIM potential, install it into
    the KIM_API, run a trivial LAMMPS simulation, and remove
    it from the KIM_API.

    :param input_file: path to the input file for the test
    :type input_file: str
    :rtype: bool
    """

    try:
        import os
        import tempfile
        from ase.lattice.cubic import FaceCenteredCubic
        from orchestrator.simulator import LAMMPSSimulator
    except ModuleNotFoundError:
        print("It doesn't look like kimkit and/or kliff is installed in your "
              'environment, skipping the potential_kimkit_integration_test!')

    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    example_kimcode_prefix = test_inputs.get("example_kimcode_prefix")

    model_driver = test_inputs.get("model_driver")
    trainer_test_input_file = test_inputs.get("trainer_test_input_file")

    lammps_executable = test_inputs.get("lammps_executable")

    with tempfile.TemporaryDirectory() as temporary_directory:
        tmp_dest_path = os.path.join(temporary_directory,
                                     "shared_inputs/potential/tmp/")

        delete_files = []

        __, potential = trainer_potential_combined_test(
            trainer_test_input_file)

        potential.generate_new_kim_id(example_kimcode_prefix)
        if potential.kim_item_type == 'portable-model':
            potential.model_driver = model_driver
        else:
            potential.model_driver = None

        try:
            atoms = FaceCenteredCubic(symbol=potential.species[0],
                                      latticeconstant=4.0,
                                      size=(1, 1, 1))

            # test installing in KIM_API
            potential.install_potential_in_kim_api(
                potential_name=potential.kim_id,
                install_locality="system",
                save_path=tmp_dest_path,
                import_into_kimkit=True)

            try:
                # test potential.evaluate() method
                forces, energy, stress = potential.evaluate(atoms)

                print(forces, energy, stress)

            except RuntimeError as e:
                # remove from KIM_API
                result = os.system(
                    f'{potential.kim_api} remove --force {potential.kim_id};'
                    ' cd / 1> /dev/null')
                if result == 0:
                    print('Potential removed from user collection')
                else:
                    raise RuntimeError(
                        "Could not remove potential from KIM_API")

                raise RuntimeError from e

            # test running a trivial LAMMPS simulation
            sim_init = {
                "code_path": lammps_executable,
                "elements": potential.species,
                "input_template": "./test_inputs/api_test.lammps"
            }
            lammps_sim = LAMMPSSimulator(sim_init)
            sim_template_fill = {
                'model_name': potential.kim_id,
                'species': ' '.join(potential.species),
            }
            calc_id = lammps_sim.run('api_test', None, sim_template_fill)
            print('LAMMPS simulation run in '
                  f'{lammps_sim.default_scheduler.get_job_path(calc_id)}')

            # remove from KIM_API
            result = os.system(
                f'{potential.kim_api} remove --force {potential.kim_id};'
                ' cd / 1> /dev/null')
            if result == 0:
                print('Potential removed from user collection')
            else:
                raise RuntimeError("Could not remove potential from KIM_API")

        except Exception as e:
            raise e

        finally:

            for file in delete_files:
                os.remove(file)

    return True
