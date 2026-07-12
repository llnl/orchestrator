from os import system, path
import numpy as np
from typing import Optional, Union
from ..storage.storage_base import Storage
from ..potential.potential_base import Potential
from ..scheduler.scheduler_base import Scheduler
from ase import Atoms
from kliff.dataset.dataset import DatasetError  # temporary
from .trainer_base import Trainer
from fitsnap3lib.scrapers.ase_funcs import get_apre
from ..utils.data_standard import (
    ENERGY_KEY,
    FORCES_KEY,
    STRESS_KEY,
    FORCES_WEIGHTS_KEY,
)


class FitSnapTrainer(Trainer):
    """
    Train and deploy a potential using FitSnap

    The trainer class is responsible for handling the loading/assignment of
    training data, as well as the actual process of training a potential.
    This trainer is intended to be used with Snap model trained with ASE
    training data.
    """

    def __init__(self, **kwargs):
        """
        Train and deploy a general parametric model potential using FitSnap
        """

        super().__init__(**kwargs)

        # arguments to reinitialize an instance of the trainer
        self.trainer_init_args = kwargs

    def checkpoint_trainer(self):
        """
        checkpoint the trainer module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        pass

    def restart_trainer(self):
        """
        restart the trainer module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        pass

    def _get_training_data(
        self,
        dataset_handle: str,
        storage: Storage,
    ) -> list[Atoms]:
        """
        Get the training data configurations

        Retrieve the dataset specified by dataset_handle from the passed
        storage module.

        :param dataset_handle: the identifier of the dataset to extract from
            the storage module
        :type dataset_handle: str
        :param storage: storage instance where the training data is saved
        :type storage: Storage
        :returns: training data of configurations
        :rtype: ASE Dataset
        """
        self.logger.info('Reading training data from storage')

        try:
            training_set = storage.get_data(dataset_handle)
        except DatasetError:
            print(('Storage module is not properly set. Cannot '
                   'get training data from Storage.'))
            exit()
        for c in training_set:
            try:
                c.info[ENERGY_KEY] = c.get_potential_energy()
            except Exception:
                pass
            try:
                c.info[STRESS_KEY] = c.get_stress()
            except Exception:
                pass
            try:
                c.set_array(FORCES_KEY, c.get_forces())
            except Exception:
                pass
        return training_set

    def _collect_weights(
        self,
        atoms: Atoms,
    ) -> np.ndarray:
        """
        Function to collect per-atom weight data from ASE atoms objects.

        :param atoms: ASE atoms object for a single configuration of atoms.
        :type atoms: Atoms
        :returns: a weight np array for a single configuration.
        :rtype: np.ndarray
        """

        return atoms.info[FORCES_WEIGHTS_KEY]

    def _convert_to_3x3_stress_tensor(self,
                                      stress_vector: np.ndarray) -> np.ndarray:
        """
        Helper function to convert the (6,) stress vector to 3x3 expected by
          FitSNAP

        :param stress_vector: 6 stress components (Voigt notation)
        :type stress_vector: np.ndarray
        :returns: transformed matrix in full 3x3 format
        :rtype: np.ndarray
        """
        return np.array([
            [stress_vector[0], stress_vector[5], stress_vector[4]],
            [stress_vector[5], stress_vector[1], stress_vector[3]],
            [stress_vector[4], stress_vector[3], stress_vector[2]],
        ])

    def _collate_fitsnap_data(
        self,
        atoms: Atoms,
        eweight: float,
        fweight: float,
        vweight: float,
    ) -> dict:
        """
        Function to organize fitting data for FitSNAP from ASE atoms objects.

        Args:
        atoms: ASE atoms object for a single configuration of atoms.

        Returns data dictionary in FitSNAP format for a single configuration.
        """

        # Transform ASE cell to be appropriate for LAMMPS.
        apre = get_apre(cell=atoms.cell)
        r = np.dot(np.linalg.inv(atoms.cell), apre)
        positions = np.matmul(atoms.get_positions(), r)
        cell = apre.T

        # Make a data dictionary for this config.

        data = {}
        data['Group'] = None
        data['File'] = None
        data['Stress'] = np.array(atoms.info[STRESS_KEY])
        if data['Stress'].shape[0] == 6:
            data['Stress'] = self._convert_to_3x3_stress_tensor(data['Stress'])
        elif data['Stress'].shape != (3, 3):
            raise ValueError('Stress tensor not supplied as 6, or 3x3 formats')
        data['Positions'] = positions
        data['Energy'] = atoms.info[ENERGY_KEY]
        data['AtomTypes'] = atoms.get_chemical_symbols()
        data['NumAtoms'] = len(atoms)
        data['Forces'] = atoms.arrays[FORCES_KEY]
        data['QMLattice'] = cell
        data['test_bool'] = 0
        data['Lattice'] = cell
        data['Rotation'] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        data['Translation'] = np.zeros((len(atoms), 3))
        # Inject the weights.
        data['eweight'] = eweight
        data['fweight'] = fweight
        data['vweight'] = vweight

        return data

    def _write_training_script(
        self,
        save_path: str,
        dataset_list: list,
        potential: Potential,
        storage: Storage,
        eweight: float = 1.0,
        fweight: float = 1.0,
        vweight: float = 1.0,
        per_atom_weights: Union[bool, np.ndarray, str] = False,
        upload_to_kimkit=True,
    ) -> str:
        """
        Write a script to run the trainer outside of memory

        This is a helper function for generating a script, training_script.py,
        which can be executed via a scheduler or offline. It additionally saves
        needed additional files with it, such as a weights.txt data file.

        :param save_path: path where the training script will be written
        :type save_path: str
        :param dataset_list: list of dataset handles which should be used for
            the training procedure
        :type dataset_list: list of str
        :param potential: Potential instance to be trained, expect its
            pre-trained state to be written to save_path/potential_to_train.pkl
        :type potential: Potential
        :param storage: an instance of the storage class, which contains the
            datasets in dataset_list
        :type storage: Storage
        :returns: name of the script that is generated (training_script.py)
        :param eweight: weight of energy data in the loss function
        :type eweight: float
        :param fweight: weight of the force data in the loss function
        :type fweight: float
        :param vweight: weight of the stress data in the loss function
        :type vweight: float
        :param per_atom_weights: True to read from dataset, or numpy array,
            or a str for a numpy.loadtxt compatible filepath
            |default| ``False``
        :type per_atom_weights: either boolean or np.ndarray
        :param upload_to_kimkit: True to upload to kimkit repository
        :type upload_to_kimkit: bool
        :returns: the name of the execution script
        :rtype: str
        """
        full_save_path = path.abspath(save_path)
        import_lines = ('from orchestrator.utils.setup_input import '
                        'init_and_validate_module_type\n')
        trainer_dict = {
            'trainer_type': self.factory_token,
            'trainer_args': self.trainer_init_args
        }
        init_trainer = ('trainer = init_and_validate_module_type("trainer", '
                        f'{trainer_dict}, single_input_dict=True)')

        storage_dict = {
            'storage_type': storage.factory_token,
            'storage_args': storage.storage_init_args
        }
        init_storage = ('storage = init_and_validate_module_type("storage", '
                        f'{storage_dict}, single_input_dict=True)')

        potential_dict = {
            'potential_type': potential.factory_token,
            'potential_args': potential.trainer_args
        }
        root_input = potential_dict['potential_args']['settings_path']
        abs_input = path.abspath(root_input)
        potential_dict['potential_args']['settings_path'] = abs_input
        init_potential = ('potential = init_and_validate_module_type('
                          f'"potential", {potential_dict}, '
                          'single_input_dict=True)\n')

        load_potential = "potential.build_potential()"

        # per_atom_fit can be boolean, np.ndarray (or list), or str
        # if list, convert to a np.ndarray first
        if type(per_atom_weights) is list:
            per_atom_weights = np.array(per_atom_weights)

        # if bool, just pass along. The info is in storage object
        if type(per_atom_weights) is bool:
            per_atom_weights_for_script = per_atom_weights
        # if str, collect file; must be a viable np.loadtxt() file path
        # viability not checked during this step
        elif type(per_atom_weights) is str:
            standard_weights_path = path.abspath(
                f'{full_save_path}/weights.txt')
            # if located elsewhere, collect to current directory
            current_abs_path = path.abspath(per_atom_weights)
            if current_abs_path != standard_weights_path:
                system(f'cp {current_abs_path} {standard_weights_path}')
            per_atom_weights_for_script = "weights.txt"
        # if np.array/list, save to the training directory and pass str
        elif type(per_atom_weights) is np.ndarray:
            np.savetxt(f'{full_save_path}/weights.txt', per_atom_weights)
            per_atom_weights_for_script = "weights.txt"
        else:
            raise TypeError('per_atom_weights not a supported type!')

        # Currently uses the scheduler from trainer, not submit_train's input
        construct_and_train = (
            f'snap, errors = trainer.train(path_type="{full_save_path}",'
            f'potential=potential,'
            f'storage=storage,'
            f'dataset_list={dataset_list},'
            f'eweight={eweight},'
            f'fweight={fweight},'
            f'vweight={vweight},'
            'write_training_script=False,')
        if type(per_atom_weights_for_script) is str:
            construct_and_train += (
                f'per_atom_weights="{per_atom_weights_for_script}",')
        else:
            construct_and_train += (
                f'per_atom_weights={per_atom_weights_for_script},')
        construct_and_train += (f'upload_to_kimkit={upload_to_kimkit})')

        script = '\n'.join([
            import_lines,
            init_trainer,
            init_storage,
            init_potential,
            load_potential,
            construct_and_train,
        ])
        with open(f'{full_save_path}/training_script.py', 'w') as fout:
            fout.write(script)

        return 'training_script.py'

    def train(
        self,
        path_type: str,
        potential: Potential,
        storage: Storage,
        dataset_list: list,
        scheduler: Optional[Scheduler] = None,
        eweight: float = 1.0,
        fweight: float = 1.0,
        vweight: float = 1.0,
        per_atom_weights: Union[bool, np.ndarray, str] = False,
        write_training_script: bool = True,
        upload_to_kimkit=True,
    ) -> list:
        """
        Train a Snap potential using FitSnap

        This is the main method of the trainer class, and uses the parameters
        supplied in the FitSnap settings file to perform the potential training

        :param path_type: if write_training_script=True, specifier for the
            scheduler path, to differentiate training runs; else, the raw
            path to save files
        :type path_type: str
        :param potential: :class:`~orchestrator.potential.fitsnap.
            FitSnapPotential` class object containing fitsnap instance
        :type potential: fitsnap instance
        :param storage: an instance of the storage class
        :type storage: Storage
        :dataset_list: the list of dataset_handles (e.g. collabfit-IDs)
            within the storage object to use as the dataset.
        :type dataset_list: list
        :param scheduler: the scheduler for managing path definition and job
            submission, if none are supplied, will use the default scheduler
            defined in this class |default| ``None``
        :type scheduler: Scheduler
        :param eweight: weight of energy data in the loss function
        :type eweight: float
        :param fweight: weight of the force data in the loss function
        :type fweight: float
        :param vweight: weight of the stress data in the loss function
        :type vweight: float
        :param per_atom_weights: True to read from dataset, or numpy array,
            or a str for a numpy.loadtxt compatible filepath
            |default| ``False``
        :type per_atom_weights: either boolean or np.ndarray
        :param write_training_script: True to write a training script in the
            scheduler created directory |default| ``True``;
            This is expected to always be left on if not being called by a
            submit_train() scheduler!
        :type write_training_script: bool
        :param upload_to_kimkit: True to upload to kimkit repository
        :type upload_to_kimkit: bool
        :returns: trained model, error metrics
        :rtype: fitsnap instance, fitsnap error attribute
        """
        # reset parameter_path for new training
        potential.parameter_path = None

        if dataset_list is None or storage is None:
            raise ValueError('A storage object and list of dataset handles'
                             ' are required!')

        if not isinstance(dataset_list, list):
            dataset_list = [dataset_list]
        combined_dataset = []
        for dataset_handle in dataset_list:
            configs = self._get_training_data(dataset_handle, storage)
            combined_dataset.extend(configs)

        snap = potential.model

        snap.data = [
            self._collate_fitsnap_data(atoms, eweight, fweight, vweight)
            for atoms in combined_dataset
        ]
        self.logger.info(f"Found {len(snap.data)} configurations")

        # for tracking files to upload to kimkit later
        include_weights_file = False

        # per_atom_fit can be boolean, np.array (or list), or str
        # data (numpy array/list) be used directly
        # filepath (str) will numpy.loadtxt() the data
        # True will load data that exists in storage for the dataset
        if type(per_atom_weights) is bool:
            if per_atom_weights:
                per_atom_fit = True
                weights_lists = [
                    self._collect_weights(atoms) for atoms in combined_dataset
                ]
                weights = np.array(
                    [elem for list in weights_lists for elem in list])
            else:
                per_atom_fit = False
        elif type(per_atom_weights) is str:
            per_atom_fit = True
            weights = np.loadtxt(per_atom_weights)
            if per_atom_weights == 'weights.txt':
                include_weights_file = True
        elif type(per_atom_weights) is list:
            per_atom_fit = True
            weights = np.array(per_atom_weights)
        elif type(per_atom_weights) is np.ndarray:
            per_atom_fit = True
            weights = per_atom_weights
        else:
            raise TypeError('per_atom_weights not a supported type!')

        # Calculate descriptors for all configurations
        snap.process_configs()

        # if weighted fitting activated by data input or True boolean,
        # overwrite the weight matrix with a custom-defined one
        if per_atom_fit:
            row_types = snap.pt.fitsnap_dict['Row_Type']
            manually_created_w_array = np.zeros(
                len(snap.pt.shared_arrays['w'].array))
            force_rows = [
                True if row == 'Force' else False for row in row_types
            ]
            assert (len(weights) * 3) == sum(force_rows), \
                f"{len(weights)} weights given, need {sum(force_rows) / 3}"
            energy_rows = [
                True if row == 'Energy' else False for row in row_types
            ]
            stress_rows = [
                True if row == 'Stress' else False for row in row_types
            ]
            manually_created_w_array[
                energy_rows] = eweight  # all identical currently
            manually_created_w_array[force_rows] = fweight * \
                np.array([val for val in weights.tolist() for i in range(3)])
            manually_created_w_array[
                stress_rows] = vweight  # all identical currently
            snap.pt.shared_arrays['w'].array = manually_created_w_array

        # Perform the fit
        snap.solver.perform_fit()

        # Analyze error metrics
        snap.solver.error_analysis()

        # This should be superfluous
        potential.model = snap

        # write equivalent training script for documentation
        # the only time this should NOT be the case is when train is being
        # called from a training_script
        if write_training_script:
            # for normal training we need to make a path to save to
            if scheduler is None:
                scheduler = self.default_scheduler
            save_path = scheduler.make_path(self.__class__.__name__, path_type)

            # Output the weights into a datafile if needed
            # If True/False, can just assume storage holding weights is enough
            if type(per_atom_weights) is bool:
                per_atom_weights_for_script = per_atom_weights
            # Otherwise, pre-emptively save the weights with a standard name,
            # just pass the filename to _write_training_script.
            elif type(per_atom_weights) in [str, list, np.ndarray]:
                np.savetxt(f'{save_path}/weights.txt', weights)
                per_atom_weights_for_script = f"{save_path}/weights.txt"
            else:
                raise TypeError(
                    'per_atom_weights: How did this not TypeError earlier?')
            if per_atom_weights_for_script == 'weights.txt':
                include_weights_file = True

            # Output a training script equivalent to what was performed
            _ = self._write_training_script(
                save_path,
                dataset_list,
                potential,
                storage,
                eweight,
                fweight,
                vweight,
                per_atom_weights_for_script,
            )

        # just use the path_type raw as the location to save the files
        else:
            save_path = path_type

        # Finally output the model files
        _ = self._save_model(
            save_path,
            potential,
            potential_name='fitsnap_potential',
            loss=snap.solver.errors,
            create_path=False,
            scheduler=scheduler,
        )

        # TODO: allow specifying auxiliary files to attach to upload?
        if upload_to_kimkit:
            training_files = [f'{save_path}/training_script.py']
            if include_weights_file is True:
                training_files.append(f'{save_path}/weights.txt')
            _ = potential.save_potential_files(work_dir=save_path,
                                               training_files=training_files,
                                               import_to_kimkit=True,
                                               write_to_tmp_dir=False)

        return snap, snap.solver.errors

    def submit_train(
        self,
        path_type: str,
        potential: Potential,
        storage: Storage,
        dataset_list: list,
        scheduler: Scheduler,
        job_details: dict,
        eweight: float = 1.0,
        fweight: float = 1.0,
        vweight: float = 1.0,
        per_atom_weights: Union[bool, np.ndarray, str] = False,
        upload_to_kimkit=True,
    ) -> int:
        """
        Asychronously train the potential based on the trainer details

        This is a main method of the trainer class, and uses the parameters
        supplied at instantiation to perform the potential training by
        minimizing a loss function. While :meth:`train` works synchronously,
        this method submits training to a job scheduler.

        :param path_type: specifier for the scheduler path, to differentiate
            training runs
        :type path_type: str
        :param potential: potential to be trained. The actual model itself is
            set as an attribute of the Potential object
        :type potential: Potential
        :param storage: an instance of the storage class
        :type storage: Storage
        :dataset_list: the list of dataset_handles (e.g. collabfit-IDs)
            within the storage object to use as the dataset.
        :type dataset_list: list
        :param scheduler: the scheduler for managing path definition and job
            submission, if none are supplied, will use the default scheduler
            defined in this class
        :type scheduler: Scheduler
        :param job_details: job parameters such as walltime or # of nodes
        :type job_details: dict
        :param eweight: weight of energy data in the loss function
        :type eweight: float
        :param fweight: weight of the force data in the loss function
        :type fweight: float
        :param vweight: weight of the stress data in the loss function
        :type vweight: float
        :param per_atom_weights: True to read from dataset, or numpy array,
            or a str for a numpy.loadtxt compatible filepath
            |default| ``False``
        :type per_atom_weights: either boolean or np.ndarray
        :param upload_to_kimkit: True to upload to kimkit repository
        :type upload_to_kimkit: bool
        :returns: calculation ID of the submitted job
        :rtype: int
        """
        # reset parameter_path for new training
        potential.parameter_path = None
        potential.trainer_args['parameter_path'] = None

        if dataset_list is None or storage is None:
            raise ValueError('A storage object and list of dataset handles'
                             ' are required!')

        if not isinstance(dataset_list, list):
            dataset_list = [dataset_list]
        save_path = scheduler.make_path(self.__class__.__name__,
                                        f'{path_type}')
        script_filename = self._write_training_script(
            save_path,
            dataset_list,
            potential,
            storage,
            eweight,
            fweight,
            vweight,
            per_atom_weights=per_atom_weights,
            upload_to_kimkit=upload_to_kimkit)
        job_details['custom_preamble'] = 'python'
        calc_id = scheduler.submit_job(
            script_filename,
            save_path,
            job_details=job_details,
        )
        return calc_id

    def _save_model(
        self,
        path_type: str,
        potential: Potential,
        potential_name: str = 'fitsnap_potential',
        loss: Optional[None] = None,
        create_path: bool = True,
        scheduler: Optional[Scheduler] = None,
    ) -> str:
        """
        Output FitSnap model files. Write error metric and LAMMPS input files

        If the potential.parameter_path is not already set, this writes the
        model coefficient and parameter (and error summary) files to disk
        from memory using the FitSnap infrastructure, then sets
        potential.parameter_path to the file location. If the parameter_path
        is already set, copies the files to the new location and updates
        the potential.parameter_path.

        :param path_type: specifier for the scheduler path, to differentiate
            training runs and where the model will be saved
        :type path_type: str
        :param potential: potential to be saved
        :type potential: Snap potential
        :param potential_name: name to save the potential as
            |default| 'fitsnap_potential'
        :type potential_name: str
        :param loss: FitSNAP error object; this can but probably should not
            be supplied by the user
        :type loss: FitSNAP error
        :param create_path: if the function needs to create a new path, or if
            path_type should be used as the full path |default| ``True``
        :type create_path: boolean
        :param scheduler: the scheduler for managing path definition, if none
            are supplied, will use the default scheduler defined in this class
            |default| ``None``
        :type scheduler: Scheduler
        :returns: path where the model is saved (inclusive)
        :rtype: str
        """
        if scheduler is None:
            scheduler = self.default_scheduler
        if create_path:
            save_path = scheduler.make_path(self.__class__.__name__, path_type)
        else:
            save_path = path_type

        if potential.parameter_path is not None:
            # potential._write_potential_to_file(save_path)
            _ = potential.save_potential_files(work_dir=save_path,
                                               import_to_kimkit=False,
                                               write_to_tmp_dir=False)
            return potential.parameter_path
        else:  # first save after a local train()

            self.logger.info(f'Saving model state in {save_path}')
            snap = potential.model
            vars(snap.config.sections['OUTFILE'])['potential_name'] = \
                save_path + '/' + potential_name
            vars(snap.config.sections['OUTFILE'])['metric_file'] = \
                save_path + '/' + potential_name + '.md'

            fit_coefficients = snap.solver.fit
            if loss is None:  # loss should probably not be supplied by user
                errors = snap.solver.errors
            else:
                errors = loss
            snap.output.output(fit_coefficients, errors)

            # for compatibility with older versions of LAMMPS and KIM drivers
            system('sed -i "s/switchinnerflag 0/# switchinnerflag 0 commented '
                   f'by orchestrator/" {save_path}/{potential_name}.snapparam')

            # for compatibility with KIM model driver SNAP__MD_536750310735_000
            # possibility of training a simulator model and reloading a
            # portable model that can't handle these settings flags, but
            # I think build_potential() should catch the settings issue
            if potential.kim_item_type == "portable-model":
                if potential.model_driver == "SNAP__MD_536750310735_000":
                    system(
                        'sed -i "s/wselfallflag 0/# wselfallflag 0 commented '
                        'by orchestrator/"'
                        f' {save_path}/{potential_name}.snapparam')
                    system('sed -i "s/chemflag 0/# chemflag 0 commented '
                           'by orchestrator/"'
                           f' {save_path}/{potential_name}.snapparam')
                    system('sed -i "s/bnormflag 0/# bnormflag 0 commented '
                           'by orchestrator/"'
                           f' {save_path}/{potential_name}.snapparam')

            potential.parameter_path = f'{save_path}/{potential_name}'
            potential.training_hash = snap.config.hash
            self.logger.info(f'Output fitsnap files with Hash: '
                             f'{potential.training_hash} at location: '
                             f'{potential.parameter_path}')

            return f'{save_path}/{potential_name}'

    def load_from_submitted_training(
        self,
        calc_id: int,
        potential: Potential,
        scheduler: Scheduler,
    ):
        """
        reload a potential that was trained via a submitted job

        :param calc_id: calculation ID of the submitted training job
        :type calc_id: int
        :param potential: :class:`~orchestrator.potential.dnn.KliffBPPotential`
            class object that will be updated with the model saved to disk
            after the training job.
        :type potential: KliffBPPotential
        :param scheduler: the scheduler for managing path definition and job
            submission, if none are supplied, will use the default scheduler
            defined in this class |default| ``None``
        :type scheduler: Scheduler
        """
        scheduler.block_until_completed(calc_id)

        if potential.name is not None:
            potential_name = potential.name
        else:
            potential_name = "fitsnap_potential"
        parameter_path = scheduler.get_job_path(calc_id) + '/' + potential_name
        potential.parameter_path = parameter_path
        self.logger.info(f'Loading potential from: {parameter_path}')
