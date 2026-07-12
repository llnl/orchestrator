from kliff.dataset import Dataset
from kliff.legacy.loss import Loss
from kliff.legacy.calculators import CalculatorTorch
from os import path
from copy import deepcopy
from typing import Optional, Union
import numpy as np
from ...potential.potential_base import Potential
from ...scheduler.scheduler_base import Scheduler
from .kliff import KLIFFTrainer
from ...utils.data_standard import FORCES_KEY, ENERGY_KEY

from typing import TYPE_CHECKING
if TYPE_CHECKING:  # avoid circular imports
    from orchestrator.storage.storage_base import Storage


class DUNNTrainer(KLIFFTrainer):
    """
    Train and deploy a fully connected neural network based on Behler-
    Parrinello symmetry functions. This trainer uses the KIM DUNN driver
    for deploying the potential which has higher performance C++ backend
    and inbuilt support for UQ.

    The trainer class is responsible for handling the loading/assignment of
    training data, as well as the actual process of training a potential. This
    trainer is intended to be used with kliff ``NeuralNetwork`` s, such as
    :class:`~orchestrator.potential.dnn.KliffBPPotential`.


    :param use_gpu: Whether to use a GPU for training |default| False
    :type use_gpu: bool
    :param loss_method: Loss function to use |default| 'mse'
    :type loss_method: str
    :param epochs: Number of epochs to train the model |default| 100
    :type epochs: int
    :param batch_size: Number of configurations per mini-batch |default| 32
    :type batch_size: int
    :param learning_rate: Learning rate used by the optimizer |default| 0.001
    :type learning_rate: float
    :param training_split: Fraction of data to use for training (rest for
                           validation) |default| 0.8
    :type training_split: float
    :param optimizer: Optimizer to use for training |default| 'Adam'
    :type optimizer: str
    :param log_per_atom_pred: Whether to log per-atom predictions during
                              training for both in-memory and submitted
                              jobs |default| True
    :type log_per_atom_pred: bool
    :param kwargs: Additional keyword arguments passed to the superclass.
    :type kwargs: dict
    """

    def __init__(
        self,
        use_gpu: bool = False,
        loss_method: str = 'mse',
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        training_split: float = 0.8,
        optimizer: str = 'Adam',
        log_per_atom_pred: bool = True,
        **kwargs,
    ):
        """
        Train and deploy a DNN potential using KLIFF

        :param use_gpu: Whether to use a GPU for training |default| False
        :type use_gpu: bool
        :param loss_method: Loss function to use |default| 'mse'
        :type loss_method: str
        :param epochs: Number of epochs to train the model |default| 100
        :type epochs: int
        :param batch_size: Number of configurations per mini-batch |default| 32
        :type batch_size: int
        :param learning_rate: Learning rate used by the optimizer
                              |default| 0.001
        :type learning_rate: float
        :param training_split: Fraction of data to use for training (rest for
                               validation) |default| 0.8
        :type training_split: float
        :param optimizer: Optimizer to use for training |default| 'Adam'
        :type optimizer: str
        :param per_atom_weights: Per atom weights for the loss function,
                                If boolean, value is provided, the weights
                                are assumed to be present in the provided
                                dataset. |default| ``False``
        :type per_atom_weights: bool
        :param kwargs: Additional keyword arguments passed to the superclass.
        :type kwargs: dict
        """
        self.use_gpu = use_gpu
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.log_per_atom_pred = log_per_atom_pred
        super().__init__(
            training_split=training_split,
            loss_method=loss_method,
            max_evals=epochs,
            optimization_method=optimizer,
            scratch=None,
            **kwargs,
        )

    def checkpoint_trainer(self):
        """
        checkpoint the trainer module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        # no current checkpoint necessary, but can save loss if desired:
        # if model_to_save.is_torch and loss is not None:
        #     loss.save_optimizer_state(f'{save_path}/optimizer_state.pkl')
        """
        save_dict = {
            self.checkpoint_name: {
                'variable': value,
            }
        }
        restarter.write_checkpoint_file(self.checkpoint_file, save_dict)
        """
        pass

    def restart_trainer(self):
        """
        restart the trainer module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        # can reload loss like:
        # loss.load_optimizer_state(file_path)
        """
        # see if any internal variables were checkpointed
        restart_dict = restarter.read_checkpoint_file(
            self.checkpoint_file,
            self.checkpoint_name,
        )
        wrote_potential = restart_dict.get('wrote_potential', False)
        if wrote_potential:
            self.logger('KIMPotential cannot currently restart from a file!')
        """
        pass

    def train(
        self,
        path_type: str,
        potential: Potential,
        storage: "Storage",
        dataset_list: list,
        scheduler: Optional[Scheduler] = None,
        eweight: float = 1.0,
        fweight: float = 1.0,
        vweight: float = 1.0,
        per_atom_weights: bool = False,
        upload_to_kimkit=True,
    ) -> list:
        """
        Train a DNN potential using KLIFF

        This is the main method of the trainer class, and uses the parameters
        supplied at instantiation to perform the potential training by
        minimizing a loss function.

        :param path_type: specifier for the scheduler path, to differentiate
                          training runs
        :type path_type: str
        :param potential: :class:`~orchestrator.potential.dnn.KliffBPPotential`
                          class object containing model to be trained as an
                          attribute
        :type potential: KliffBPPotential
        :param storage: an instance of the storage class
        :type storage: Storage
        :dataset_list: the list of dataset_handles (e.g. collabfit-IDs)
            within the storage object to use as the dataset.
        :type dataset_list: list
        :param scheduler: the scheduler for managing path definition and job
                         submission, if none are supplied, will use the
                         default scheduler defined in this class
                         |default| ``None``
        :type scheduler: Scheduler
        :param eweight: weight of energy data in the loss function
        :type eweight: float
        :param fweight: weight of the force data in the loss function
        :type fweight: float
        :param vweight: weight of the stress data in the loss function
        :type vweight: float
        :param per_atom_weights: Per atom weights for the loss function,
                                If boolean, value is provided, the weights
                                are assumed to be present in the provided
                                dataset. |default| ``False``
        :type per_atom_weights: bool
        :param upload_to_kimkit: True to upload to kimkit repository
        :type upload_to_kimkit: bool
        :returns: trained model, loss object
        :rtype: NeuralNetwork, Loss (KliFF)
        """
        if dataset_list is None or storage is None:
            raise ValueError('A storage object and list of dataset handles'
                             ' are required!')

        if not isinstance(dataset_list, list):
            dataset_list = [dataset_list]
        if scheduler is None:
            scheduler = self.default_scheduler

        combined_dataset = []
        for dataset_handle in dataset_list:
            configs = self._get_training_data(dataset_handle, storage)
            combined_dataset.extend(configs)

        dataset = Dataset.from_ase(ase_atoms_list=combined_dataset,
                                   energy_key=ENERGY_KEY,
                                   forces_key=FORCES_KEY)

        calc = CalculatorTorch(potential.model, gpu=self.use_gpu)
        _ = calc.create(dataset.get_configs(), reuse=False)

        # Create loss_path for logging if enabled
        loss_path = None
        if self.log_per_atom_pred:
            loss_path = scheduler.make_path(
                self.__class__.__name__,
                f'{path_type}_loss',
            )

        loss = Loss(calc,
                    log_per_atom_pred=self.log_per_atom_pred,
                    log_per_atom_pred_path=loss_path)
        _ = loss.minimize(
            method=self.loss_method,
            num_epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.learning_rate,
        )

        save_path_and_name = self._save_model(
            path_type,
            potential,
            loss=loss,
        )
        save_path = '/'.join(save_path_and_name.split('/')[:-1])

        if upload_to_kimkit:
            _ = potential.save_potential_files(work_dir=save_path,
                                               import_to_kimkit=True,
                                               write_to_tmp_dir=True)

        return potential.model, loss

    def _write_training_script(
        self,
        save_path: str,
        loss_path: str,
        dataset_list: list,
        potential: Potential,
        storage: "Storage",
        per_atom_weights: Union[bool, np.ndarray] = False,
        upload_to_kimkit=True,
    ) -> str:
        """
        write a script to run the trainer outside of memory

        this is a helper function for generating a script, training_script.py,
        which can be executed via a scheduler or offline

        :param save_path: path where the training script will be written
        :type save_path: str
        :param loss_path: path where the training losses will be saved, passed
                          to the kliff Loss object
        :type loss_path: str
        :param dataset_list: list of dataset handles which should be used for
                             the training procedure
        :type dataset_list: list of str
        :param potential: Potential instance to be trained, expect its
                          pre-trained state to be written to
                          save_path/potential_to_train.pkl
        :type potential: Potential
        :param storage: instance of the storage module which contains the data
                        to train on
        :type storage: Storage
        :param per_atom_weights: Per atom weights for the loss function,
                                If boolean, value is provided, the weights
                                are assumed to be present in the provided
                                dataset. |default| ``False``
        :type per_atom_weights: bool
        :param upload_to_kimkit: True to upload to kimkit repository,
            currently unsupported
        :type upload_to_kimkit: bool
        :returns: name of the script that is generated (training_script.py)
        :rtype: str
        """
        import_lines = (
            'from kliff.dataset import Dataset\n'
            'from kliff.legacy.loss import Loss\n'
            'from kliff.legacy.calculators import CalculatorTorch\n'
            'from orchestrator.utils.setup_input import '
            'init_and_validate_module_type\n')

        instance_trainer_args = deepcopy(self.trainer_init_args)
        instance_trainer_args["epochs"] = self.trainer_init_args.get(
            "max_evals", 1000)
        # remove unwanted kwargs
        _ = instance_trainer_args.pop("max_evals", None)  # epochs
        _ = instance_trainer_args.pop("optimization_method", None)  # optimizer
        _ = instance_trainer_args.pop("scratch", None)  # no scratch needed
        trainer_dict = {
            'trainer_type': self.factory_token,
            'trainer_args': instance_trainer_args
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
        init_potential = ('potential = init_and_validate_module_type('
                          f'"potential", {potential_dict}, '
                          'single_input_dict=True)\n')

        load_potential = "potential.load_potential('potential_to_train.pkl')"

        construct_dataset = (
            'combined_dataset = []\n'
            f'for ds_handle in {dataset_list}:\n'
            '    configs = trainer._get_training_data(ds_handle, storage)\n'
            '    combined_dataset.extend(configs)\n'
            'dataset = Dataset.from_ase(ase_atoms_list=combined_dataset,\n'
            f'                       energy_key="{ENERGY_KEY}",\n'
            f'                       forces_key="{FORCES_KEY}")\n')

        construct_trainer = (
            'calc = CalculatorTorch(potential.model, '
            'gpu=trainer.use_gpu)\n'
            '_ = calc.create(dataset.get_configs(), reuse=False)\n'
            f'loss_path = "{loss_path}"\n'
            'log_per_atom_pred = trainer.log_per_atom_pred\n'
            'if log_per_atom_pred and loss_path is None:\n'
            '   log_per_atom_pred = False  # Cannot log without a path\n'
            f"loss = Loss(calc, log_per_atom_pred=log_per_atom_pred,\n "
            f"               log_per_atom_pred_path='{loss_path}')\n")

        execute_training = ('_ = loss.minimize(method=trainer.loss_method, '
                            'num_epochs=trainer.epochs, '
                            'batch_size=trainer.batch_size, '
                            'lr=trainer.learning_rate)')

        save_potential = ('potential._write_potential_to_file('
                          "'trained_potential.pkl')")

        save_model = (
            'save_path_and_name = trainer._save_model('
            '".",'
            'potential,'
            'loss=loss,'
            'create_path=False,'
            ')'
            '\n'
            'save_path = "/".join(save_path_and_name.split("/")[:-1])')

        if upload_to_kimkit:
            save_potential.append(
                '\n'
                'potential.save_potential_files(work_dir=save_path,'
                'import_to_kimkit=True,'
                'write_to_tmp_dir=True)')

        script = '\n'.join([
            import_lines,
            init_trainer,
            init_storage,
            init_potential,
            load_potential,
            construct_dataset,
            construct_trainer,
            execute_training,
            save_potential,
            save_model,
        ])
        with open(f'{save_path}/training_script.py', 'w') as fout:
            fout.write(script)
        return 'training_script.py'

    def submit_train(
        self,
        path_type: str,
        potential: Potential,
        storage: "Storage",
        dataset_list: list,
        scheduler: Scheduler,
        job_details: dict,
        eweight: float = 1.0,
        fweight: float = 1.0,
        vweight: float = 1.0,
        per_atom_weights: bool = False,
        upload_to_kimkit=True,
    ) -> int:
        """
        Asynchronously train the potential based on the trainer details

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
                         submission, if none are supplied, will use the
                         default scheduler defined in this class
        :type scheduler: Scheduler
        :param eweight: weight of energy data in the loss function
        :type eweight: float
        :param fweight: weight of the force data in the loss function
        :type fweight: float
        :param vweight: weight of the stress data in the loss function
        :type vweight: float
        :param per_atom_weights: Per atom weights for the loss function,
                                If boolean, value is provided, the weights
                                are assumed to be present in the provided
                                dataset. |default| ``False``
        :type per_atom_weights: bool
        :param upload_to_kimkit: True to upload to kimkit repository
        :type upload_to_kimkit: bool
        :returns: calculation ID of the submitted job
        :rtype: int
        """
        if dataset_list is None or storage is None:
            raise ValueError('A storage object and list of dataset handles'
                             ' are required!')

        if not isinstance(dataset_list, list):
            dataset_list = [dataset_list]
        save_path = scheduler.make_path(self.__class__.__name__,
                                        f'{path_type}')

        # Only create loss_path if logging is enabled
        loss_path = None
        if self.log_per_atom_pred:
            loss_path = scheduler.make_path(self.__class__.__name__,
                                            f'{path_type}_loss')
            loss_path = path.abspath(loss_path)

        script = self._write_training_script(
            save_path,
            loss_path,
            dataset_list,
            potential,
            storage,
            upload_to_kimkit=upload_to_kimkit,
        )
        potential._write_potential_to_file(
            f'{save_path}/potential_to_train.pkl')
        job_details['custom_preamble'] = 'python'
        calc_id = scheduler.submit_job(
            script,
            save_path,
            job_details=job_details,
        )
        return calc_id

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
                          class object that will be updated with the model
                          saved to disk after the training job.
        :type potential: KliffBPPotential
        :param scheduler: the scheduler for managing path definition and job
                         submission, if none are supplied, will use the
                         default scheduler defined in this class
                         |default| ``None``
        :type scheduler: Scheduler
        """
        model_path = scheduler.get_job_path(calc_id) + '/trained_potential.pkl'
        self.logger.info(f'Loading potential from: {model_path}')
        scheduler.block_until_completed(calc_id)
        potential.load_potential(model_path)
