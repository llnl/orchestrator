import pathlib
import shutil
from glob import glob
from ase import Atoms
import yaml
from kliff.legacy.loss import Loss
from ..trainer_base import Trainer
from ...utils.data_standard import (
    ENERGY_WEIGHT_KEY,
    FORCES_WEIGHTS_KEY,
)
from typing import Optional, Union, Dict, Any, List
import numpy as np
from ...potential.potential_base import Potential
from ...scheduler.scheduler_base import Scheduler

from typing import TYPE_CHECKING
if TYPE_CHECKING:  # avoid circular imports
    from orchestrator.storage.storage_base import Storage
    from pathlib import Path


class KLIFFTrainer(Trainer):
    """
    Train and deploy a potential using KLIFF

    The trainer class is responsible for handling the loading/assignment of
    training data, as well as the actual process of training a potential. One
    should use specific subclasses of KLIFFTrainer instead of this base class.

    :param training_split: Fraction of the dataset to be allocated for
        training (e.g., 0.8 for 80%). Defaults to 0.8.
    :type training_split: float
    :param loss_method: The type of loss function to be used during training
        (e.g., "mse" for mean squared error).
    :type loss_method: str
    :param max_evals: Maximum number of evaluations (e.g., iterations or
        function calls) for the optimizer. Defaults to 1000.
    :type max_evals: int
    :param optimization_method: The optimization algorithm to employ for
        training the potential (e.g., "L-BFGS-B", "Adam")
    :type optimization_method: str
    :param scratch: Path to a directory for storing temporary or scratch
        files during training. If None, it defaults to
        './scratch_kliff' within the execution directory.
    :type scratch: str, optional
    :param kwargs: Arbitrary keyword arguments that may be used by
        specific subclasses or for advanced configuration options.
    :type kwargs: dict
    """

    def __init__(
        self,
        training_split: float = 0.8,
        loss_method: str = "mse",
        max_evals: int = 1000,
        optimization_method: str = "L-BFGS-B",
        scratch: str = None,
        **kwargs,
    ):
        """
        set variables and initialize the recorder and default scheduler

        :param training_split: Fraction of the dataset to be allocated for
            training (e.g., 0.8 for 80%). Defaults to 0.8.
        :type training_split: float
        :param loss_method: The type of loss function to be used during
            training (e.g., "mse" for mean squared error).
        :type loss_method: str
        :param max_evals: Maximum number of evaluations (e.g., iterations or
            function calls) for the optimizer. Defaults to 1000.
        :type max_evals: int
        :param optimization_method: The optimization algorithm to employ for
            training the potential (e.g., "L-BFGS-B", "Adam")
        :type optimization_method: str
        :param scratch: Path to a directory for storing temporary or scratch
            files during training. If None, it defaults to
            './scratch_kliff' within the execution directory.
        :type scratch: str, optional
        :param kwargs: Arbitrary keyword arguments that may be used by
            specific subclasses or for advanced configuration options.
        :type kwargs: dict
        """
        self.loss_method = loss_method
        self.max_evals = max_evals
        self.optimization_method = optimization_method
        self.training_split = training_split

        if scratch is None:
            self.scratch = './scratch_kliff'

        # arguments to reinitialize an instance of the trainer
        self.trainer_init_args = {
            'loss_method': loss_method,
            'max_evals': max_evals,
            'optimization_method': optimization_method,
            'scratch': scratch,
        }

        super().__init__(**kwargs)

    def _get_training_data(
        self,
        dataset_handle: str,
        storage: "Storage",
    ) -> List[Atoms]:
        """
        Get the training data configurations

        Retrieve the dataset specified by dataset_handle from the passed
        storage module. This dataset can be augmented or otherwise modified
        (i.e. adding weights) as necessary for training.

        :param dataset_handle: the identifier of the dataset to extract from
                               the storage module
        :type dataset_handle: str
        :param storage: storage instance where the training data is saved
        :type storage: Storage
        :returns: training data of configurations
        :rtype: KliFF Dataset
        """
        self.logger.info('Reading training data from storage')

        # TODO: direct initialization of ds from colabfit
        try:
            training_set = storage.get_data(dataset_handle)
        except Exception as e:  # catch all?
            print(('Storage module is not properly set. Cannot '
                   'get training data from Storage.'))
            print(f'Caught this error {e}')
            raise e

        return training_set

    def _save_model(
        self,
        path_type: Union[str, "Path"],
        potential: Potential,
        potential_name: str = 'kim_potential',
        loss: Optional[Loss] = None,
        create_path: bool = True,
        scheduler: Optional[Scheduler] = None,
    ) -> str:
        """
        Deploy a KIM model. Writes KIM potential to Current Working Directory

        Write the model (and loss) data to disk from memory using the KIM/KliFF
        infrastructure. Written pkl files can be directly loaded, while the
        installed version of the potential can be run using KIM libraries.

        :param path_type: specifier for the scheduler path, to differentiate
                          training runs and where the model will be saved
        :type path_type: str
        :param potential: potential to be saved; one of
                         :class:`~orchestrator.potential.dnn.KliffBPPotential`
                         ,:class:`~orchestrator.potential.kim.KIMPotential`
        :type potential: KliffBPPotential or KIMPotential or a kliff model
        :param potential_name: name to save the potential as
                              |default| 'kim_potential'
        :type potential_name: str
        :param loss: loss object to save, optional. Used if potential is a
                    pytorch model. |default| ``None``
        :type loss: Loss (kliff.loss)
        :param create_path: if the function needs to create a new path, or if
                            path_type should be used as the full path
                            |default| ``True``
        :type create_path: boolean
        :param scheduler: the scheduler for managing path definition, if none
                          are supplied, will use the default scheduler defined
                          in this class |default| ``None``
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

        self.logger.info(f'Saving model state in {save_path}')
        # potential._write_potential_to_file(f'{save_path}/final_model')
        _ = potential.save_potential_files(work_dir=f'{save_path}/final_model',
                                           import_to_kimkit=False,
                                           write_to_tmp_dir=False)
        try:
            if potential.model.is_torch and loss is not None:
                loss.save_optimizer_state(f'{save_path}/optimizer_state.pkl')
        except AttributeError:
            # non torch KIMModel
            pass

        potential.install_potential_in_kim_api(save_path=save_path,
                                               potential_name=potential.kim_id,
                                               install_locality='CWD')

        # clean up any default files if present.
        # Usually after the training you might have kliff.log, *.pkl, and
        # kliff_saved_model folder.
        fully_qualified_save_path = f'{save_path}/{potential_name}'

        # kliff log file
        try:
            shutil.move("kliff.log", fully_qualified_save_path)
        except Exception as err:
            self.logger.info(f'Failed to move kliff.log: {err}'
                             ' May be this file does not exist?')

        # any saved fingerprints
        try:
            pkl_files = glob("finger*.pkl")
            for file in pkl_files:
                shutil.move(file, fully_qualified_save_path)
        except Exception as err:
            self.logger.info(f'Failed to move {pkl_files}: {err}'
                             ' May be no pkl files exist?')

        # kliff_saved_model folder
        try:
            shutil.move("kliff_saved_model", fully_qualified_save_path)
        except Exception as err:
            self.logger.info(f'Failed to move kliff_saved_model: {err}'
                             ' May be this folder does not exist?')

        return fully_qualified_save_path

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
    ) -> tuple[Potential, Loss]:
        """
        Train the potential based on the specific trainer details

        KLIFFTrainer should not be used for training, it is a parent class to
        specific implementations

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
                         default scheduler defined in this class |default|
                         ``None``
        :type scheduler: Scheduler
        :param eweight: weight of energy data in the loss function
        :type eweight: float
        :param fweight: weight of the force data in the loss function
        :type fweight: float
        :param vweight: weight of the stress data in the loss function
        :type vweight: float
        :param per_atom_weights: True to read from dataset, |default|
                                ``False``
        :type per_atom_weights: bool
        :param upload_to_kimkit: True to upload to kimkit repository
        :type upload_to_kimkit: bool
        :returns: trained model, loss object
        :rtype: implementation dependent
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not have '
                                  f'train implemented. Use a subclass!')

    def submit_train(
        self,
        path_type: str,
        potential: Potential,
        storage_args: dict,
        storage: "Storage",
        dataset_list: list,
        scheduler: Optional[Scheduler] = None,
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
        :returns: calculation ID of the submitted job
        :rtype: int
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not have '
                                  f'submit_train implemented. Use a subclass!')

    @staticmethod
    def _create_training_manifest(
        workspace_name: str,
        seed: int,
        dataset_type: str,
        dataset_path: str,
        model_name: str,
        batch_size: int,
        epochs: int,
        optimizer: str,
        loss: str = 'mse',
        weights: Optional[Dict[str, Any]] = None,
        train_size: Optional[Union[int, np.ndarray]] = None,
        val_size: Optional[Union[int, np.ndarray]] = None,
        export: Optional[Dict[str, Any]] = None,
        transform_params: Union[bool, List[str]] = True,
        shuffle: bool = True,
        lr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Create a training manifest skeleton for KLIFF
        :param workspace_name: name of the workspace
        :type workspace_name: str
        :param seed: random seed for reproducibility
        :type seed: int
        :param dataset_type: type of dataset (e.g. 'ase', 'path')
        :type dataset_type: str
        :param dataset_path: path to the dataset
        :type dataset_path: str
        :param model_name: name of the model
        :type model_name: str
        :param batch_size: batch size for training
        :type batch_size: int
        :param epochs: number of training epochs
        :type epochs: int
        :param optimizer: optimizer to use for training
        :type optimizer: str
        :param loss: loss function to use for training
        :type loss: str
        :param weights: weights for the loss function
        :type weights: dict
        :param train_size: size of the training dataset
        :type train_size: int or np.ndarray
        :param val_size: size of the validation dataset
        :type val_size: int or np.ndarray
        :param export: export options for the model
        :type export: dict
        :param transform_params: parameters for data transformation
        :type transform_params: bool or list
        :param shuffle: whether to shuffle the dataset
        :type shuffle: bool
        :param lr: learning rate for the optimizer
        :type lr: float
        :returns: training manifest dictionary
        :rtype: dict
        """
        manifest = {
            "workspace": {
                "name": workspace_name,
                "seed": seed,
                "resume": False,
            },
            "dataset": {
                "type": dataset_type,
                "path": dataset_path,
                "shuffle": shuffle,
            },
            "model": {
                "name": model_name,
            },
            "training": {
                "loss": {
                    "function": loss,
                    "weights": weights
                },
                "batch_size": batch_size,
                "epochs": epochs,
                "optimizer": {
                    "name": optimizer
                },
                "training_dataset": {
                    "train_size": train_size,
                },
                "validation_dataset": {
                    "val_size": val_size
                },
                "verbose": True,
            },
            "transforms": {}
        }
        if isinstance(transform_params, list):
            manifest["transforms"]["parameter"] = transform_params

        if lr is not None:
            manifest["training"]["optimizer"]["lr"] = lr

        if export is not None:
            manifest["export"] = export

        return manifest

    @staticmethod
    def _generate_per_atom_weights_yaml(
        dataset: list[Atoms],
        path: Optional[Union[str, pathlib.Path, None]] = None,
    ) -> str:
        """
        Generate a YAML file for per-atom weights

        :param dataset: list of ASE Atoms objects with weights embedded
        :type dataset: list[Atoms]
        :param path: path to save the YAML file, file will be saved
                     as path/weights.yaml default is current working directory
        :type path: str or pathlib.Path

        :returns: path to the generated YAML file

        """
        if path is None:
            path = pathlib.Path.cwd()
        else:
            path = pathlib.Path(path)

        path = path / 'weights.yaml'

        path.parent.mkdir(parents=True, exist_ok=True)
        weights = []

        with open(path, 'w') as f:
            for atom in dataset:
                energy_weight = atom.info.get(ENERGY_WEIGHT_KEY, 1.0)
                forces_weight = atom.info.get(FORCES_WEIGHTS_KEY, None)
                if forces_weight is None:
                    forces_weight = np.ones(atom.get_number_of_atoms())

                weights.append({
                    "energy": energy_weight,
                    "forces": forces_weight.tolist()
                })

            yaml.safe_dump(weights, f)

        return str(path)
