import os
import shutil
from copy import deepcopy
import numpy as np

import ase.io
from kliff.trainer.kim_trainer import KIMTrainer
from .kliff import KLIFFTrainer
from ...utils.data_standard import FORCES_KEY, ENERGY_KEY

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from orchestrator.potential.potential_base import Potential
    from orchestrator.potential.kim import KIMPotential
    from orchestrator.scheduler.scheduler_base import Scheduler
    from kliff.models import KIMModel
    from orchestrator.storage.storage_base import Storage


class ParametricModelTrainer(KLIFFTrainer):
    """
    Train and deploy a general parametric model potential using KLIFF

    The trainer class is responsible for handling the loading/assignment of
    training data, as well as the actual process of training a potential. This
    trainer is intended to be used with kliff Parametric model.

    :param model_name: name of the model to train
    :type model_name: str
    :param params_to_update: List of model parameters to update during training
    :type params_to_update: list
    :param training_split: Fraction of data to use for training (rest for
                           validation) |default| 1.0
    :type training_split: float
    :param loss_method: Loss function to use |default| 'mse'
    :type loss_method: str
    :param max_evals: Maximum number of optimization evaluations |default| 1000
    :type max_evals: int
    :param optimization_method: Optimization algorithm to use |default|
                                'L-BFGS-B'
    :type optimization_method: str
    :param scratch: Path to scratch directory for temporary files |default|
                    None
    :type scratch: str or None
    """

    def __init__(
        self,
        model_name: str,
        params_to_update: list,
        training_split:
        float = 1.0,  # train on all data for physics based models
        loss_method: str = "mse",
        max_evals: int = 1000,
        optimization_method: str = "L-BFGS-B",
        scratch: Optional[str] = None,
        **kwargs,
    ):
        """
        Train and deploy a general parametric model potential using KLIFF

        The trainer class is responsible for handling the loading/assignment of
        training data, as well as the actual process of training a potential.
        This trainer is intended to be used with kliff Parametric model.

        :param model_name: name of the model to train
        :type model_name: str
        :param params_to_update: List of model parameters to update during
                                 training
        :type params_to_update: list
        :param training_split: Fraction of data to use for training (rest for
                               validation) |default| 1.0
        :type training_split: float
        :param loss_method: Loss function to use |default| 'mse'
        :type loss_method: str
        :param max_evals: Maximum number of optimization evaluations
                         |default| 1000
        :type max_evals: int
        :param optimization_method: Optimization algorithm to use |default|
                                    'L-BFGS-B'
        :type optimization_method: str
        :param scratch: Path to scratch directory for temporary files
                        |default| None
        :type scratch: str or None
        """
        if scratch is None:
            scratch = "kliff_scratch"

        # TODO: make this unique or direct from colabfit
        dataset_path = scratch + "/kliff_ds.xyz"

        trainer_manifest = KLIFFTrainer._create_training_manifest(
            workspace_name=scratch,
            seed=1234,  # fixed seed for now
            dataset_type="ase",
            dataset_path=dataset_path,
            model_name=model_name,
            batch_size=None,
            epochs=max_evals,
            optimizer=optimization_method,
            loss=loss_method,
            weights={
                "energy": None,
                "forces": None
            },
            train_size=-1,  # determine at runtime
            export=None,  # export using orch
            transform_params=params_to_update,
            shuffle=False,
        )
        super().__init__(
            loss_method=loss_method,
            max_evals=max_evals,
            optimization_method=optimization_method,
            scratch=scratch,
            training_split=training_split,
            **kwargs,
        )
        self.trainer_manifest = deepcopy(trainer_manifest)

    def checkpoint_trainer(self):
        pass

    def restart_trainer(self):
        pass

    def train(
        self,
        path_type: str,
        potential: "KIMPotential",
        storage: "Storage",
        dataset_list: list,
        scheduler=None,
        eweight: float = 1.0,
        fweight: float = 1.0,
        vweight: float = 1.0,
        per_atom_weights: bool = False,
        upload_to_kimkit=True,
    ) -> tuple["KIMModel", None]:  # all information is in trainer args
        """
        Train a parametric potential using KLIFF

        This is the main method of the trainer class, and uses the parameters
        supplied at instantiation to perform the potential training by
        minimizing a loss function.

        :param path_type: specifier for the scheduler path, to differentiate
            training runs
        :type path_type: str
        :param potential: :class:`~orchestrator.potential.kim.KIMPotential`
            class object containing model to be trained as an attribute
        :type potential: KIMPotential
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
        :param per_atom_weights: Per atom weights for the loss function,
            If boolean, value is provided, the weights are assumed to be
            present in the provided dataset. |default| ``False``
        :type per_atom_weights: bool
        :param upload_to_kimkit: True to upload to kimkit repository
        :type upload_to_kimkit: bool
        :returns: trained model, loss object
        :rtype: KIMModel, None
        """

        if dataset_list is None or storage is None:
            raise ValueError('A storage object and list of dataset handles'
                             ' are required!')
        if not isinstance(dataset_list, list):
            dataset_list = [dataset_list]
        if scheduler is None:
            scheduler = self.default_scheduler

        workspace = self.trainer_manifest["workspace"]["name"]
        if os.path.exists(workspace):
            shutil.rmtree(workspace)
        os.makedirs(workspace, exist_ok=True)

        combined_dataset = []
        for dataset_handle in dataset_list:
            configs = self._get_training_data(dataset_handle, storage)
            combined_dataset.extend(configs)

        dataset_file = self.trainer_manifest["dataset"]["path"]
        ase.io.write(dataset_file, combined_dataset)

        self.trainer_manifest["training"]["loss"]["weights"] = {
            "energy": eweight,
            "forces": fweight
        }

        if per_atom_weights:
            weights_file = KLIFFTrainer._generate_per_atom_weights_yaml(
                combined_dataset, self.trainer_manifest["workspace"]["name"])
            self.trainer_manifest["training"]["training_dataset"][
                "weights"] = weights_file
        self.trainer_manifest["dataset"]["keys"] = {
            "energy": ENERGY_KEY,
            "forces": FORCES_KEY
        }

        train_size = int(len(combined_dataset) * self.training_split)

        self.trainer_manifest["training"]["training_dataset"][
            "train_size"] = train_size

        potential.model.set_params_mutable(
            self.trainer_manifest["transforms"]["parameter"])

        kim_trainer = KIMTrainer(
            self.trainer_manifest,
            potential.model)  # Avoid modifying input args, just in case

        kim_trainer.train()

        potential.model = kim_trainer.model

        save_path_and_name = self._save_model(
            path_type,
            potential,
        )

        save_path = '/'.join(save_path_and_name.split('/')[:-1])

        if upload_to_kimkit:
            _ = potential.save_potential_files(work_dir=save_path,
                                               import_to_kimkit=True,
                                               write_to_tmp_dir=True)

        return potential.model, None

    def submit_train(
        self,
        path_type: str,
        potential: "Potential",
        storage_args: dict,
        scheduler: "Scheduler",
        job_details: dict,
        eweight: float = 1.0,
        fweight: float = 1.0,
        vweight: float = 1.0,
        per_atom_weights: Union[bool, np.ndarray, str] = False,
        upload_to_kimkit=True,
    ) -> int:
        """
        Asychronously train the potential based on the trainer details
        """
        raise NotImplementedError("not yet supported for parametric kliff")

    def load_from_submitted_training(
        self,
        calc_id: int,
        potential: "Potential",
        scheduler: "Scheduler",
    ):
        """
        reload a potential that was trained via a submitted job
        """
        raise NotImplementedError("not yet supported for parameteric kliff")
