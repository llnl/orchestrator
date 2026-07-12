from abc import ABC, abstractmethod
from ..utils.recorder import Recorder
from ..scheduler.factory import scheduler_builder
from typing import Optional, Union
import numpy as np
from ase import Atoms
from ..storage.storage_base import Storage
from ..potential.potential_base import Potential
from ..scheduler.scheduler_base import Scheduler


class Trainer(Recorder, ABC):
    """
    Abstract base class to manage the training of different potentials

    The trainer class is responsible for handling the loading/assignment of
    training data, as well as the actual process of training a potential
    """

    def __init__(self, **kwargs):
        """
        set variables and initialize the recorder and default scheduler
        """
        super().__init__()

        #: default scheduler to use within the trainer class
        self.default_scheduler = scheduler_builder.build(
            'LOCAL',
            {'root_directory': './trainer'},
        )

    @abstractmethod
    def checkpoint_trainer(self):
        """
        checkpoint the trainer module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        pass

    @abstractmethod
    def restart_trainer(self):
        """
        restart the trainer module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        pass

    @abstractmethod
    def _get_training_data(
        self,
        dataset_handle: str,
        storage: Storage,
    ) -> list[Atoms]:
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
        :returns: training dataset
        :rtype: list of ASE Atoms
        """
        pass

    @abstractmethod
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
        per_atom_weights: Union[bool, np.ndarray] = False,
        write_training_script: bool = True,
        upload_to_kimkit=True,
    ) -> list:
        """
        Train the potential based on the specific trainer details

        This is a main method of the trainer class, and uses the parameters
        supplied at instantiation to perform the potential training by
        minimizing a loss function.

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
            defined in this class |default| ``None``
        :type scheduler: Scheduler
        :param per_atom_weights: True to read from dataset, or numpy array
            |default| ``False``
        :type per_atom_weights: either boolean or np.ndarray
        :param write_training_script: True to write a training script in the
            working trainer directory |default| ``True``
        :type write_training_script: bool
        :param upload_to_kimkit: True to upload to kimkit repository
        :type upload_to_kimkit: bool
        :returns: trained model, loss object
        :rtype: implementation dependent
        """
        pass

    @abstractmethod
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
            submission
        :type scheduler: Scheduler
        :param eweight: weight of energy data in the loss function
        :type eweight: float
        :param fweight: weight of the force data in the loss function
        :type fweight: float
        :param vweight: weight of the stress data in the loss function
        :type vweight: float
        :param per_atom_weights: True to read from dataset, or numpy array
            |default| ``False``
        :type per_atom_weights: either boolean or np.ndarray
        :param upload_to_kimkit: True to upload to kimkit repository
        :type upload_to_kimkit: bool
        :returns: calculation ID of the submitted job
        :rtype: int
        """
        pass

    @abstractmethod
    def _save_model(
        self,
        path_type,
        potential,
        loss=None,
        create_path=True,
        scheduler=None,
    ):
        """
        Save the model and (optionally) loss data

        Write the model (and loss) data to disk from memory

        :param path_type: specifier for the scheduler path, to differentiate
            training runs and where the model will be saved
        :type path_type: str
        :param potential: potential to be saved. This method takes a full
            :class:`~orchestrator.potential.potential_base.Potential` class
            object
        :type potential: Potential
        :param loss: loss object to save, optional. |default| ``None``
        :type loss: implementation dependent
        :param create_path: if the function needs to create a new path, or if
            path_type should be used as the full path |default| ``True``
        :type create_path: boolean
        :param scheduler: the scheduler for managing path definition, if none
            are supplied, will use the default scheduler defined in this class
            |default| ``None``
        :type scheduler: Scheduler
        :returns: path where the model is saved
        :rtype: str
        """
        pass

    @abstractmethod
    def load_from_submitted_training(
        self,
        calc_id: int,
        potential: Potential,
        scheduler: Scheduler,
    ):
        """
        reload a potential that was trained via a submitted job
        """
        pass
