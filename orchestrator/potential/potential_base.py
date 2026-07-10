from abc import ABC, abstractmethod
import os
import numpy as np
from ase import Atoms
from typing import Union, Optional
from ..storage import Storage
from ..workflow import Workflow
from ..utils.recorder import Recorder
from dataclasses import dataclass


@dataclass
class ModelHyperparameters:
    cutoff_radius: float


class Potential(Recorder, ABC):
    """
    Abstract base class to manage the construction of interatomic potentials

    The potential class encapsulates the interatomic potential data and
    parameterization. Potentials can either be constructed from scratch or
    loaded from existing data. Each concrete potential constructor should
    define the hyperparameters that can be set for the potential.
    """

    # Instead of providing potential_files list to __init__, provide the
    # model-specific files; the concrete class will handle construction of
    # potential_files.
    _required_files = None
    _optional_files = None
    potential_files = None  # will be set on instantiation

    def __init__(
        self,
        species: list[str],
        potential_name: str = 'orchestrator_potential',
        template: Optional[str] = None,
        **kwargs,
    ):
        """
        set variables and initialize the recorder

        :param species: list of strings containing element symbols
        :type species: list[str]
        :param potential_name: Name of the potential, used as filename prefix
        :type potential_name: str
        :param template: Optional template file from which to load the model.
        :type template: str
        :param hyperparameters:
        """

        self.species = species
        self.potential_name = potential_name
        self.hyperparameters = None  # will be set by child init()
        self.trainer_hyperparameters = None  # Will be set by .train()
        self.template = template

        # move this here because KIM mix-in needs potential_name
        super().__init__(**kwargs)

        # Determine if the class inherits the optional mixins
        if hasattr(self, 'install_model'):
            self.kim_api_compatible = True
        else:
            self.kim_api_compatible = False

        self.kimkit_compatible = hasattr(self, 'initialize_from_kim')

        # ASE calculator object used for evaluation with this potential
        self._potential_calculator = None

        if self.potential_files is None:
            self.potential_files = {}

    @classmethod
    @abstractmethod
    def initialize_from_files(cls, files):
        """
        Alternative constructor for starting from externally generated files

        Child classes should define the explicit files that should be passed
        in. This constructor can be accessed by adding "initialize_from":
        "potential_files" in the ``potential_args`` dict.
        """
        raise NotImplementedError

    @abstractmethod
    def load_potential(self, path: str):
        """
        Parameterize the potential by loading the potential files from a path.

        Note that this is specifically intended to be used in conjunction with
        save_potential, as both files will assume hard-coded file names. If you
        wish to load a potential using files generated external to the
        orchestrator, you should use `__init__()`.

        :param path: Path to directory containing potential files
        :type path: str
        """
        pass

    @abstractmethod
    def save_potential(
        self,
        path: str,
        makedirs: bool = False,
    ) -> Union[str, list[str]]:
        """
        Save the potential and other necessary data to disk.

        Note that this uses a strict naming convention for integration with
        load_potential().

        :param path: Directory path where the model should be saved
        :param makedirs: If True, creates folder if it doesn't exist. Default
            is False.
        :returns: List of paths to the saved model files
        """
        # whatever is saved should be "invertible" with load_potential()
        # self.potential_name is used to name the potential files at the given
        # path
        pass

    def _check_files_set_and_exist(self, check_exist=True):
        """Asserts that all files in self._required_files exist."""
        missing = []
        for k in self._required_files:
            path = self.potential_files.get(k)
            if path is None:
                missing.append(k)
                continue
            if check_exist and not os.path.exists(path):
                missing.append(k)
        if missing:
            raise FileNotFoundError(f'Given files: {missing} not found')
        else:
            return True

    @abstractmethod
    def _initialize_calculator(self):
        """Set up the ASE calculator."""
        pass

    def evaluate(
        self,
        atoms: Atoms,
        return_energy: bool = True,
        return_forces: bool = True,
        return_stress: bool = True,
    ) -> Union[
            float,
            np.ndarray,
            tuple[float],
            tuple[np.ndarray],
            tuple[float, np.ndarray],
            tuple[np.ndarray, np.ndarray],
            tuple[float, np.ndarray, np.ndarray],
    ]:
        """Evaluate the energy, forces, and stress of a configuration of
        atoms specified in an ASE atoms object.

        :param atoms: Atomic configuration as an ASE atoms object
        :type atoms: ase.Atoms
        :param return_energy: Whether to calculate and return the energy
        :type return_energy: bool
        :param return_forces: Whether to calculate and return the forces
        :type return_forces: bool
        :param return_stress: Whether to calculate and return the stress tensor
        :type return_stress: bool

        :returns: Selected combination of energy (float), forces (np.ndarray),
                 and stress (np.ndarray) based on boolean flags
        :rtype: Various possible return types depending on which values are
            requested
        """

        if self._potential_calculator is None:
            # this function will raise exceptions if it fails
            self._initialize_calculator()

        atoms.calc = self._potential_calculator

        # Calculate only the requested quantities
        results = []

        if return_energy:
            energy = atoms.get_potential_energy()
            results.append(energy)

        if return_forces:
            forces = atoms.get_forces()
            results.append(forces)

        if return_stress:
            stress = atoms.get_stress()
            results.append(stress)

        # Return results based on how many items we have
        if len(results) == 1:
            return results[0]  # Return single value unwrapped from tuple
        else:
            return tuple(results)  # Return a tuple of the requested values

    def get_hyperparameters(self) -> ModelHyperparameters | dict:
        """
        return the hyperparameters of the potential
        """
        return self.hyperparameters

    @abstractmethod
    def train(
        self,
        dataset_list: list[str],
        storage: Storage,
        workflow: Workflow,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        stress_weight: float = 1.0,
        train_frac: float = 0.8,
        test_frac: float = 0.1,
        val_frac: float = 0.1,
        per_atom_weights: Union[bool, str, np.ndarray] = False,
        **kwargs,
    ) -> tuple[str, float]:
        """
        Train the potential using the provided data.

        Note that this function will automatically apply per-atom masking (set
        loss contribution to zero for masked atoms)if SELECTION_MASK_KEY is set
        in the atoms.info dictionaries.

        :param dataset_list: list of dataset handles to use for training
        :type dataset_list: list[str]
        :param storage: Storage object to access training data
        :type storage: Storage
        :param workflow: Workflow object for job management
        :type workflow: Workflow
        :param energy_weight: Weight for energy terms in the loss function
        :type energy_weight: float
        :param force_weight: Weight for force terms in the loss function
        :type force_weight: float
        :param stress_weight: Weight for stress terms in the loss function
        :type stress_weight: float
        :param train_frac: Fraction of data to use for training
        :type train_frac: float
        :param test_frac: Fraction of data to use for testing
        :type test_frac: float
        :param val_frac: Fraction of data to use for validation
        :type val_frac: float
        :param per_atom_weights: Controls per-atom weighting
        :type per_atom_weights: Union[bool, str, np.ndarray]
        :param kwargs: Additional potential-specific training parameters

        :returns: Tuple containing (path? to trained potential, training error)
        :rtype: tuple[str, float]
        """

        # Get training data
        if not isinstance(dataset_list, list):
            dataset_list = [dataset_list]

        combined_dataset = []
        for dataset_handle in dataset_list:
            configs = storage.get_data(dataset_handle)
            combined_dataset.extend(configs)

        # Process data - either have an
        processed_data = self._format_training_data(combined_dataset)

        print(f'Do training with {processed_data}')

        # set model path after training
        model_path = None

        # Record training hyperparameters
        self.trainer_hyperparameters = ...

        # Train model (implementation depends on model type)

        # Return trained model and error metric
        return model_path, 0.0

    @abstractmethod
    def submit_train(
        self,
        dataset_list: list[str],
        storage: Storage,
        workflow: Workflow,
        job_details: dict,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        stress_weight: float = 1.0,
        train_frac: float = 0.8,
        test_frac: float = 0.1,
        val_frac: float = 0.1,
        per_atom_weights: Union[bool, str, np.ndarray] = False,
        **kwargs,
    ) -> Union[str, int]:
        """
        Train the potential using the provided data via a submitted job

        :param dataset_list: list of dataset handles to use for training
        :type dataset_list: list[str]
        :param storage: Storage object to access training data
        :type storage: Storage
        :param workflow: Workflow object for job management
        :type workflow: Workflow
        :param job_details: information controlling job submission
        :type job_details: dict
        :param energy_weight: Weight for energy terms in the loss function
        :type energy_weight: float
        :param force_weight: Weight for force terms in the loss function
        :type force_weight: float
        :param stress_weight: Weight for stress terms in the loss function
        :type stress_weight: float
        :param train_frac: Fraction of data to use for training
        :type train_frac: float
        :param test_frac: Fraction of data to use for testing
        :type test_frac: float
        :param val_frac: Fraction of data to use for validation
        :type val_frac: float
        :param per_atom_weights: Controls per-atom weighting
        :type per_atom_weights: Union[bool, str, np.ndarray]
        :param kwargs: Additional potential-specific training parameters

        :returns: Tuple containing (path? to trained potential, training error)
        :rtype: tuple[str, float]
        """
        pass

    @abstractmethod
    def load_from_submitted_training(
        self,
        calc_id: Union[str, int],
        workflow: Workflow,
    ):
        """
        Load a potential that was trained via a submitted job
        """
        # include checks that training finished appropriately
        # should be able to use load_potential under the hood
        pass

    @abstractmethod
    def get_lammps_commands(self) -> str:
        """
        Extract required commands to inject into a LAMMPS input file to
        enable using external potentials with direct LAMMPS interface.
        """
        pass
