from abc import ABC, abstractmethod
from ..utils.recorder import Recorder
from typing import Optional, Dict, Any, Union
from ..scheduler.factory import scheduler_builder
from ..scheduler import Scheduler
from ..storage import Storage
from ..potential import Potential


class TargetProperty(Recorder, ABC):
    """
    General class to manage target property calculations

    :param target_property_args: general argument structure which is specified
        by individual implementations
    :type args: dict
    """

    def __init__(
        self,
        checkpoint_file: str = "./orchestrator_checkpoint.json",
        checkpoint_name: str = "property",
        default_root_directory: str = "./target_property",
        **kwargs,
    ):
        """
        :param target_property_args: general argument structure which is
            specified by individual implementations
        :type target_property_args: dict
        """

        super().__init__()
        self.additional_args = kwargs
        self.checkpoint_file = checkpoint_file
        self.checkpoint_name = checkpoint_name

        self.default_scheduler = scheduler_builder.build(
            'LOCAL',
            {'root_directory': default_root_directory},
        )

        self.restart_property()

    @abstractmethod
    def checkpoint_property(self):
        """
        checkpoint the property module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        pass

    @abstractmethod
    def restart_property(self):
        """
        restart the property module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        pass

    @abstractmethod
    def calculate_property(
        self,
        iter_num: int = 0,
        modified_params: Optional[Dict[str, Any]] = None,
        potential: Optional[Union[str, Potential]] = None,
        scheduler: Optional[Scheduler] = None,
        storage: Optional[Storage] = None,
        **kwargs,
    ):
        """
        Perform analysis to calculate a property of interest.

        Derived classes should list explicit arguments required
        to calculate their properties. This module can utilize
        other modules within the orchestrator to carry out the
        target calculations.

        :param potential: interatomic potential to be used in LAMMPS
        :type potential: str
        :param scheduler: the scheduler for managing job submission, if none
            are supplied, will use the default scheduler defined in this class
            |default| ``None``
        :type scheduler: Scheduler
        :returns: a dictionary with property output, errors, and calc ids as
            a tuple (different indices can correspond to different calc types)
        :rtype: dict
        """
        pass

    @abstractmethod
    def _conduct_sim(
        self,
        sim_params: Dict[str, Any],
        scheduler: Scheduler,
        sim_path: str,
    ) -> int:
        """
        Perform the simulation for the target property calculations

        sim_params is a dictionary of key-value pairs that can
        be used to define various parameters related to conducting
        simulations (e.g. temperature, pressure, random seed, etc..).
        The dictionary is described in the input json file.

        :param sim_params: simulation specific parameters
        :type sim_params: dict
        :param scheduler: the scheduler for managing job submission
        :type scheduler: Scheduler
        :param sim_path: path to perform simulations for
            target property calculations
        :type sim_path: str
        """
        pass

    @abstractmethod
    def calculate_with_error(
        self,
        n_calc: int,
        modified_params: Optional[Dict[str, Any]] = None,
        potential: Optional[Union[str, Potential]] = None,
        scheduler: Optional[Scheduler] = None,
    ):
        """
        Calculate a target property with mean and standard deviation
        Derived classes should list explicit arguments required
        to calculate their properties.

        Mean and standard deviation will be obtained from multiple
        number of calculations (n_calc)

        :param n_calc: total number of calculations to perform
        :type n_calc: int
        :param potential: interatomic potential to be used in LAMMPS
        :type potential: str
        :param scheduler: the scheduler for managing job submission
        :type scheduler: Scheduler
        :returns: mean and standard deviation of the calculated property
        """
        pass
