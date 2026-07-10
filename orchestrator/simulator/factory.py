from ..utils.module_factory import ModuleFactory, ModuleBuilder
from ..utils.exceptions import ModuleAlreadyInFactoryError
from .simulator_base import Simulator

simulator_factory = ModuleFactory(Simulator)


class SimulatorBuilder(ModuleBuilder):
    """
    Constructor for simulators added in the factory

    set the factory to be used for the builder. The default is to use the
    simulator_factory generated at the end of this module. A user defined
    ModuleFactory can optionally be supplied instead.

    :param factory: a simulator factory |default| :data:`simulator_factory`
    :type factory: ModuleFactory
    """

    def __init__(self, factory=simulator_factory):
        """
        constructor for the SimulatorBuilder, sets the factory to build from

        :param factory: a simulator factory |default| :data:`simulator_factory`
        :type factory: ModuleFactory
        """
        if factory.base_class.__name__ == Simulator.__name__:
            super().__init__(factory)
        else:
            raise Exception('Supplied factory is not for Simulators!')

    def build(self, simulator_type, simulator_args=None):
        """
        Return an instance of the specified simulator

        The build method takes the specifier and input arguments to construct
        a concrete simulator instance.

        :param simulator_type: token of a simulator which has been added to the
            factory
        :type simulator_type: str
        :param simulator_args: dictionary of parameters to instantiate the
            Simulator, such as code_path, elements, and input_template
        :type code_path: dict
        :returns: instantiated concrete Simulator
        :rtype: Simulator
        """
        if simulator_args is None:
            simulator_args = {}

        match simulator_type:
            case 'LAMMPS':
                from .lammps import LAMMPSSimulator
                try:
                    simulator_factory.add_new_module('LAMMPS', LAMMPSSimulator)
                except ModuleAlreadyInFactoryError:
                    pass

        simulator_constructor = self.factory.select_module(simulator_type)
        return simulator_constructor(**simulator_args)


#: simulator builder object which can be imported for use in other modules
simulator_builder = SimulatorBuilder()
