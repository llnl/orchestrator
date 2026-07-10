from ..utils.module_factory import ModuleFactory, ModuleBuilder
from ..utils.exceptions import ModuleAlreadyInFactoryError
from .property_base import TargetProperty

#: default factory for target property calculator, includes MeltingPoint
target_property_factory = ModuleFactory(TargetProperty)


class TargetPropertyBuilder(ModuleBuilder):
    """
    Constructor for target property calculators added in the factory

    set the factory to be used for the builder. The default is to use
    the target_property_factory generated at the end of this module.
    A user defined ModuleFactory can optionally be supplied instead.

    :param factory: a target property factory |default|
        :data:`target_property_factory`
    :type factory: ModuleFactory
    """

    def __init__(self, factory=target_property_factory):
        """
        constructor for the TargetPropertyBuilder, sets the factory
        to build from

        :param factory: a target property factory |default|
            :data:`target_property_factory`
        :type factory: ModuleFactory
        """
        if factory.base_class.__name__ == TargetProperty.__name__:
            super().__init__(factory)
        else:
            raise Exception(
                'Supplied factory is not for target property calculators!')

    def build(self,
              target_property_type,
              target_property_args=None) -> TargetProperty:
        """
        Return an instance of the specified target property calculator

        The build method takes the specifier and input arguments to construct
        a concrete target property instance.

        :param target_property_type: token of a target property calculator
            which has been added to the factory
        :type target_property_type: str
        :param target_property_args: input arguments to instantiate the
            requested target property class
        :type target_property_args: dict
        :returns: instantiated concrete Target Property Calculator
        :rtype: Target Property Calculator
        """
        if target_property_args is None:
            target_property_args = {}

        match target_property_type:
            case 'MeltingPoint':
                from .melting_point import MeltingPoint
                try:
                    target_property_factory.add_new_module(
                        'MeltingPoint', MeltingPoint)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'ElasticConstants':
                from .elastic_constants import ElasticConstants
                try:
                    target_property_factory.add_new_module(
                        'ElasticConstants', ElasticConstants)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'KIMRun':
                from .kimrun import KIMRun
                try:
                    target_property_factory.add_new_module('KIMRun', KIMRun)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'CalculateLammpsThermo':
                from .calculate_lammps_thermo import CalculateLammpsThermo
                try:
                    target_property_factory.add_new_module(
                        'CalculateLammpsThermo', CalculateLammpsThermo)
                except ModuleAlreadyInFactoryError:
                    pass

        target_property_constructor = self.factory.select_module(
            target_property_type)
        return target_property_constructor(**target_property_args)


#: TargetProperty builder object which can be imported for use in other modules
target_property_builder = TargetPropertyBuilder()
