from ..utils.module_factory import ModuleFactory, ModuleBuilder
from ..utils.exceptions import ModuleAlreadyInFactoryError
from .potential_base import Potential

#: default factory for potentials, includes DNN (Behler Parrinello) and KIM
potential_factory = ModuleFactory(Potential)


class PotentialBuilder(ModuleBuilder):
    """
    Constructor for potentials added in the factory

    set the factory to be used for the builder. The default is to use the
    potential_factory generated at the end of this module. A user defined
    ModuleFactory can optionally be supplied instead.

    :param factory: a potential factory |default| :data:`potential_factory`
    :type factory: ModuleFactory
    """

    def __init__(self, factory=potential_factory):
        """
        constructor for the PotentialBuilder, sets the factory to build from

        :param factory: a potential factory |default| :data:`potential_factory`
        :type factory: ModuleFactory
        """
        if factory.base_class.__name__ == Potential.__name__:
            super().__init__(factory)
        else:
            raise Exception('Supplied factory is not for Potentials!')

    def build(self, potential_type, potential_args=None) -> Potential:
        """
        Return an instance of the specified potential

        The build method takes the specifier and input arguments to construct
        a concrete potential instance.

        :param potential_type: token of a potential which has been added to the
            factory
        :type potential_type: str
        :param potential_args: input arguments to instantiate the requested
            potential class
        :type args: dict
        :returns: instantiated concrete Potential
        :rtype: Potential
        """
        if potential_args is None:
            potential_args = {}

        match potential_type:
            case 'nequip_allegro':
                from .nequip_allegro import NequIPAllegroPotential
                try:
                    potential_factory.add_new_module('nequip_allegro',
                                                     NequIPAllegroPotential)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'SNAP':
                from .fitsnap import SNAPPotential
                try:
                    potential_factory.add_new_module('SNAP', SNAPPotential)
                except ModuleAlreadyInFactoryError:
                    pass
            # case 'DNN':
            #     from .dnn import KliffBPPotential
            #     try:
            #         potential_factory.add_new_module('DNN', KliffBPPotential)
            #     except ModuleAlreadyInFactoryError:
            #         pass
            # case 'KIM':
            #     from .kim import KIMPotential
            #     try:
            #         potential_factory.add_new_module('KIM', KIMPotential)
            #     except ModuleAlreadyInFactoryError:
            #         pass
            # case 'ChIMES':
            #     from .chimes import ChIMESPotential
            #     try:
            #         potential_factory.add_new_module('ChIMES',
            # ChIMESPotential)
            #     except ModuleAlreadyInFactoryError:
            #         pass

        potential_constructor = self.factory.select_module(potential_type)
        # check if alternative initialization should be used
        try:
            initialization_from = potential_args.pop('initialize_from')
        except KeyError:
            initialization_from = 'default'
        # use the proper constructor
        if initialization_from == 'kim_id':
            if hasattr(potential_constructor, 'initialize_from_kim'):
                built_class = potential_constructor.initialize_from_kim(
                    **potential_args)
            else:
                raise ValueError('kim_id supplied but this potential type '
                                 'does not support instantiation from KIM ID')
        elif initialization_from == 'potential_files':
            built_class = potential_constructor.initialize_from_files(
                **potential_args)
        elif initialization_from == 'default':
            built_class = potential_constructor(**potential_args)
        else:
            raise ValueError('initialize_from is specified but is not one of '
                             'the supported keys: ["default", "kim_id", '
                             '"potential_files"]')
        built_class.factory_token = potential_type
        return built_class


#: potential builder object which can be imported for use in other modules
potential_builder = PotentialBuilder()
