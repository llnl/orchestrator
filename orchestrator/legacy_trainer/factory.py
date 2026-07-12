from ..utils.module_factory import ModuleFactory, ModuleBuilder
from ..utils.exceptions import ModuleAlreadyInFactoryError
from .trainer_base import Trainer

#: default factory for trainers, includes DNN (kliff) and KLIFF (parametric
#: model)
trainer_factory = ModuleFactory(Trainer)


class TrainerBuilder(ModuleBuilder):
    """
    Constructor for trainers added in the factory

    set the factory to be used for the builder. The default is to use the
    trainer_factory generated at the end of this module. A user defined
    ModuleFactory can optionally be supplied instead.

    :param factory: a trainer factory |default| :data:`trainer_factory`
    :type factory: ModuleFactory
    """

    def __init__(self, factory=trainer_factory):
        """
        constructor for the TrainerBuilder, sets the factory to build from

        :param factory: a trainer factory |default| :data:`trainer_factory`
        :type factory: ModuleFactory
        """
        if factory.base_class.__name__ == Trainer.__name__:
            super().__init__(factory)
        else:
            raise Exception('Supplied factory is not for Trainers!')

    def build(self, trainer_type, trainer_args=None) -> Trainer:
        """
        Return an instance of the specified trainer

        The build method takes the specifier and input arguments to construct
        a concrete trainer instance.

        :param trainer_type: token of a trainer which has been added to the
            factory
        :type trainer_type: str
        :param trainer_args: arguments to control trainer behavior
        :type trainer_args: dict
        :returns: instantiated concrete Trainer
        :rtype: Trainer
        """
        if trainer_args is None:
            trainer_args = {}

        match trainer_type:
            case 'KLIFF':
                from .kliff import ParametricModelTrainer
                try:
                    trainer_factory.add_new_module(
                        'KLIFF',
                        ParametricModelTrainer,
                    )
                except ModuleAlreadyInFactoryError:
                    pass
            case 'DNN':
                from .kliff import DUNNTrainer
                try:
                    trainer_factory.add_new_module('DNN', DUNNTrainer)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'FitSnap':
                from .fitsnap import FitSnapTrainer
                try:
                    trainer_factory.add_new_module('FitSnap', FitSnapTrainer)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'ChIMES':
                from .chimes import ChIMESTrainer
                try:
                    trainer_factory.add_new_module('ChIMES', ChIMESTrainer)
                except ModuleAlreadyInFactoryError:
                    pass

        trainer_constructor = self.factory.select_module(trainer_type)
        built_class = trainer_constructor(**trainer_args)
        built_class.factory_token = trainer_type
        return built_class


#: trainer builder object which can be imported for use in other modules
trainer_builder = TrainerBuilder()
