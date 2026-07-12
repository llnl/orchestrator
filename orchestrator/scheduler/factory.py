from typing import Optional, Union
from ..utils.module_factory import ModuleFactory, ModuleBuilder
from ..utils.exceptions import ModuleAlreadyInFactoryError
from .scheduler_base import Scheduler

#: default factory for schedulers, includes LOCAL
scheduler_factory = ModuleFactory(Scheduler)


class SchedulerBuilder(ModuleBuilder):
    """
    Constructor for schedulers added in the factory

    set the factory to be used for the builder. The default is to use the
    scheduler_factory generated at the end of this module. A user defined
    ModuleFactory can optionally be supplied instead.

    :param factory: a scheduler factory |default| :data:`scheduler_factory`
    :type factory: ModuleFactory
    """

    def __init__(self, factory: Optional[ModuleFactory] = scheduler_factory):
        """
        constructor for the SchedulerBuilder, sets the factory to build from

        :param factory: a scheduler factory |default| :data:`scheduler_factory`
        :type factory: ModuleFactory
        """
        if factory.base_class.__name__ == Scheduler.__name__:
            super().__init__(factory)
        else:
            raise Exception('Supplied factory is not for Schedulers!')

    def build(
        self,
        scheduler_type: str,
        scheduler_args: dict[str, Union[str, int, float]],
    ) -> Scheduler:
        """
        Return an instance of the specified scheduler

        The build method takes the specifier and input arguments to construct
        a concrete scheduler instance.

        :param scheduler_type: token of a scheduler which has been added to the
            factory
        :type scheduler_type: str
        :param scheduler_args: arguments to control scheduler behavior
            |default| ``None``
        :type scheduler_args: dict
        :returns: instantiated concrete Scheduler
        :rtype: Scheduler
        """
        if scheduler_args is None:
            scheduler_args = {}

        match scheduler_type:
            case 'LOCAL':
                from .local import LocalScheduler
                try:
                    scheduler_factory.add_new_module('LOCAL', LocalScheduler)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'SLURM':
                from .slurm import SlurmScheduler
                try:
                    scheduler_factory.add_new_module('SLURM', SlurmScheduler)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'LSF':
                from .lsf import LSFScheduler
                try:
                    scheduler_factory.add_new_module('LSF', LSFScheduler)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'FLUX':
                from .flux import FluxScheduler
                try:
                    scheduler_factory.add_new_module('FLUX', FluxScheduler)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'AiiDA':
                from .aiida import AiidaScheduler
                try:
                    scheduler_factory.add_new_module('AiiDA', AiidaScheduler)
                except ModuleAlreadyInFactoryError:
                    pass

        scheduler_constructor = self.factory.select_module(scheduler_type)
        return scheduler_constructor(**scheduler_args)


#: scheduler builder object which can be imported for use in other modules
scheduler_builder = SchedulerBuilder()
