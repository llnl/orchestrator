from typing import Optional, Union
from ..utils.module_factory import ModuleFactory, ModuleBuilder
from ..utils.exceptions import ModuleAlreadyInFactoryError
from .workflow_base import Workflow

#: default factory for workflows, includes LOCAL
workflow_factory = ModuleFactory(Workflow)


class WorkflowBuilder(ModuleBuilder):
    """
    Constructor for workflows added in the factory

    set the factory to be used for the builder. The default is to use the
    workflow_factory generated at the end of this module. A user defined
    ModuleFactory can optionally be supplied instead.

    :param factory: a workflow factory |default| :data:`workflow_factory`
    :type factory: ModuleFactory
    """

    def __init__(self, factory: Optional[ModuleFactory] = workflow_factory):
        """
        constructor for the WorkflowBuilder, sets the factory to build from

        :param factory: a workflow factory |default| :data:`workflow_factory`
        :type factory: ModuleFactory
        """
        if factory.base_class.__name__ == Workflow.__name__:
            super().__init__(factory)
        else:
            raise Exception('Supplied factory is not for Workflows!')

    def build(
        self,
        workflow_type: str,
        workflow_args: dict[str, Union[str, int, float]],
    ) -> Workflow:
        """
        Return an instance of the specified workflow

        The build method takes the specifier and input arguments to construct
        a concrete workflow instance.

        :param workflow_type: token of a workflow which has been added to the
            factory
        :type workflow_type: str
        :param workflow_args: arguments to control workflow behavior |default|
            ``None``
        :type workflow_args: dict
        :returns: instantiated concrete Workflow
        :rtype: Workflow
        """
        if workflow_args is None:
            workflow_args = {}

        match workflow_type:
            case 'LOCAL':
                from .local import LocalWF
                try:
                    workflow_factory.add_new_module('LOCAL', LocalWF)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'SLURM':
                from .slurm import SlurmWF
                try:
                    workflow_factory.add_new_module('SLURM', SlurmWF)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'LSF':
                from .lsf import LSFWF
                try:
                    workflow_factory.add_new_module('LSF', LSFWF)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'FLUX':
                from .flux import FluxWF
                try:
                    workflow_factory.add_new_module('FLUX', FluxWF)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'AiiDA':
                from .aiida import AiidaWF
                try:
                    workflow_factory.add_new_module('AiiDA', AiidaWF)
                except ModuleAlreadyInFactoryError:
                    pass

        workflow_constructor = self.factory.select_module(workflow_type)
        return workflow_constructor(**workflow_args)


#: workflow builder object which can be imported for use in other modules
workflow_builder = WorkflowBuilder()
