import json
from typing import Any, Optional, Union
import importlib
from .module_factory import ModuleBuilder

#: list of modules supported for automated setup. :func:`setup_orch_modules`
#: returns modules in this order.
supported_modules = [
    'augmentor',
    'descriptor',
    'oracle',
    'potential',
    'scheduler',
    'score',
    'simulator',
    'storage',
    'target_property',
]

# rudimentary for now, but can be expanded in the future
#: dict of required args for each module type
required_args_dict = {
    'augmentor': ['augmentor_type', 'augmentor_args'],
    'descriptor': ['descriptor_type', 'descriptor_args'],
    'oracle': ['oracle_type', 'oracle_args'],
    'potential': ['potential_type', 'potential_args'],
    'scheduler': ['scheduler_type', 'scheduler_args'],
    'score': ['score_type', 'score_args'],
    'simulator': ['simulator_type', 'simulator_args'],
    'storage': ['storage_type', 'storage_args'],
    'target_property': ['target_property_type'],
}


def read_input(input_file: str) -> dict:
    """
    Reads a JSON input file to specify details of orchestrator modules.

    :param input_file: Path to the input JSON file.
    :type input_file: str
    :returns: Parsed input parameters as a dictionary.
    :rtype: dict
    """
    with open(input_file, 'r') as fin:
        infile = json.load(fin)
    return infile


def setup_orch_modules(jsondict: dict) -> list:
    """
    Initialize the main classes from Orchestrator

    Given the flexibility of the orchestrator, setup only instantiate modules
    which are given in the input file. This setup currently supports the
    :class:`~.Augmentor`, :class:`~.DescriptorBase`, :class:`~.Oracle` and
    :class:`~.AiidaOracle`, :class:`~.Potential`, :class:`~.Scheduler`
    :class:`~.ScoreBase`, :class:`~.Simulator`, :class:`~.Storage`, and
    :class:`~.TargetProperty` modules.

    :param jsonfile: Input arguments parsed by :meth:`read_input` from the
        JSON file.
    :type jsonfile: dict
    :returns: tuple of modules, set to ``None`` if not in input and a dict of
        like-modules if multiple sections are present in the input, i.e.
        'scheduler1', 'scheduler2', 'default_scheduler', ...
    :rtype: list of modules in order of `supported modules`
    """
    return_list = []
    for module_type in supported_modules:
        return_list.append(init_and_validate_module_type(
            module_type, jsondict))

    return return_list


def init_and_validate_module_type(
    module_name: str,
    input_args: dict,
    single_input_dict: Optional[bool] = False,
) -> Any:
    """
    Initialize Orchestrator classes from their sections of the input file

    This function both builds an Orchestrator module(s) and also ensures that
    the minimum keywords are provided.

    :param module_name: name of any Orchestrator module which has a builder
    :type module_name: str
    :param input_args: full input dictionary to initialize `module_name` from.
        Multilpe classes can be specified as long as their token includes the
        base module name.
    :type input_args: dict
    :param single_input_dict: Optional flag if the input arguments are a dict
        of _type and _arg keywords for a single class instance instead of a
        full input dict. |default| ``False``
    :type single_input_dict: bool
    :returns: instantiated Orchestrator class or dictionary of multiply
        specified classes.
    :rtype: Orchestrator class or dict of Orchestrator classes
    """
    # get the specific builder, required args, and all relevant input sections
    module_builders = _get_module_builders(module_name)
    required_args = required_args_dict[module_name]
    if single_input_dict:
        if f'{module_name}_type' not in input_args:
            raise ValueError('single_input_dict was set to True but the input '
                             'does not appear to be a single module dict')
        module_keys = ['dummy']
    else:
        module_keys = [key for key in input_args if module_name in key]

    # initialize module as a dict assuming multiple will be defined
    module = {}
    for module_key in module_keys:
        if single_input_dict:
            module_args = input_args
        else:
            module_args = input_args[module_key]
        _check_required_args(module_args, required_args, module_name)
        # see if multiple builder types exist
        if isinstance(module_builders, dict):
            module_builder = None
            # see if specific module types are specified
            for build_key in module_builders:
                if build_key in module_args[f'{module_name}_type']:
                    module_builder = module_builders[build_key]
                    break
            # not a specific module type, use the default builder
            if module_builder is None:
                module_builder = module_builders['default']
        # only one builder type for this module
        else:
            module_builder = module_builders

        # not all modules have init args. set to empty dict if missing
        module_init_args = module_args.get(f'{module_name}_args', {})
        module[module_key] = module_builder.build(
            module_args[f'{module_name}_type'],
            module_init_args,
        )
    # address case where module type is not present in input
    if len(module_keys) == 0:
        module = None
    # if only one module, return the module itself not a dict of like-modules
    if len(module_keys) == 1:
        module = next(iter(module.values()))

    return module


def _check_required_args(args: dict, required_args: list[str], section: str):
    """
    Checks if required arguments are missing from a section of the input file.

    :param args: Arguments supplied to the constructor.
    :type args: dict
    :param required_args: List of required arguments for the section.
    :type required_args: list of str
    :param section: Name of the section in the input JSON file.
    :type section: str
    :raises ValueError: If a required argument is missing.
    """
    if not isinstance(args, dict):
        raise ValueError(f"Expected a dictionary for section '{section}', got "
                         "{type(args).__name__}.")

    for rarg in required_args:
        if rarg not in args:
            raise ValueError(f'Arg "{rarg}" is required in section {section}')


def _get_module_builders(module_name: str) -> Union[dict, ModuleBuilder]:
    """
    Load and retreive the module builders for a request module

    This function decentralizes imports to reduce load times on environment
    start-up. If multiple builders exist, they are returned as a dictionary,
    with one builder required to be dentoted as the default selection.

    :param module_name: name of the module for which a builder should be loaded
        and returned
    :type module_name: str
    :returns: a single module builder or dict of module builders if
    """

    single_module_map = {
        'augmentor': ('..augmentor', 'augmentor_builder'),
        'descriptor': ('..computer.descriptor', 'descriptor_builder'),
        # 'oracle' handled separately below
        'potential': ('..potential', 'potential_builder'),
        'score': ('..computer.score', 'score_builder'),
        'simulator': ('..simulator', 'simulator_builder'),
        'storage': ('..storage', 'storage_builder'),
        'target_property': ('..target_property', 'target_property_builder'),
        'trainer': ('..trainer', 'trainer_builder'),
        'scheduler': ('..scheduler', 'scheduler_builder'),
    }

    match module_name:
        case 'oracle':
            # first deal with any case where there are multiple builders
            from ..oracle import oracle_builder
            try:
                # AiiDA may not always be installed
                from ..oracle import aiida_oracle_builder
                module_builders = {
                    'default': oracle_builder,
                    'AiiDA': aiida_oracle_builder
                }
            except ImportError:
                # if not installed, only one builder is needed
                module_builders = oracle_builder
        case single_module_name if single_module_name in single_module_map:
            # then deal with cases where we have only a single builder type
            module_path, builder_name = single_module_map[single_module_name]
            # dynamically import module
            builder_module = importlib.import_module(module_path, __package__)
            module_builders = getattr(builder_module, builder_name)
        case _:
            raise ValueError(f"Invalid module choice of '{module_name}', must "
                             f'be one of {supported_modules}')

    return module_builders
