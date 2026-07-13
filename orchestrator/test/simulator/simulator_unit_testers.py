import json
import numpy as np
from orchestrator.utils.setup_input import init_and_validate_module_type
from orchestrator.utils.data_standard import METADATA_KEY


def simulator_unit_test(input_file: str) -> bool:
    """
    basic test of the simulator module

    :param input_file: input file path with keys to initialize the simulator
    :type input_file: str
    :returns: boolean flag that the function completed execution. Does not
        necessarily indicate a correct output, but is used to determine if the
        pytest should be run
    :rtype: bool
    """
    with open(input_file, 'r') as fin:
        test_inputs = json.load(fin)

    built_simulator = init_and_validate_module_type(
        'simulator',
        test_inputs,
        single_input_dict=True,
    )

    calc_id = built_simulator.run(
        test_inputs.get('path_type'),
        test_inputs.get('model_path'),
        test_inputs.get('template_fill'),
        make_config_path=test_inputs.get('init_conf'),
        make_config_seed=test_inputs.get('random_seed', 42),
    )
    calc_path = built_simulator.default_scheduler.get_job_path(calc_id)
    parsed_atoms = built_simulator.parse_for_storage(calc_path)
    pos = []
    cell = []
    atype = []
    for config in parsed_atoms:
        pos.append(config.positions)
        cell.append(config.get_cell()[:])
        atype.append(config.get_chemical_symbols())
    pos = np.array(pos)
    cell = np.array(cell)
    atype = np.array(atype)

    run_path = parsed_atoms[0].info[METADATA_KEY]['data_source'].rsplit(
        '/', 1)[0]
    outfile = run_path + '/parsed_excerpts.dat'
    with open(outfile, 'w') as fout:
        fout.write(str(pos.shape) + '\n')
        fout.write(str(pos[1][2]) + '\n')
        fout.write(str(cell[[0, -1]]) + '\n')
        fout.write(str(atype.shape) + '\n')
        fout.write(str(atype[0, :3]) + '\n')

    return True
