import numpy as np
import random
import ase.io
from ase import Atoms
from glob import glob
from os.path import getmtime, split, join
from typing import Union, Optional, List, Dict, Any, Tuple
from ..simulator import simulator_builder
from ..potential import Potential
from ..utils.exceptions import AnalysisError, DensityOOBError
from . import TargetProperty
from ..workflow import Workflow
from ..storage import Storage
from orchestrator.target_property.analysis import AnalyzeLammpsLog
from ..utils.restart import restarter
from ..utils.isinstance import isinstance_no_import


class MeltingPoint(TargetProperty):
    """
    Class for determining melting point of a material

    Explore temperature range and determine final temperature that
    results in solid/liquid phase. Simulation parameters and required
    paths are read from json file.

    :param simulator_type: name of the simulator to perform simulations
    :type simulator_type: str
    :param simulator_path: path to the simulator executable
    :type simulator_path: str
    :param elements: list of elements which are present in the simulation
    :type elements: list
    :param input_template: LAMMPS template input files. This is a dictionary
        of key-value pairs that can be used to define any simulation input
        template files required for conducting simulations (e.g. one input
        simulations).
    :type input_template: dict
    :param job_details: optional parameters for running the job
    :type job_details: dict
    :param npt_job_details: optional parameters for running the NPT job
    :type npt_job_details: dict
    :param nph_job_details: optional parameters for running the NPH job
    :type nph_job_details: dict
    """

    def __init__(
        self,
        simulator_type: str,
        simulator_path: str,
        input_template: str,
        job_details: dict,
        elements: list = [],
        npt_job_details: dict = {},
        nph_job_details: dict = {},
        **kwargs: Any,
    ):
        """
        Initialization of the MeltingPoint class with args dict

        :param simulator_type: name of the simulator to perform simulations
        :type simulator_type: str
        :param simulator_path: path to the simulator executable
        :type simulator_path: str
        :param elements: list of elements which are present in the simulation
        :type elements: list
        :param input_template: LAMMPS template input file. This is a dictionary
            of key-value pairs that can be used to define any simulation input
            template files required for conducting simulations (e.g. one input
            template for npt simulations and another inpute template for nph
            simulations).
        :type input_template: dict
        :param job_details: optional parameters for running the job
        :type job_details: dict
        :param npt_job_details: optional parameters for running the NPT job
        :type npt_job_details: dict
        :param nph_job_details: optional parameters for running the NPH job
        :type nph_job_details: dict
        """

        self.npt_job_details = npt_job_details
        self.npt_job_details = {**job_details, **npt_job_details}
        self.nph_job_details = nph_job_details
        self.nph_job_details = {**job_details, **nph_job_details}

        self.input_template_npt = input_template['NPT']
        self.input_template_nph = input_template['NPH']

        npt_simulator_args = {
            'code_path': simulator_path,
            'elements': elements,
            'input_template': self.input_template_npt,
        }
        self.built_simulator_npt = simulator_builder.build(
            simulator_type,
            npt_simulator_args,
        )

        nph_simulator_args = {
            'code_path': simulator_path,
            'elements': elements,
            'input_template': self.input_template_nph,
        }
        self.built_simulator_nph = simulator_builder.build(
            simulator_type,
            nph_simulator_args,
        )
        self.progress_flag = 'init'
        self.current_state = {}
        self.outstanding_npt = []
        self.outstanding_nph = None
        self.npt_calcs = []
        self.nph_calcs = []
        super().__init__(**kwargs)

    def checkpoint_property(self) -> None:
        """
        checkpoint the property module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        save_dict = {
            self.checkpoint_name: {
                'progress_flag': self.progress_flag,
                'current_state': self.current_state,
                'npt_calcs': self.npt_calcs,
                'nph_calcs': self.nph_calcs
            }
        }
        restarter.write_checkpoint_file(self.checkpoint_file, save_dict)

    def restart_property(self) -> None:
        """
        restart the property module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        restart_dict = restarter.read_checkpoint_file(
            self.checkpoint_file,
            self.checkpoint_name,
        )
        self.progress_flag = restart_dict.get('progress_flag',
                                              self.progress_flag)
        self.current_state = restart_dict.get('current_state',
                                              self.current_state)
        self.npt_calcs = restart_dict.get('npt_calcs', self.npt_calcs)
        self.nph_calcs = restart_dict.get('nph_calcs', self.nph_calcs)
        if len(self.npt_calcs) > 0:
            self.outstanding_npt = self.npt_calcs[-1]
        if len(self.nph_calcs) > 0:
            self.outstanding_nph = self.nph_calcs[-1]
        if self.progress_flag != 'init':
            if self.progress_flag == 'done':
                # restart information exists but the last calculation ended
                self.restart = False
                self.npt_calcs = []
                self.nph_calcs = []
            else:
                # restart information exists and we actually want to restart
                self.restart = True
        else:
            # no restart information exists
            self.restart = False

    def calculate_property(
        self,
        path_type: str,
        melting_calc_params: dict,
        sim_params: dict,
        model_path: str = None,
        random_seed_use: bool = False,
        iter_num: int = 0,
        modified_params: Optional[Dict[str, Any]] = None,
        potential: Optional[Union[str, Potential]] = None,
        workflow: Optional[Workflow] = None,
        storage: Optional[Storage] = None,
        **kwargs,
    ) -> Dict[str, Union[float, Tuple[List[int], List[int]]]]:
        """
        Find final equilibrium temperature for the melting point

        Iterate over range of temperatures between
        temp_min and temp_max, and adjust the temperature range
        depending on the analysis of liquid and solid phases.
        Continue spawning simulations with different temperatures
        until difference between maximum and minimum temperatures
        reduces below a set threshold (temp_threshold). After finding
        equilibrium temperature, perform NPH simulations and check if
        two phases exist by analyzing q_x parameter. Adjust the temperature
        until getting two phases.The extract_q and extract_msd methods
        from AnalyzeLammpsLog calculate local bond parameter q_x and mean
        square displacement, respectively. Additional parameters used by
        this method are temp_threshold (float, threshold temperature
        difference between temp_max and temp_min to determine the
        convergence of the final equilibriumm temperature), temp_min
        (float, min temperature guess), temp_max (float, max temperature
        guess), temp_incr (float, temperature increment for screening
        temperature in NPH stage),num_temp (int, number of temperatures
        to test between minimum and maximum temperature guesses,
        max_iter (int, maximum number of iterations for screening
        temperature in NPH stage). These are defined by
        melting_calc_params and set at class instantiation.
        This method also uses temp (float, simulation temperature),
        press (float, simulation pressure), which are defined
        by sim_params and set at class instantiation.

        :param path_type: path to perform thermo calculations
        :type path_type: str
        :param model_path: path to store the potential file
        :type model_path: str
        :param melting_calc_params: melting point estimation specific
            parameters
        :type melting_calc_params: dict
        :param sim_params: simulation specific parameters
        :type sim_params: dict
        :param random_seed_use: option to use random seed in the simulation
        :type random_seed_use: boolean
        :param iter_num: iteration number of the calculation of melting point
            if executed over multiple iterations
        :type iter_num: int
        :param modified_params: simulation parameters to modify the initially
            provided values, including interatomic potentials, simulation
            temperature and pressure
        :type modified_params: dict
        :param workflow: the workflow for managing job submission
        :type workflow: Workflow
        :param potential: interatomic potential to be used in LAMMPS. Can be
            either the string of a KIM potential available via the KIM API or
            a Potential object created by the Orchestrator.
        :type potential: str or Potential
        :param workflow: the workflow for managing job submission
        :type workflow: Workflow
        :param storage: the storage module where the configurations will be
            saved (not currently used).
        :type storage: Storage
        :returns: dictionary with the final temperature that results in solid/
            liquid phases as the property_value, std based on time-averaging
            for the property_std, and a tuple of the NPT calculation ID list
            and the final NPH calculation ID as the calc_ids
        :rtype: dict
        """
        # ensure restart is properly read
        self.restart_property()

        if workflow is None:
            workflow = self.default_wf

        if modified_params is not None:
            sim_params['temp'] = modified_params['temp']
            sim_params['press'] = modified_params['press']

        if isinstance_no_import(potential, 'Potential'):
            if hasattr(potential, 'install_potential_in_kim_api') and callable(
                    potential.install_potential_in_kim_api):
                module_name = self.__class__.__name__
                save_root = workflow.make_path(
                    module_name,
                    'potential_for_melting_point',
                )
                potential_name = potential.kim_id
                potential.install_potential_in_kim_api(
                    potential_name=potential_name, save_path=save_root)
                model_path = f'{save_root}/{potential_name}'

                sim_params['potential'] = potential_name
            else:
                try:
                    potential_string = potential.get_lammps_commands()
                    sim_params['potential'] = potential_string
                    self.logger.info(
                        'Using generated LAMMPS commands for potential')
                except (ValueError, FileNotFoundError) as e:
                    self.logger.warning(
                        f'Could not generate LAMMPS commands: {e}. ')
                    raise
        else:
            if potential is not None:
                sim_params['potential'] = potential

        timestep = sim_params.get('timestep', 0.001)
        lattice_param = sim_params.get('lattice_param', 'unknown')
        assert lattice_param != 'unknown' and lattice_param > 0, \
            "Lattice parameter is not given or assigned to a wrong value"

        temp_threshold = melting_calc_params.get('temp_thresh', 1000.0)
        assert temp_threshold > 0, \
            "Temperature threshold must be a positive value"
        temp_min = melting_calc_params.get('temp_min', 0.0)
        assert isinstance(temp_min, float) or isinstance(temp_min, int), \
            "Minimum temperature must be a number"
        temp_max = melting_calc_params.get('temp_max', 12000.0)
        assert isinstance(temp_max, float) or isinstance(temp_max, int), \
            "Maximum temperature must be a number"
        num_temp = melting_calc_params.get('num_temp', 4)
        assert num_temp > 0, "Number of temperatures must be a positive value"
        temp_incr = melting_calc_params.get('temp_incr', 100.0)
        assert temp_incr > 0, "Temperature threshold must be a positive value"
        exp_den_solid = melting_calc_params.get('exp_den_solid', 'none')
        assert exp_den_solid != 'none' and exp_den_solid > 0, \
            "Exp. solid density is not given or assigned to a wrong value"
        exp_den_liquid = melting_calc_params.get('exp_den_liquid', 'none')
        assert exp_den_liquid != 'none' and exp_den_liquid > 0, \
            "Exp. liquid density is not given or assigned to a wrong value"
        den_tol = melting_calc_params.get('den_tol', 10.0)
        assert den_tol > 0, "Density tolerance must be a positive value"
        max_iter = melting_calc_params.get('max_iter', 4)
        assert max_iter > 0, "Maximum iterations must be a positive value"
        eps_dbscan = melting_calc_params.get('eps_dbscan', 0.025)
        assert eps_dbscan != 'none' and eps_dbscan > 0, \
            "Eps parameter is not set or assigned to a wrong value"
        temp_final = None
        calc_id_error = []

        if not self.restart or self.progress_flag == 'npt':
            if self.progress_flag == 'npt':
                # if progress flag is npt, current state is guaranteed to have
                # temp_{min,max} keys
                temp_min = self.current_state['temp_min']
                temp_max = self.current_state['temp_max']
                self.logger.info('Temp bounds set from checkpoint, waiting for'
                                 f' calcs {self.npt_calcs} to finish')

            self.logger.info(f'Initial min and max temperatures are: '
                             f'{temp_min} K and {temp_max} K')
            phase = 'unknown'

            while (temp_max - temp_min) > temp_threshold:
                temp_list = np.linspace(temp_min, temp_max, num_temp + 2)
                self.logger.info(f'Temperatures to test : {temp_list}')
                if len(self.outstanding_npt) == 0:
                    for temp in temp_list:
                        sim_params['temp'] = temp
                        calc_id = self._conduct_sim(
                            model_path,
                            sim_params,
                            workflow,
                            path_type + '/' + str(iter_num) + '/NPT',
                            random_seed_use=random_seed_use,
                        )
                        self.outstanding_npt.append(calc_id)
                    # checkpoint after the batch of calcs are submitted
                    self.npt_calcs.append(self.outstanding_npt)
                    self.current_state = self.current_state | {
                        'temp_min': temp_min,
                        'temp_max': temp_max,
                    }
                    self.progress_flag = 'npt'
                    self.checkpoint_property()

                workflow.block_until_completed(self.outstanding_npt)

                phase_list = []
                q_list = []

                for c_id in self.outstanding_npt:
                    run_path_npt = workflow.get_job_path(c_id)
                    log_npt = run_path_npt + '/' + 'lammps.out'

                    if 'ERROR: Lost atoms:' in open(log_npt).read():
                        self.logger.info(f'Lost atoms error for: {c_id}')
                        calc_id_error.append(c_id)

                if len(calc_id_error) > 0:
                    traj_files = glob(f'{run_path_npt}/*.lammpstrj')
                    file_times = [getmtime(f) for f in traj_files]
                    newest_file = split(traj_files[np.argmax(file_times)])[1]
                    results_dict = {
                        'property_value': newest_file,
                        'property_std': None,
                        'calc_ids': calc_id_error,
                        'success': False,
                    }
                    return results_dict

                for c_id in self.outstanding_npt:
                    run_path_npt = workflow.get_job_path(c_id)
                    log_msd_npt = run_path_npt + '/' + 'log_msd.dat'
                    q_profile_npt = run_path_npt + '/' + 'q_profile.dat'

                    two_phase_flag, two_phase_temp, two_phase_temp_std, \
                        label_unique, average_q_npt, \
                        total_atoms = AnalyzeLammpsLog.extract_q(
                            [run_path_npt, log_msd_npt, q_profile_npt,
                             str(eps_dbscan)])

                    phase = AnalyzeLammpsLog.extract_msd(
                        [log_msd_npt,
                         str(timestep),
                         str(lattice_param)])
                    _, _, density_npt, _ = AnalyzeLammpsLog.extract_property(
                        [log_msd_npt, 'Density'])

                    self.check_density(phase, density_npt, exp_den_solid,
                                       exp_den_liquid, den_tol, c_id)
                    phase_list.append(phase)
                    q_list.append(np.mean(average_q_npt))

                # done with the npt calcs on this iteration, reset for looping
                self.outstanding_npt = []
                assert 'solid' in phase_list, ('All resulting phases are '
                                               'liquid. Please adjust Tmin and'
                                               ' Tmax')
                assert 'liquid' in phase_list, ('All resulting phases are '
                                                'solid. Please adjust Tmin and'
                                                ' Tmax')
                index_solid = [
                    i for i in range(len(phase_list))
                    if phase_list[i] == 'solid'
                ]
                index_liquid = [
                    i for i in range(len(phase_list))
                    if phase_list[i] == 'liquid'
                ]
                max_index_solid = max(index_solid)
                min_index_solid = min(index_solid)
                min_index_liquid = min(index_liquid)
                max_index_liquid = max(index_liquid)
                if min_index_liquid > max_index_solid:
                    temp_max = temp_list[min_index_liquid]
                    temp_min = temp_list[max_index_solid]
                    q_liquid = q_list[max_index_liquid]
                    q_solid = q_list[min_index_solid]
                    self.current_state = self.current_state | {
                        'temp_min': temp_min,
                        'temp_max': temp_max,
                    }
                    # checkpoint after analysis is done
                    self.checkpoint_property()
                else:
                    raise AnalysisError('Phases are not in right order, please'
                                        ' check msd analysis')

                self.logger.info(f'New minimum and maximum temperatures are: '
                                 f'{temp_min} K and {temp_max} K ')

        # this block will execute if restart == True and progress == nph
        # or just naturally once the previous block ends
        two_phase_flag = False
        if self.progress_flag == 'nph':
            # if progress flag is nph, current state is guaranteed to have
            # n_iter and temp_estimate keys
            n_iter = self.current_state['n_iter']
            temp_estimate = self.current_state['temp_estimate']
            self.logger.info(f'Restarting at n_iter = {n_iter} (temp_estimate '
                             f' = {temp_estimate})')
        else:
            n_iter = 0
            temp_estimate = temp_min

        while not two_phase_flag:
            if self.outstanding_nph is None:
                if n_iter == 0:
                    temp_estimate = temp_estimate
                    average_q_nph = 0
                else:
                    if abs(np.mean(average_q_nph)
                           - q_solid) < abs(np.mean(average_q_nph) - q_liquid):
                        temp_estimate += temp_incr
                    else:
                        temp_estimate -= temp_incr

                sim_params['temp'] = temp_estimate
                self.logger.info(f'Iteration {n_iter}: Starting NPH simulation'
                                 ' with an estimated equilibrium temperature: '
                                 f' {temp_estimate} K')
                self.outstanding_nph = self._conduct_sim(
                    model_path,
                    sim_params,
                    workflow,
                    path_type + '/' + str(iter_num) + '/NPH',
                    random_seed_use=random_seed_use,
                )
                self.nph_calcs.append(self.outstanding_nph)
                # checkpoint after the batch of calc is submitted
                self.current_state = self.current_state | {
                    'n_iter': n_iter,
                    'temp_estimate': temp_estimate,
                }
                self.progress_flag = 'nph'
                self.checkpoint_property()

            workflow.block_until_completed(self.outstanding_nph)

            run_path_nph = workflow.get_job_path(self.outstanding_nph)

            log_nph = run_path_nph + '/' + 'lammps.out'

            if 'ERROR: Lost atoms:' in open(log_nph).read():
                calc_id_error.append(self.outstanding_nph)
                self.logger.info(
                    f'Lost atoms error for: {self.outstanding_nph}')

            if len(calc_id_error) > 0:
                traj_files = glob(f'{run_path_nph}/*.lammpstrj')
                file_times = [getmtime(f) for f in traj_files]
                newest_file = split(traj_files[np.argmax(file_times)])[1]
                results_dict = {
                    'property_value': newest_file,
                    'property_std': None,
                    'calc_ids': calc_id_error,
                    'success': False,
                }
                return results_dict

            self.outstanding_nph = None
            q_profile_nph = run_path_nph + '/' + 'q_profile.dat'
            log_msd_nph = run_path_nph + '/' + 'log_msd.dat'
            two_phase_flag, two_phase_temp, two_phase_temp_std, \
                label_unique, average_q_nph, \
                total_atoms = AnalyzeLammpsLog.extract_q(
                    [run_path_nph, log_msd_nph, q_profile_nph,
                     str(eps_dbscan)])

            _, _, density_nph, _ = AnalyzeLammpsLog.extract_property(
                [log_msd_nph, 'Density'])
            density_upper = (exp_den_solid + exp_den_liquid) / 2 + den_tol
            density_lower = (exp_den_solid + exp_den_liquid) / 2 - den_tol
            if density_nph < density_upper and density_nph > density_lower:
                self.logger.info(
                    f'Overall density ({density_nph} g/cm^3) is in expected'
                    f' range, continue melting point calculations ')
            else:
                raise AnalysisError(
                    'Overall density for two phase is out of expected range')

            for i in range(0, len(label_unique)):
                self.logger.info(
                    f'Phase {label_unique[i]} includes {total_atoms[i]} atoms '
                    f'with an average order parameter of {average_q_nph[i]}')

            if not two_phase_flag:
                n_iter += 1
                if n_iter > max_iter or temp_estimate >= temp_max:
                    self.logger.info('Two phase cannot be generated '
                                     'please check your settings. '
                                     'Exiting the melting calculations')
                    temp_final = -1
                    temp_std_final = -1
                    break
            else:
                temp_final = two_phase_temp
                temp_std_final = two_phase_temp_std
                self.logger.info(f'Final equilibrium temperature that results '
                                 f'in solid/liquid phases is: {temp_final} K '
                                 f'with a std of {two_phase_temp_std} K ')

        for key in ['temp_min', 'temp_max', 'n_iter', 'temp_estimate']:
            _ = self.current_state.pop(key, None)
        self.progress_flag = 'done'
        return_npt = self.npt_calcs
        self.npt_calcs = []
        return_nph = self.nph_calcs
        self.nph_calcs = []
        self.checkpoint_property()

        # return results
        results_dict = {
            'property_value': temp_final,
            'property_std': temp_std_final,
            'calc_ids': (return_npt, return_nph),
            'success': True,
        }
        return results_dict

    def _conduct_sim(
        self,
        model_path: str,
        sim_params: Dict[str, Any],
        workflow: Workflow,
        sim_path: str,
        random_seed_use: bool = False,
    ) -> Union[int, str]:
        """
        Launch a single LAMMPS simulation for the target property calculations

        This method prepares and submit a simulation using the LAMMPS simulator
        configured at class initialization. When multiple simulations are
        requested, this method is called repeatedly by ``calculate_property``.
        The method fills the LAMMPS input template, submits the job through the
        provided workflow, and returns a calculation ID that can be used to
        track job completion and retrieve simulation outputs.

        Additional parameters used by this method are temp (float, simulation
        temperature), press (float, simulation pressure), ice_temp (float, ice
        temperature), below_temp_diff (float, temperature to subtract to define
        below temperature), above_temp_diff (float, temperature to add to
        define above temperature), farabove_temp_diff (float, temperature to
        add to define far above temperature), units (string, units to be used
        in LAMMPS simulations), atom_style (string, atom style name),
        pair_style (string, pair style name), potential (string, potential
        name), element (string, type of element), mass (float, mass of the
        element), lattice (string, lattice type), lattice_param (float, lattice
        spacing), lx, ly, lz (int, lattice numbers in x, y and z directions),
        timestep (float, timestep size in time units), q_num_neigh (int,
        number of nearest neighbors for q_x calculations), nph_steps (int,
        number of steps for the NPH stage) which are defined by sim_params
        and set at class instantiation. These parameters are modified by
        calculate_property method on-the-fly and supplied to this method for
        setting simulation temperature and pressure.

        :param model_path: path to store the potential file
        :type model_path: str
        :param sim_params: simulation specific parameters containing all
            the parameters mentioned above
        :type sim_params: dict
        :param workflow: the workflow for managing job submission
        :type workflow: Workflow
        :param sim_path: path to perform simulations for
            target property calculations
        :type sim_path: str
        :param random_seed_use: option to use random seed in the simulation.
            If True, generates a random seed between 1 and 10000; otherwise
            uses a fixed seed of 999
        :type random_seed_use: bool
        :returns: calculation ID corresponding to a spawned simulation
        :rtype: int, str
        """

        press = sim_params.get('press', 0.0)
        temp = sim_params.get('temp', 298.15)
        ice_temp = sim_params.get('ice_temp', 0.0)
        below_temp_diff = sim_params.get('below_temp_diff', 100.0)
        above_temp_diff = sim_params.get('above_temp_diff', 100.0)
        farabove_temp_diff = sim_params.get('farabove_temp_diff', 2000.0)
        units = sim_params.get('units', 'metal')
        atom_style = sim_params.get('atom_style', 'atomic')
        pair_style = sim_params.get('pair_style', 'unknown')
        assert pair_style != 'unknown' and isinstance(pair_style, str), \
            "Pair style is not given or assigned to wrong type"
        potential = sim_params.get('potential', 'unknown')
        assert potential != 'unknown' and isinstance(potential, str), \
            "Potential is not given or assigned to wrong type"
        element = sim_params.get('element', 'unknown')
        assert element != 'unknown' and isinstance(element, str), \
            "Element is not given or assigned to wrong type"
        mass = sim_params.get('mass', 'unknown')
        assert mass != 'unknown' and mass > 0, \
            "Mass is not given or assigned to a wrong value"
        lattice = sim_params.get('lattice', 'unknown')
        assert lattice != 'unknown' and isinstance(lattice, str), \
            "Lattice is not given or assigned to wrong type"
        lattice_param = sim_params.get('lattice_param', 'unknown')
        assert lattice_param != 'unknown' and lattice_param > 0, \
            "Lattice parameter is not given or assigned to a wrong value"
        l_x_npt = sim_params.get('l_x_npt', 10)
        l_y_npt = sim_params.get('l_y_npt', 10)
        l_z_npt = sim_params.get('l_z_npt', 10)
        l_x_nph = sim_params.get('l_x_nph', 10)
        l_y_nph = sim_params.get('l_y_nph', 10)
        l_z_nph = sim_params.get('l_z_nph', 70)
        timestep = sim_params.get('timestep', 0.001)
        order_parameter = sim_params.get('order_parameter', 6)
        q_num_neigh = sim_params.get('q_num_neigh', 'unknown')
        assert q_num_neigh != 'unknown' and q_num_neigh > 0, \
            "NNN for q parameter is not given or assigned to a wrong value"
        npt_steps = sim_params.get('npt_steps', 500000)
        nph_steps = sim_params.get('nph_steps', 100000)

        if random_seed_use is True:
            random_seed = random.randint(1, 10000)
        else:
            random_seed = 999

        template_fill = {
            'temperature': temp,
            'pressure': press,
            'ice_temp': ice_temp,
            'below_temp_diff': below_temp_diff,
            'above_temp_diff': above_temp_diff,
            'farabove_temp_diff': farabove_temp_diff,
            'units': units,
            'atom_style': atom_style,
            'seed': random_seed,
            'pair_style': pair_style,
            'potential': potential,
            'element': element,
            'mass': mass,
            'lattice': lattice,
            'lattice_param': lattice_param,
            'l_x_npt': l_x_npt,
            'l_y_npt': l_y_npt,
            'l_z_npt': l_z_npt,
            'l_x_nph': l_x_nph,
            'l_y_nph': l_y_nph,
            'l_z_nph': l_z_nph,
            'timestep': timestep,
            'order_parameter': order_parameter,
            'q_num_neigh': q_num_neigh,
            'npt_steps': npt_steps,
            'nph_steps': nph_steps
        }

        if 'NPT' in sim_path:
            self.npt_job_details[
                'input_file_name'] = self.input_template_npt.rsplit('/', 1)[1]
            calc_id = self.built_simulator_npt.run(
                sim_path,
                model_path,
                template_fill,
                workflow=workflow,
                job_details=self.npt_job_details,
            )
        elif 'NPH' in sim_path:
            self.nph_job_details[
                'input_file_name'] = self.input_template_nph.rsplit('/', 1)[1]
            calc_id = self.built_simulator_nph.run(
                sim_path,
                model_path,
                template_fill,
                workflow=workflow,
                job_details=self.nph_job_details,
            )
        else:
            self.logger.info('Ensemble type has not been implemented '
                             'for melting point calculations')
            exit()

        return calc_id

    def calculate_with_error(
        self,
        num_calculations: int = 1,
        modified_params: Optional[Dict[str, Any]] = None,
        potential: Optional[Union[str, Potential]] = None,
        workflow: Optional[Workflow] = None,
    ):
        """
        Calculate a target property with mean and standard deviation

        Mean and standard deviation will be obtained from multiple
        number of calculations (num_calculations). This method automatically
        enables random_seed_use for all calculations to ensure statistical
        independence between runs.

        :param num_calculations: total number of calculations to perform
        :type num_calculations: int
        :param modified_params: simulation parameters to modify the initially
            provided values, including interatomic potentials, simulation
            temperature and pressure (not currently used, always None)
        :type modified_params: dict
        :param potential: interatomic potential to be used in LAMMPS
            (not currently used, always None)
        :type potential: str or Potential
        :param workflow: the workflow for managing job submission
        :type workflow: Workflow
        :returns: dictionary with the average final temperature that results
            in solid/liquid phases as the property_value, std from the
            num_calculations outputs for the property_std, and a tuple with
            an empty NPT list and all the final NPH calculation IDs as the
            calc_ids
        :rtype: dict
        """
        if workflow is None:
            workflow = self.default_wf

        # calculate_with_error requires random_seed_use to be True
        random_seed_use = True

        melt_temp_ave = None
        melt_temp_std = None
        # if progress had been made, populate these values from restart dict
        # otherwise set to default/starting values
        start_iteration = self.current_state.get('iter_num', 0)
        calc_id_nph_list = self.current_state.get('calc_list', [])
        melt_temp_list = self.current_state.get('temp_list', [])
        if self.progress_flag == 'error_loop':
            # we're restarting without any outstanding calculate_property
            # running, so we don't want to have it try to restart from the last
            # calculation it saved
            self.restart = False

        for i in range(start_iteration, num_calculations):
            results_dict = self.calculate_property(
                iter_num=i,
                random_seed_use=random_seed_use,
                modified_params=None,
                potential=None,
                workflow=workflow,
            )
            melt_temp = results_dict['property_value']
            melt_std = results_dict['property_std']
            c_ids = results_dict['calc_ids']
            if isinstance(melt_temp, float):
                melt_temp_list.append((melt_temp, melt_std))
                # note the final calculation from NPH runs
                calc_id_nph_list.append(c_ids[1][-1])
            self.progress_flag = 'error_loop'
            self.npt_calcs = None
            self.nph_calc = None
            # don't need to merge state dict with conduct_sim quantities
            self.current_state = {
                'temp_list': melt_temp_list,
                'calc_list': calc_id_nph_list,
                'iter_num': i,
            }
            self.checkpoint_property()
            raise Exception('try restart from here')

        self.logger.info(f'Final melt/std list is: {melt_temp_list}')
        if len(melt_temp_list) > 1:
            melt_temp_ave = np.mean([t[0] for t in melt_temp_list])
            melt_temp_std = np.std([t[0] for t in melt_temp_list])
        else:
            self.logger.info(
                'Number of successful melting temperature calculations is '
                'less than two. Mean and std cannot be estimated')
            exit()

        self.logger.info(
            f'Ave melting temp:{melt_temp_ave} with std:{melt_temp_std}')
        self.progress_flag = 'init'
        self.npt_calcs = None
        self.nph_calc = None
        self.current_state = {}
        self.checkpoint_property()

        # return results
        results_dict = {
            'property_value': melt_temp_ave,
            'property_std': melt_temp_std,
            'calc_ids': ([], calc_id_nph_list),
        }
        return results_dict

    def save_configurations(
        self,
        calc_ids: Union[int, List[int]],
        storage: Storage,
        dataset_handle: str,
        workflow: Workflow,
    ) -> str:
        """
        save configurations generated by the melting point module

        :param calc_ids: single or list of ``calc_ids`` associated with
            simulator jobs. The path is extracted from the
            :class:`~orchestrator.workflow.workflow_base.JobStatus`.
        :type calc_ids: list of int or int
        :param storage: storage module that hosts the dataset
        :type storage: Storage
        :param dataset_handle: handle for the dataset where configs will be
            stored
        :type dataset_handle: str
        :param workflow: the workflow that managed job submission for the
            provided calc_id
        :type workflow: Workflow
        :returns: updated dataset_handle including the new configurations
        :rtype: str
        """
        # the save configuration simulator method is agnostic to the input
        # template, so nph/npt will not make a difference
        return self.built_simulator_nph.save_configurations(
            calc_ids,
            storage,
            dataset_handle,
            workflow,
        )

    def check_density(
        self,
        phase: str,
        density: float,
        exp_den_solid: float,
        exp_den_liquid: float,
        den_tol: float,
        calc_id: int,
    ) -> None:
        """
        check if density is within an expected range during npt simulations

        :param phase: phase detected in the simulation (e.g. solid, liquid)
        :type phase: str
        :param density: density calculated from the simulations
        :type density: float
        :param exp_den_solid: expected density of the material from the
            experiments. This does not need to be exact value, especially
            if the experimental value is not available
        :type exp_den_solid: float
        :param exp_den_liquid: expected density of the material from the
            experiments. This does not need to be exact value, especially
            if the experimental value is not available
        :type exp_den_liquid: float
        :param den_tol: density tolerance to allow calculated density to
            deviate from the experimental value
        :type den_tol: float
        :param calc_id: calculation id associated with a simulator job
        :type calc_id: int
        """

        if phase == 'solid':
            if (density < exp_den_solid + den_tol) and (density > exp_den_solid
                                                        - den_tol):
                self.logger.info(
                    f'Solid density ({density} g/cm^3) is in expected'
                    f' range, continue melting point calculations ')
            else:
                self.failed_job_id = calc_id
                raise DensityOOBError(
                    'density for solid phase is out of expected range')
        elif phase == 'liquid':
            if (density < exp_den_liquid
                    + den_tol) and (density > exp_den_liquid - den_tol):
                self.logger.info(
                    f'Liquid density ({density} g/cm^3) is in expected'
                    f' range, continue melting point calculations ')
            else:
                self.failed_job_id = calc_id
                raise DensityOOBError(
                    'density for liquid phase is out of expected range')
        else:
            raise AnalysisError(
                'No phase has been detected, please check msd analysis')

    @staticmethod
    def sample_configs(
        beg: int,
        end: int,
        step: int,
        elements_list: list[str],
        traj_name: Optional[str] = 'dump.lammpstrj',
        calc_ids: Optional[list[int]] = None,
        workflow: Optional[Workflow] = None,
        in_paths: Optional[list[str]] = None,
    ) -> list[Atoms]:
        """
        sample configurations from the NPT or NPH simulations

        :param beg: first frame of the trajectory to start sampling
        :type beg: int
        :param end: final frame of the trajectory for sampling
        :type end: int
        :param step: frequency of sampling configurations from beg to end
        :type step: int
        :param elements_list: list of elements in same order that they were
            passed to the simulator. Used to correctly assign chemical IDs to
            extracted ASE Atoms
        :type elements_list: list of str
        :param traj_name: name of the trajectory used for sampling
            |default| 'dump.lammpstrj'
        :type traj_name: str
        :param calc_ids: list of ``calc_ids`` associated with
            simulator jobs. The path is extracted from the
            :class:`~orchestrator.workflow.workflow_base.JobStatus`.
            |default| ``None``
        :type calc_ids: list of int
        :param workflow: the workflow that managed job submission for the
            provided calc_id. Required if calc_ids are given. |default|
            ``None``
        :type workflow: Workflow
        :param in_paths: list of paths that include trajectories to use
            for sampling. This option can be used in case ``calc_ids``
            are not available. If used, workflow does not need to be
            provided. |default| ``None``
        :type in_paths: list of str
        :returns: a list of ASE Atoms objects
        :rtype: list
        """
        atoms_list = []

        if calc_ids is None:
            for path in in_paths:
                traj = join(path, traj_name)
                atoms = ase.io.read(traj, index=f'{beg}:{end}:{step}')
                atoms_list.extend(atoms)
        else:
            for calc_id in calc_ids:
                path = workflow.get_job_path(calc_id)
                traj = join(path, traj_name)
                atoms = ase.io.read(traj, index=f'{beg}:{end}:{step}')
                atoms_list.extend(atoms)
        for config in atoms_list:
            atom_ids = config.get_atomic_numbers()
            atom_labels = [elements_list[i - 1] for i in atom_ids]
            config.set_chemical_symbols(atom_labels)

        return atoms_list
