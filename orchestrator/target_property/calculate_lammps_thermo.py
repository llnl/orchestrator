import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
import os
from glob import glob
from os.path import getmtime, split
from typing import Union, Optional, Any
from ..simulator import simulator_builder
from ..potential import Potential
from . import TargetProperty
from ..workflow import Workflow
from ..storage import Storage
from orchestrator.target_property.analysis import AnalyzeLammpsLog
from ..utils.restart import restarter
from ..utils.isinstance import isinstance_no_import


class CalculateLammpsThermo(TargetProperty):
    """
    Target property class for computing various properties from LAMMPS
    simulations using the LAMMPS `thermo` output

    This module runs LAMMPS simulations (e.g. NPT, NVT ensembles), extracts
    user-selected thermo quantitites from the LAMMPS log file, performs
    time and ensemble averaging, and generates diagnostic plots.

    Simulation parameters, paths for force-field files and property selections
    are read from json file and provided explicitly via `calculate_property`.

    Supports two modes of operation:
    1. Forcefield mode: Uses LAMMPS forcefield files
    2. Potential mode: Uses Orchestrator Potential object or name of an
        existing potential file (e.g. cuMishinAlloy.eam)


    Typical use cases include density, temperature, pressure and stress
    calculations for single-component or mixture systems. Ensemble-specific
    input templates are selected via `input_template` (e.g. NPT, NVT).

    :param simulator_type: name of the simulator to perform simulations
    :type simulator_type: str
    :param simulator_path: path to the simulator executable
    :type simulator_path: str
    :param elements: list of elements which are present in the simulation
    :type elements: list
    :param input_template: LAMMPS template input file. If None, uses default
        template that supports both NPT and NVT ensembles
    :type input_template: str
    :param job_details: job submission parameters for running the job
    :type job_details: dict
    :param ensemble: ensemble type ('npt' or 'nvt'). Only used with default
        template. Default is 'npt'.
    :type ensemble: str
    """

    # Default LAMMPS simulation parameters
    default_units = 'real'
    default_atom_style = 'full'
    default_bond_style = 'harmonic'
    default_angle_style = 'harmonic'
    default_dihedral_style = 'hybrid harmonic opls'
    default_improper_style = 'harmonic'
    default_pair_style = 'hybrid/overlay lj/cut 11.0 buck 11.0 \
    coul/wolf 0.2 11.0'

    default_pair_coeff = 'coul/wolf'
    default_special_bonds = 'lj/coul 0.0 0.0 1.0'
    default_timestep = 1.0
    default_run_steps = 1000
    default_temperature = 298.15
    default_pressure = 1.0
    default_random_seed = 999
    default_ensemble = 'npt'

    # Required files for forcefield mode
    default_data_file = None
    default_ff_param = None
    default_ff_files = None

    def __init__(
        self,
        simulator_type: str,
        simulator_path: str,
        job_details: dict,
        input_template: str = None,
        elements: list = [],
        ensemble: str = 'npt',
        **kwargs: Any,
    ):
        """
        Initialization of the CalculateLammpsThermo class

        :param simulator_type: name of the simulator to perform simulations
        :type simulator_type: str
        :param simulator_path: path to the simulator executable
        :type simulator_path: str
        :param elements: list of elements which are present in the simulation
        :type elements: list
        :param input_template: LAMMPS template input file. If None, uses
            a default template that supports both NPT and NVT ensembles.
        :type input_template: str, optional
        :param job_details: job submission parameters for running the job
        :type job_details: dict
        :param ensemble: ensemble type ('npt' or 'nvt'). Only used with default
            template. Default is 'npt'.
        :type ensemble: str
        """

        self.job_details = job_details

        # Validate and store ensemble type
        ensemble = ensemble.lower()
        if ensemble not in ['npt', 'nvt']:
            raise ValueError(
                f"ensemble must be 'npt' or 'nvt', got '{ensemble}'")
        self.ensemble = ensemble

        if input_template:
            if os.path.exists(input_template):
                self.input_template = os.path.abspath(input_template)
            else:
                raise ValueError(
                    'An input template file is supplied but it does not exist!'
                )
        else:
            # Use default template if none provided
            source_file_location = os.path.dirname(os.path.abspath(__file__))
            self.input_template = (f'{source_file_location}/'
                                   'templates/lammps_main_default.lmp')

            # Check if default template exists
            if not os.path.exists(self.input_template):
                raise FileNotFoundError(
                    f'Default template not found at {self.input_template}. '
                    'Please provide an input_template path.')

        ensemble_simulator_args = {
            'code_path': simulator_path,
            'elements': elements,
            'input_template': self.input_template,
        }

        self.built_simulator_ensemble = simulator_builder.build(
            simulator_type,
            ensemble_simulator_args,
        )

        self.progress_flag = 'init'
        self.current_state = {}
        self.outstanding_ensemble = []

        self.ensemble_calcs = []

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
                'ensemble_calcs': self.ensemble_calcs,
                'ensemble': self.ensemble,
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

        self.ensemble_calcs = restart_dict.get('ensemble_calcs',
                                               self.ensemble_calcs)

        # Restore ensemble type if available
        self.ensemble = restart_dict.get('ensemble', self.default_ensemble)

        if len(self.ensemble_calcs) > 0:
            self.outstanding_ensemble = self.ensemble_calcs[-1]

        if self.progress_flag != 'init':
            if self.progress_flag == 'done':
                # restart information exists but the last calculation ended
                self.restart = False
                self.ensemble_calcs = []
            else:
                # restart information exists and we actually want to restart
                self.restart = True
        else:
            # no restart information exists
            self.restart = False

    def calculate_property(
        self,
        path_type: str,
        sim_params: dict,
        property_selections: list,
        lammps_resources: str = None,
        model_path: str = None,
        property_units: list = None,
        random_seed_use: bool = False,
        num_sims: int = 1,
        potential: Optional[Union[str, Potential]] = None,
        workflow: Optional[Workflow] = None,
        storage: Optional[Storage] = None,
        **kwargs,
    ) -> dict[str, Union[float, tuple[list[int], list[int]]]]:
        """
        Calculate properties using LAMMPS thermo function and plot them

        This method supports two modes:
        1. Forcefield mode: requires lammps_resources, data_file, ff_param
            and ff_files
        2. Potential mode: requires a Potential object or name of an existing
            potential file name


        :param path_type: path to perform thermo calculations
        :type path_type: str
        :param sim_params: simulation specific parameters
        :type sim_params: dict
        :param property_selections: list of thermo properties to calculate
            through LAMMPS
        :type property_selections: list
        :param lammps_resources: path of the LAMMPS forcefield and data files
            (required for forcefield mode)
        :type lammps_resources: str
        :param model_path: path to store the potential file
        :type model_path: str
        :param property_units: list of units corresponding to thermo properties
            defined in the selections to use in plots
        :type property_units: list
        :param random_seed_use: option to use random seed in the simulation
        :type random_seed_use: bool
        :param num_sims: number of simulations to perform for averaging and
            calculating standard deviation
        :type num_sims: int
        :param potential: interatomic potential to be used in LAMMPS. Can be
            either the string of a KIM potential available via the KIM API or
            a Potential object created by the Orchestrator.
        :type potential: str or Potential
        :param workflow: the workflow for managing job submission
        :type workflow: Workflow
        :param storage: the storage module
        :type storage: Storage
        :returns: dictionary with the final property values, std based
            on time-averaging for the property_std, and a tuple of the
            simulation calculation ID list
        :rtype: dict
        """

        if not property_units:
            property_units = [""] * len(property_selections)
        else:
            if len(property_units) < len(property_selections):
                property_units = property_units + [""] * (
                    len(property_selections) - len(property_units))
            else:
                property_units = property_units[:len(property_selections)]

        if workflow is None:
            workflow = self.default_wf

        # Determine if using Potential object or forcefield mode
        use_potential_mode = False

        if isinstance_no_import(potential, 'Potential'):
            use_potential_mode = True
            if hasattr(potential, 'install_potential_in_kim_api') and callable(
                    potential.install_potential_in_kim_api):
                module_name = self.__class__.__name__
                save_root = workflow.make_path(
                    module_name,
                    'potential_for_lammps_thermo',
                )
                potential_name = potential.kim_id
                potential.install_potential_in_kim_api(
                    potential_name=potential_name, save_path=save_root)
                model_path = f'{save_root}/{potential_name}'

                sim_params['potential'] = potential_name
            else:
                raise NotImplementedError('LAMMPS Thermo calculations only '
                                          'supports Potentials with working '
                                          'install_potential_in_kim_api '
                                          'methods at this time')
        else:
            if potential is not None:
                sim_params['potential'] = potential
                use_potential_mode = True

        # Validate required parameters based on mode
        if not use_potential_mode:
            # Forcefield mode requires these parameters
            if lammps_resources is None:
                raise ValueError("lammps_resources must be provided "
                                 "for forcefield mode")

            data_file = sim_params.get('data_file', self.default_data_file)
            if data_file is None:
                raise ValueError("data_file must be provided in sim_params "
                                 "for forcefield mode")

            ff_param = sim_params.get('ff_param', self.default_ff_param)
            if ff_param is None:
                raise ValueError("ff_param must be provided in sim_params "
                                 "for forcefield mode")

            ff_files = sim_params.get('ff_files', self.default_ff_files)
            if ff_files is None:
                raise ValueError("ff_files must be provided in sim_params "
                                 "for forcefield mode")

        timestep = sim_params.get('timestep', self.default_timestep)

        calc_id_error = []

        if not self.restart or self.progress_flag == 'running':
            if len(self.outstanding_ensemble) == 0:
                for i in range(num_sims):
                    calc_id = self._conduct_sim(
                        sim_params,
                        workflow,
                        f"{path_type}/run_{i}",
                        model_path,
                        lammps_resources,
                        random_seed_use=random_seed_use,
                        use_potential_mode=use_potential_mode,
                    )
                    self.outstanding_ensemble.append(calc_id)

                # checkpoint after the batch of calcs are submitted
                self.ensemble_calcs.append(self.outstanding_ensemble)
                self.progress_flag = 'running'
                self.checkpoint_property()

            workflow.block_until_completed(self.outstanding_ensemble)

            for c_id in self.outstanding_ensemble:
                run_path_ensemble = workflow.get_job_path(c_id)
                log_ensemble = run_path_ensemble + '/' + 'lammps.out'

                if 'ERROR: Lost atoms:' in open(log_ensemble).read():
                    self.logger.info(f'Lost atoms error for: {c_id}')
                    calc_id_error.append(c_id)

            if len(calc_id_error) > 0:
                traj_files = glob(f'{run_path_ensemble}/*.lammpstrj')
                file_times = [getmtime(f) for f in traj_files]
                newest_file = split(traj_files[np.argmax(file_times)])[1]
                results_dict = {
                    'property_value': newest_file,
                    'property_std': None,
                    'calc_ids': calc_id_error,
                    'success': False,
                }
                return results_dict

            property_time_series = {sel: [] for sel in property_selections}

            property_avgs = {sel: [] for sel in property_selections}
            property_stds = {sel: [] for sel in property_selections}

            for c_id in self.outstanding_ensemble:
                run_path_ensemble = workflow.get_job_path(c_id)
                log_lammps = run_path_ensemble + '/' + 'lammps.out'

                for sel, unit in zip(property_selections, property_units):
                    time, prop, prop_avg, prop_std = \
                        AnalyzeLammpsLog.extract_property([log_lammps, sel])

                    property_time_series[sel].append((time, prop))
                    property_avgs[sel].append(prop_avg)
                    property_stds[sel].append(prop_std)

            for sel, unit in zip(property_selections, property_units):
                self._plot_property(
                    property_name=sel,
                    property_time_series=property_time_series,
                    timestep=timestep,
                    unit=unit,
                    output_ensemble_plots=True,
                )

            self.outstanding_ensemble = []

        self.progress_flag = 'done'
        return_ensemble = self.ensemble_calcs
        self.ensemble_calcs = []
        self.checkpoint_property()

        final_property_avg = {}
        final_property_std = {}

        for sel in property_selections:
            vals = np.array(property_avgs[sel], dtype=float)

            final_property_avg[sel] = float(np.mean(vals))
            final_property_std[sel] = float(np.std(vals, ddof=1)) \
                if len(vals) > 1 else 0.0

        # return results
        results_dict = {
            'property_value': final_property_avg,
            'property_std': final_property_std,
            'calc_ids': (return_ensemble),
            'success': True,
        }
        return results_dict

    def _conduct_sim(
        self,
        sim_params: dict[str, Any],
        workflow: Workflow,
        sim_path: str,
        model_path: str,
        lammps_resources: str,
        random_seed_use: bool = False,
        use_potential_mode: bool = False,
    ) -> Union[int, str]:
        """
        Launch a single LAMMPS simulation for thermo property calculations

        This method prepares and submit a simulation using the LAMMPS simulator
        configured at class initialization. When multiple simulations are
        requested, this method is called repeatedly by ``calculate_property``.
        The method fills the LAMMPS input template, submits the job through the
        provided workflow, and returns a calculation ID that can be used to
        track job completion and retrieve simulation outputs.

        The simulation parameters are taken from ``sim_params`` and include
        thermodynamic conditions (temperature, pressure), force-field and
        topology information, timestep, and total simulation run length.
        A random seed is generated if ``random_seed_use`` is enabled.

        Two modes are supported:
        1. Forcefield mode: Uses LAMMPS forcefield files
        2. Potential mode: Uses Orchestrator Potential object or name of an
            existing potential file

        :param sim_params: Dictionary of simulation parameters used to
            populate the LAMMPS input template. Expected keys are: ``temp``,
            ``press``, ``timestep``, ``run_steps``, ``ensemble``

            For **forcefield mode**: ``units``, ``atom_style``, ``bond_style``,
            ``angle_style``, ``dihedral_style``, ``improper_style``,
            ``pair_style``, ``pair_coeff``, ``special_bonds``, ``data_file``,
            ``ff_param``, ``ff_files``

            For **potential mode**: ``units``, ``atom_style``, ``pair_style``,
            ``potential``, ``element``, ``mass``, ``lattice``,
            ``lattice_param``, ``l_x``, ``l_y``, ``l_z``
        :type sim_params: dict
        :param workflow: the workflow for managing job submission
        :type workflow: Workflow
        :param sim_path: path to perform simulations for
            simple property calculations
        :type sim_path: str
        :param model_path: path to store the potential file
        :type model_path: str
        :param lammps_resources: path of the LAMMPS forcefield and data files
            (only required in forcefield mode)
        :type lammps_resources: str
        :param random_seed_use: option to use random seed in the simulation
        :type random_seed_use: bool
        :param use_potential_mode: if True, uses Potential mode; if False,
            uses forcefield mode
        :type use_potential_mode: bool
        :returns: calculation ID corresponding to a spawned simulation
        :rtype: int
        """

        # Common parameters
        press = sim_params.get('press', self.default_pressure)
        temp = sim_params.get('temp', self.default_temperature)
        timestep = sim_params.get('timestep', self.default_timestep)
        run_steps = sim_params.get('run_steps', self.default_run_steps)

        # Get ensemble from sim_params or use class default
        ensemble = sim_params.get('ensemble', self.ensemble)
        ensemble = ensemble.lower()
        if ensemble not in ['npt', 'nvt']:
            raise ValueError(
                f"ensemble must be 'npt' or 'nvt', got '{ensemble}'")

        # Generate or use fixed random seed
        if random_seed_use:
            random_seed = random.randint(1, 10000)
        else:
            random_seed = self.default_random_seed

        # Common parameters to use in the template file
        template_fill = {
            'temperature': temp,
            'pressure': press,
            'seed': random_seed,
            'timestep': timestep,
            'run_steps': run_steps,
            'ensemble': ensemble,
        }

        if use_potential_mode:
            # Potential mode
            units = sim_params.get('units', 'metal')
            atom_style = sim_params.get('atom_style', 'atomic')
            pair_style = sim_params.get('pair_style', 'eam/alloy')
            potential = sim_params.get('potential', 'cuMishinAlloy.eam')
            element = sim_params.get('element', 'Cu')

            # Check if data_file is provided
            data_file = sim_params.get('data_file')

            template_fill.update({
                'ensemble_mode': 'potential',
                'units': units,
                'atom_style': atom_style,
                'pair_style': pair_style,
                'potential': potential,
                'element': element,
            })

            if data_file:
                # Using data file
                if lammps_resources:
                    lammps_resources_abs = os.path.abspath(lammps_resources)
                    data_file_full = os.path.join(lammps_resources_abs,
                                                  data_file)
                else:
                    data_file_full = os.path.abspath(data_file)

                template_fill['data_file'] = data_file_full

            else:
                # Creating box from scratch
                mass = sim_params.get('mass')
                lattice = sim_params.get('lattice')
                lattice_param = sim_params.get('lattice_param')
                l_x = sim_params.get('l_x', 10)
                l_y = sim_params.get('l_y', 10)
                l_z = sim_params.get('l_z', 10)

                if not mass:
                    raise ValueError(
                        "'mass' must be provided in sim_params "
                        "when creating box from scratch in potential mode")
                if not lattice:
                    raise ValueError(
                        "'lattice' must be provided in sim_params "
                        "when creating box from scratch in potential mode")
                if not lattice_param:
                    raise ValueError(
                        "'lattice_param' must be provided in sim_params "
                        "when creating box from scratch in potential mode")

                template_fill.update({
                    'data_file': None,
                    'mass': mass,
                    'lattice': lattice,
                    'lattice_param': lattice_param,
                    'l_x': l_x,
                    'l_y': l_y,
                    'l_z': l_z,
                })

        else:
            # Forcefield mode
            lammps_resources_abs = os.path.abspath(lammps_resources)

            units = sim_params.get('units', self.default_units)
            atom_style = sim_params.get('atom_style', self.default_atom_style)
            pair_style = sim_params.get('pair_style', self.default_pair_style)
            bond_style = sim_params.get('bond_style', self.default_bond_style)
            angle_style = sim_params.get('angle_style',
                                         self.default_angle_style)
            dihedral_style = sim_params.get('dihedral_style',
                                            self.default_dihedral_style)
            improper_style = sim_params.get('improper_style',
                                            self.default_improper_style)
            pair_coeff = sim_params.get('pair_coeff', self.default_pair_coeff)
            special_bonds = sim_params.get('special_bonds',
                                           self.default_special_bonds)

            # Get required parameters
            data_file = sim_params.get('data_file')
            ff_param = sim_params.get('ff_param')
            ff_files = sim_params.get('ff_files')

            ff_param_full = os.path.join(lammps_resources_abs, ff_param)
            data_file_full = os.path.join(lammps_resources_abs, data_file)

            # Create forcefield parameter file
            try:
                with open(ff_param_full, "x") as file:
                    for f in ff_files:
                        file.write("include " + lammps_resources_abs + "/" + f
                                   + "\n")
            except FileExistsError:
                pass

            template_fill.update({
                'ensemble_mode': 'forcefield',
                'units': units,
                'atom_style': atom_style,
                'bond_style': bond_style,
                'angle_style': angle_style,
                'dihedral_style': dihedral_style,
                'improper_style': improper_style,
                'pair_style': pair_style,
                'pair_coeff': pair_coeff,
                'special_bonds': special_bonds,
                'data_file': data_file_full,
                'ff_param': ff_param_full,
            })

        calc_id = self.built_simulator_ensemble.run(
            sim_path,
            model_path,
            template_fill,
            workflow=workflow,
            job_details=self.job_details,
        )

        return calc_id

    def calculate_with_error(
        self,
        n_calc,
        modified_params=None,
        potential=None,
        workflow=None,
    ):
        raise NotImplementedError("Use calculate_property() with num_sims > 1 "
                                  "instead to get property statistics.")

    def _plot_property(
        self,
        property_name: str,
        property_time_series: list,
        timestep: float,
        unit: str = "",
        output_ensemble_plots: bool = True,
    ):

        plt.figure()

        x_vals = [t for t, _ in property_time_series[property_name]]
        y_vals = [p for _, p in property_time_series[property_name]]

        # convert time to ps
        time_ps = [x * (timestep * 0.001) for x in x_vals]

        for i, (t, y) in enumerate(zip(time_ps, y_vals)):
            plt.plot(t, y, alpha=0.6, linewidth=1.2, label=f"run {i}")

        # ensemble plots
        if output_ensemble_plots and len(y_vals) > 1:
            min_len = min(len(y) for y in y_vals)
            y_stack = np.vstack([y[:min_len] for y in y_vals])
            t_ref = time_ps[0][:min_len]

            mean = np.mean(y_stack, axis=0)
            std = np.std(y_stack, axis=0, ddof=1)

            plt.plot(
                t_ref,
                mean,
                color="black",
                linewidth=2.5,
                label="ensemble mean",
            )

            plt.fill_between(
                t_ref,
                mean - std,
                mean + std,
                color="black",
                alpha=0.25,
                label="ensemble std",
            )

        plt.xlabel("time (ps)")
        ylabel = f"{property_name} (${unit}$)" if unit else property_name
        plt.ylabel(ylabel)

        plt.legend(fontsize=8, frameon=False)
        plt.tight_layout()

        plots_dir = "plots"

        os.makedirs(plots_dir, exist_ok=True)

        date_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = os.path.join(plots_dir,
                                   f"{property_name}_{date_stamp}.pdf")

        plt.savefig(output_file)
        plt.close()
