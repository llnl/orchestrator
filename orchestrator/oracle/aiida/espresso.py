from .oracle_base import AiidaOracle
from ase import Atoms, units
import subprocess as sp
from aiida import load_profile
from aiida.orm import load_node, StructureData
from aiida.engine.processes.builder import ProcessBuilder
from aiida.engine.processes.workchains.workchain import WorkChain
from aiida_quantumespresso.common.types import RelaxType
from ...scheduler.scheduler_base import Scheduler
from ...utils.data_standard import (
    ENERGY_KEY,
    FORCES_KEY,
    STRESS_KEY,
    METADATA_KEY,
)


class AiidaEspressoOracle(AiidaOracle):
    """
    Class for submitting quantum espresso calculations through AiiDA.

    Resposible for formatting the input for a workchain.

    :param oracle_args: arguments for instantiating a EspressoOracle. This
        includes any input values needed for the specified workchain.
    """

    def __init__(self,
                 code_str: str = None,
                 workchain: str = None,
                 clean_workdir: bool = True,
                 group: str = None,
                 **kwargs: dict):
        """
        Class for creating, running, and parsing Quantum Espresso calculations.

        :param code_str: Name of the code in the AiiDA database.
            e.g. pw@server
        :param workchain: Name of the workchain in AiiDA for Quantum Espresso.
            e.g. quantumespresso.relax
        :param clean_workdir: Will clean the working directory on the remote
            machine if True.
        :param group: Creates a group node in AiiDA to store all of the
            calculations for easy parsing afterwards based on the string name.
        """
        super().__init__(code_str=code_str,
                         workchain=workchain,
                         clean_workdir=clean_workdir,
                         group=group,
                         **kwargs)

    def parse_for_storage(
        self,
        run_path: str = '',
        calc_id: int = None,
        scheduler: Scheduler = None,
    ) -> Atoms:
        """
        Process calculation output to extract data in a consistent format.

        Parse the output from the Espresso calculation into ASE Atoms objects.
        The resulting Atoms will include the atomic configurations, cell info,
        and possibly energies, forces, and stresses. Units are: total system
        energy in eV, forces on each atom in eV/A, and stress on the system in
        eV/A^3

        :param run_path: Unique AiiDA identifier to load a node from the
            database
        :param calc_id: Calculation ID returned from an Oracle.
        :param scheduler: Scheduler object from orchestrator that has attached
            metadata.
        :returns: tuple of Atoms of the configurations and attached properties
            as well as a dictionary of metadata that should be stored with the
            configuration.
        """

        load_profile()

        if not calc_id and not scheduler:
            raise RuntimeError(
                'AiiDA Oracles must have `calc_id` and `scheduler` provided.')

        pk = calc_id

        node = load_node(pk)

        inputs = node.inputs.base.pw.parameters.get_dict()
        calculation = inputs['CONTROL'].get('calculation')
        if calculation != 'scf':
            atoms = node.outputs.output_structure.get_ase()
        else:
            atoms = node.inputs.structure.get_ase()

        traj = node.outputs.output_trajectory
        atoms.info[ENERGY_KEY] = traj.get_array('energy')[-1]
        atoms.info[STRESS_KEY] = traj.get_array('stress')[-1]
        atoms.set_array(FORCES_KEY, traj.get_array('forces')[-1])

        user = sp.run(
            'whoami',
            capture_output=True,
            shell=True,
            encoding='UTF-8',
        ).stdout.strip()

        universal = self.translate_universal_parameters(inputs, atoms)
        universal['code'] = 'Quantum Espresso'
        universal['executable_path'] = \
            str(node.inputs.base.pw.code.get_executable())

        new_metadata = {
            'generated_by': user,
            'data_source': f'AiiDA pk<{pk}>',
            'parameters': {
                'code': inputs,
                'universal': universal
            }
        }
        old_metadata = scheduler.get_attached_metadata(pk)
        combined_metadata = old_metadata | new_metadata
        atoms.info[METADATA_KEY] = combined_metadata

        return atoms

    def _oracle_specific_inputs(
        self,
        workchain: WorkChain,
        config: Atoms,
        job_details: dict,
    ) -> ProcessBuilder:
        """
        Specify code specific input values.

        Each code in AiiDA might need additional input values to successfully
        submit a workchain.

        :param builder: AiiDA builder object used to attach input parameters
            for job submission.
        :param config: ASE Atoms object of the current structure.
        :param job_details: Specific job submission information for the oracle.
        :returns: Updated builder object for job submission.
        """

        code = self.get_code(self.code_str)
        parameters = self.get_parameters(self.parameters)
        parameters = parameters.get_dict()

        options = self.get_options(job_details, code.computer)

        # Convert ASE Atoms object to AiiDA
        self.structure = StructureData(ase=config)

        # Get the type of calculation for relaxation.
        relax_type = self.get_relax_type(parameters)

        overrides = {
            'base': {
                'kpoints_distance': self.kspacing,
                'pseudo_family': self.potential_family,
                'pw': {
                    'parameters': parameters
                }
            },
            'base_final_scf': {
                'kpoints_distance': self.kspacing,
                'pseudo_family': self.potential_family,
                'pw': {
                    'parameters': parameters
                }
            }
        }

        builder = workchain.get_builder_from_protocol(code=code,
                                                      structure=self.structure,
                                                      options=options,
                                                      overrides=overrides,
                                                      relax_type=relax_type)

        return builder

    def get_relax_type(self, parameters: dict) -> RelaxType:
        """
        Selects the correct RelaxType based on the input parameters for
        Quantum Espresso.

        :param parameters: Input parameters for a Quantum Espresso calculation.
        :returns: The specified relaxation type
        """

        relax_type = RelaxType.NONE

        # Get the calculation type and cell_dofree value if available.
        calculation = parameters['CONTROL'].get('calculation', 'scf')
        cell = parameters.get('CELL', None)
        if cell is not None:
            cell_dofree = cell.get('cell_dofree', None)

        if calculation == 'scf':
            relax_type = RelaxType.NONE
        elif calculation == 'relax':
            relax_type = RelaxType.POSITIONS
        elif calculation == 'vc-relax':
            if cell_dofree is None:
                relax_type = RelaxType.POSITIONS_CELL
            elif cell_dofree == 'shape':
                relax_type = RelaxType.POSITIONS_SHAPE
            elif cell_dofree == 'volume':
                relax_type = RelaxType.POSITIONS_VOLUME

        return relax_type

    @staticmethod
    def translate_universal_parameters(parameters: dict,
                                       structure: Atoms) -> dict:
        """
        Orchestrator has predefined universal input values for varying codes
        to allow some transferability. Each Oracle will need a function to
        translate those values from the specific code. This function will
        take the Quantum Espresso input values and convert them to the
        universal values to be stored with the dataset on the initial
        submission.

        :param parameters: Dictionary containing all the pw.in parameters to
            run the simulation.
        :param structure: ASE Atoms object that will be used to map the
            magnetic moments.
        :returns: Dictionary of universal input parameters for database
            storage.
        """

        translated = {}

        for key, value in parameters.items():
            for k, v in value.items():
                match k:
                    case 'ecutwfc':
                        translated['planewave_cutoff'] = v * units.Ry
                    case 'smearing':
                        translated['smearing_method'] = v
                    case 'degauss':
                        translated['smearing_value'] = v * units.Ry
                    case 'etot_conv_thr':
                        translated['energy_convergence'] = v
                    case 'forc_conv_thr':
                        translated['force_convergence'] = v
                    case 'nspin':
                        translated['spin_mode'] = v
                    case 'starting_magnetization':
                        translated['magnetic_moments'] = \
                            AiidaEspressoOracle._convert_to_magmom(
                                v, structure)
                    case 'calculation':
                        if v == 'relax':
                            translated['ion_relax'] = True
                        if v == 'vc-relax':
                            translated['ion_relax'] = True
                            translated['cell_relax'] = True
                    case 'mixing_mode':
                        translated['mixing_mode'] = v
                    case 'mixing_beta':
                        translated['mixing_value'] = v
                    case 'diagonalization':
                        translated['diagonalization_method'] = v
                    case 'ion_dynamics':
                        translated['ion_optimization_method'] = v
                    case _:
                        pass

        return translated
