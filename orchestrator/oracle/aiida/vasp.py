from .oracle_base import AiidaOracle
import subprocess as sp
from ase import Atoms
from aiida import load_profile
from aiida.orm import load_node, Dict, StructureData
from aiida.engine.processes.builder import ProcessBuilder
from aiida.common.extendeddicts import AttributeDict
from aiida.engine.processes.workchains.workchain import WorkChain
from ...scheduler.scheduler_base import Scheduler
from ...utils.data_standard import (
    ENERGY_KEY,
    FORCES_KEY,
    STRESS_KEY,
    METADATA_KEY,
)


class AiidaVaspOracle(AiidaOracle):
    """
    Class for creating, running, and parsing VASP calculations

    Responsible for making any needed input files, run the code, and extract
    energy, forces, and stress tensor from output.
    """

    def __init__(self,
                 code_str: str = None,
                 workchain: str = None,
                 settings: dict = None,
                 clean_workdir: bool = True,
                 group: str = None,
                 **kwargs: dict):
        """
        Class for creating, running, and parsing VASP calculations.

        :param code_str: Name of the code in the AiiDA database.
            e.g. vasp_std@server
        :param workchain: Name of the workchain in AiiDA for VASP.
            e.g. vasp.relax
        :param settings: Controls the parsing behavior and other attributes.
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

        self.settings = settings
        self.kwargs = kwargs

    def parse_for_storage(
        self,
        run_path: str = '',
        calc_id: int = None,
        scheduler: Scheduler = None,
    ) -> Atoms:
        """
        Process calculation output to extract data in a consistent format

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
        :returns: ASE Atoms object of the configuration and attached properties
            as well as a dictionary of metadata that should be stored with the
            configuration.
        """

        if not calc_id and not scheduler:
            raise RuntimeError(
                'AiiDA Oracles must have `calc_id` and `scheduler` provided.')

        pk = calc_id

        load_profile()

        node = load_node(pk)

        output = node.outputs

        if 'relax' in output:
            atoms = output.relax.structure.get_ase()
        else:
            atoms = output.structure.get_ase()

        traj = output.trajectory
        misc = output.misc.get_dict()

        stress = traj.get_array('stress')[-1]
        forces = traj.get_array('forces')[-1]
        energy = misc['total_energies']['energy_extrapolated_electronic']
        inputs = misc['parameters']

        atoms.info[ENERGY_KEY] = energy
        atoms.info[STRESS_KEY] = stress
        atoms.set_array(FORCES_KEY, forces)

        user = sp.run(
            'whoami',
            capture_output=True,
            shell=True,
            encoding='UTF-8',
        ).stdout.strip()

        universal = self.translate_universal_parameters(inputs)
        universal['code'] = 'VASP'
        universal['version'] = misc['version']
        universal['executable_path'] = \
            str(node.inputs.vasp.code.get_executable())

        new_metadata = {
            'generated_by': user,
            'data_source': f'AiiDA pk<{pk}>',
            'parameters': {
                'code': inputs,
                'universal': universal
            },
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

        :param workchain: AiiDA WorkChain object that will be used to create
            the builder for submitting the job.
        :param config: ASE Atoms object of the current structure.
        :param job_details: Specific job submission information for the oracle.
        :returns: Updated builder object for job submission.
        """

        builder = workchain.get_builder()

        builder.vasp.code = self.get_code(self.code_str)
        parameters = self.get_parameters(self.parameters)
        if 'incar' not in parameters.keys():
            parameters = {'incar': parameters}
        builder.vasp.parameters = parameters

        options = self.get_options(job_details, builder.vasp.code.computer)
        builder.vasp.options = Dict(options)

        # Structure
        # Convert ASE Atoms object to AiiDA
        self.structure = StructureData(ase=config)
        builder.structure = self.structure

        # POTCAR data
        builder.vasp.potential_family = self.potential_family
        potential_mapping = self.potential_mapping
        if builder.vasp.potential_family and not potential_mapping:
            atoms = self.structure.get_ase()
            potential_mapping = {}
            for atom in set(atoms.get_chemical_symbols()):
                potential_mapping[atom] = atom
            builder.vasp.potential_mapping = Dict(potential_mapping)

        # Kpoints
        kpoints = self.kpoints
        kspacing = self.kspacing
        builder.vasp.kpoints = self.get_kpoints(kpoints=kpoints,
                                                kspacing=kspacing)

        # Settings
        if not self.settings:
            self.settings = self.default_settings()
        builder.vasp.settings = Dict(self.settings)

        # Relax options
        if self.relax:
            self.relax = AttributeDict(dictionary=self.relax)
        else:
            self.relax = self.default_relax_options(self.relax)
        builder.relax_settings = self.relax

        # Set the rest of the vasp specific oracle args
        builder.vasp = self.set_other_oracle_args(builder.vasp, self.kwargs)

        return builder

    def default_settings(self) -> AttributeDict:
        """
        The settings object controls parsing of the VASP calculation.

        :returns: Attribute dictionary with parser settings.
        """
        # Set default values that always need to be parsed.
        settings = AttributeDict()
        settings.parser_settings = {
            'include_node': ['energies', 'trajectory'],
            'include_quantity': ['forces', 'stress', 'parameters'],
            'electronic_step_energies': True
        }

        return settings

    def default_relax_options(self, overrides: dict = None) -> AttributeDict:
        """
        To perform structure relaxations with AiiDA for VASP, a relax object
        is created to control the various options.

        :param overrides: Values that will change the default behavior. This
            can include changing `relax.perform` from False to True to
            perform a geometry optimization.

        :returns: Attribute dictionary with relaxation objects
        """

        relax = AttributeDict()

        # Relax options
        relax.perform = False
        # Select relaxation algorithm
        relax.algo = 'cg'
        # Set force cutoff limit (EDIFFG, but no sign needed)
        relax.force_cutoff = 0.01
        # Turn on relaxation of positions (strictly not needed as the default
        # is on). The three next parameters correspond to the well known
        # ISIF=3 setting
        relax.positions = True
        # Turn on relaxation of the cell shape (defaults to False)
        relax.shape = False
        # Turn on relaxation of the volume (defaults to False)
        relax.volume = False
        # Set maximum number of ionic steps
        relax.steps = 100

        if overrides:
            for k, v in overrides.items():
                relax[k] = v

        return relax

    @staticmethod
    def translate_universal_parameters(parameters: dict) -> dict:
        """
        Orchestrator has predefined universal input values for varying codes
        to allow some transferability. Each Oracle will need a function to
        translate those values from the specific code. This function will
        take the VASP INCAR values and convert them to the universal values
        to be stored with the dataset on the initial submission.

        :param parameters: Dictionary containing all the INCAR parameters to
            run the simulation.
        :returns: Dictionary of universal input parameters for database
            storage.
        """

        translated = {}

        for key, value in parameters.items():
            match key:
                case 'GGA':
                    translated['xc'] = value
                case 'ENCUT':
                    translated['planewave_cutoff'] = value
                case 'ISMEAR':
                    translated['smearing_method'] = value
                case 'SIGMA':
                    translated['smearing_value'] = value
                case 'EDIFF':
                    translated['energy_convergence'] = value
                case 'EDIFFG':
                    translated['force_convergence'] = value
                case 'ISPIN':
                    if parameters.get('LNONCOLLINEAR', False):
                        translated['spin_mode'] = 4
                    else:
                        translated['spin_mode'] = value
                case 'MAGMOM':
                    translated['magnetic_moments'] = value
                case 'ISIF':
                    if value == 2:
                        translated['ion_relax'] = True
                    if value == 3:
                        translated['ion_relax'] = True
                        translated['cell_relax'] = True
                case 'IMIX':
                    translated['mixing_mode'] = value
                case 'AMIX':
                    translated['mixing_value'] = value
                case 'IVDW':
                    translated['vdw_correction'] = value
                case 'LDAUTYPE':
                    translated['hubbard_method'] = value
                case 'LDAUL':
                    translated['hubbard_orbitals'] = value
                case 'LDAUU':
                    translated['hubbard_u'] = value
                case 'LDAUJ':
                    translated['hubbard_j'] = value
                case 'IALGO':
                    translated['diagonalization_method'] = value
                case 'IBRION':
                    translated['ion_optimization_method'] = value
                case _:
                    pass

        return translated
