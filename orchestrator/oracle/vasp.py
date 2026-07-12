import subprocess
from os import PathLike
from os.path import abspath, exists
from ase import Atoms
from ase.io import read
from ase.io.espresso import kspacing_to_grid
from ase.build import sort
import numpy as np
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.vasp.outputs import Vasprun
from typing import Union
from .oracle_base import Oracle
from ..scheduler.scheduler_base import Scheduler
from ..utils.data_standard import (
    ENERGY_KEY,
    FORCES_KEY,
    STRESS_KEY,
    METADATA_KEY,
)


class VaspOracle(Oracle):
    """
    Class for creating, running, and parsing quantum espresso calculations

    Resposible for making PWscf input file, run the code and extract energy,
    forces, and stress tensor from PWscf output
    """

    def __init__(
        self,
        code_path: Union[str, PathLike] = None,
        **kwargs,
    ):
        """
        set variables and initialize the recorder and default scheduler

        :param code_path: path of the VASP executable
        """

        self.code_path = code_path
        if self.code_path is None:
            raise KeyError('A path to the VASP executable (code_path) must be '
                           'provided in oracle_args to instantiate a '
                           'VaspOracle.')

        self.output_filename = 'OUTCAR'
        super().__init__(**kwargs)

    def write_input(
        self,
        run_path: str,
        input_args: dict,
        config: Atoms,
    ) -> str:
        """
        Write input files for VASP calculation.

        :param run_path: directory path where the file is written
        :param input_args: dictionary of input arguments for the INCAR file.
            This will contain multiple values with the following format::

                input_args = {
                    incar: {INCAR parameters},
                    kpoints: [5, 5, 5],
                    kspacing: 0.1,
                    pseudo_path: '/path/to/pseudos',
                    pseudo_mapping: {
                        'Fe': 'Fe_pv'
                    }
                }

            `pseudo_mapping` can be set to use a specific pseudo based on the
            folder name. Otherwise, the default is to use the basic pseudo.
            Kpoints or kspacing (no 2pi factor) may be defined. kspacing is
            used by default
        :param config: ASE Atoms object of the configuration, containing
            position, cell, and atom type information at minimum
        :returns: name of written input file
        """

        file_name = 'INCAR'

        if not isinstance(config, Atoms):
            raise ValueError(
                '`config` must be an ASE Atoms object with the structure '
                'information.')

        incar = input_args.get('incar', None)
        if not isinstance(incar, dict):
            raise ValueError('The `incar` key in the `input_args` must be a '
                             'dictionary with VASP input parameters.')
        incar = Incar.from_dict(incar)
        incar.write_file(f'{run_path}/{file_name}')

        kpoints = input_args.get('kpoints', None)
        kspacing = input_args.get('kspacing', None)
        kpts = None
        # use kspacing as the default
        if kspacing is not None:
            if not isinstance(kspacing, float):
                raise ValueError('Specify the kspacing as a float.')
            else:
                kpts = kspacing_to_grid(config, kspacing)
        elif kpoints is not None:
            if not isinstance(kpoints, list) and not all(
                    isinstance(x, int) for x in kpoints):
                raise ValueError(
                    'The `kpoints` key in `input_args` must be a list of ints.'
                )
            else:
                kpts = kpoints
        else:
            raise ValueError('kspacing or kpoints must be set in input_args')

        kpoints = Kpoints(kpts=kpts)
        kpoints.write_file(f'{run_path}/KPOINTS')

        config = sort(config)
        config.write(f'{run_path}/POSCAR', vasp6=True)

        potcar = "cat "
        pseudo_path = input_args.get('pseudo_path', None)
        if not exists(pseudo_path):
            raise ValueError(
                'The provided `pseudo_path` in `input_args` does not appear '
                'to be a working path.')
        pseudo_mapping = input_args.get('pseudo_mapping', {})
        elements = sorted(set(config.get_chemical_symbols()))
        for element in elements:
            pseudo = pseudo_mapping.get(element, None)
            if pseudo is None:
                pseudo = f'{pseudo_path}/{element}/POTCAR '
            else:
                pseudo = f'{pseudo_path}/{pseudo}/POTCAR '
            potcar += pseudo
        potcar += f"> {run_path}/POTCAR"

        subprocess.call(potcar, shell=True)

        return file_name

    def get_run_command(self, **kwargs) -> str:
        """
        return the command to run a VASP calculation

        this method formats the run command based on the ``code_path`` internal
        variable set at instantiation of the Oracle, which the
        :class:`~.scheduler_base.Scheduler` will execute in
        the proper ``run_path``.

        :returns: single line string with code execution statement
        """
        command = (f'{self.code_path}  > vasp.out')
        return command

    def parse_for_storage(
        self,
        run_path: str = '',
        calc_id: Union[int, str] = None,
        scheduler: Scheduler = None,
    ) -> Atoms:
        """
        process calculation output to extract data in a consistent format

        Parse the output from the VASP calculation into ASE Atoms objects.
        The resulting Atoms will include the atomic configurations, cell info,
        and possibly energies, forces, and stresses. Units are: total system
        energy in eV, forces on each atom in eV/A, and stress on the system in
        eV/A^3

        :param run_path: directory where the oracle output file resides.
            If not provided, will be extracted from the scheduler using calc_id
        :param calc_id: Calculation ID to look up via scheduler.get_job_path().
            Can be int or str depending on scheduler implementation.
        :param scheduler: Scheduler object of Orchestrator.
        :returns: Atoms of the configuration and attached properties and a
            dictionary of metadata that should be stored with the
            configuration.
        """
        if not run_path:
            run_path = scheduler.get_job_path(calc_id)
        data_file = f'{run_path}/{self.output_filename}'
        vasprun = Vasprun(f'{run_path}/vasprun.xml')
        if not vasprun.converged_electronic:
            raise RuntimeError(f'Calc at {run_path} is not converged')
        atoms = read(data_file, format='vasp-out')
        atoms.info[ENERGY_KEY] = float(vasprun.final_energy)
        atoms.info[STRESS_KEY] = vasprun.ionic_steps[-1]['stress']
        atoms.set_array(FORCES_KEY,
                        np.array(vasprun.ionic_steps[-1]['forces']))
        code = self.get_parameters(run_path)
        universal = self.translate_universal_parameters(code)
        universal['code'] = 'VASP'
        universal['executable_path'] = self.code_path
        code_parameters = {'code': code, 'universal': universal}

        user = subprocess.run(
            'whoami',
            capture_output=True,
            shell=True,
            encoding='UTF-8',
        ).stdout.strip()

        new_metadata = {
            'generated_by': user,
            'raw_path': abspath(data_file),
            'code_parameters': code_parameters
        }
        if calc_id is not None:
            new_metadata['data_source'] = f'scheduler calc_id<{calc_id}>'
            old_metadata = scheduler.get_attached_metadata(calc_id)
            combined_metadata = old_metadata | new_metadata
        else:
            combined_metadata = new_metadata
        atoms.info[METADATA_KEY] = combined_metadata

        return atoms

    def get_parameters(
        self,
        runpath: str = None,
    ) -> dict:
        """
        Read in the input parameters from a VASP calculation.

        :param runpath: directory where the oracle calculation files reside.
        :return: Dict of VASP input parameters.
        """

        vasprun = Vasprun(f'{runpath}/vasprun.xml')

        parameters = vasprun.parameters

        return parameters

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
