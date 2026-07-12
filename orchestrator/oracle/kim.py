from os import PathLike
from os.path import abspath
from ase import Atoms
from ase.calculators.kim.kim import KIM
from typing import Union
from .oracle_base import Oracle
from ..utils.input_output import safe_read, safe_write, try_loading_ase_keys
from ..scheduler.scheduler_base import Scheduler
from ..utils.data_standard import METADATA_KEY


class KIMOracle(Oracle):
    """
    Class for creating, running, and parsing ASE calculations with KIM

    Resposible for making ASE "input file", running the calculator, and
    extracting energy and forces from the output. Calculating the stress
    tensor is not currently supported.
    """

    def __init__(self, potential: str, **kwargs):
        """
        set variables and initialize the recorder and default scheduler

        :param potential: string of the KIM potential installed in the KIM API
        :type potential: str
        """
        self.potential = potential
        if self.potential is None:
            raise KeyError(('A potenital string (potential) must be provided '
                            'in oracle_args to instantiate a KIMOracle'))

        self.output_filename = 'kim_calc_oracle.xyz'
        self.calculator = KIM(self.potential)
        super().__init__(**kwargs)

    def write_input(
        self,
        run_path: str,
        input_args: dict,
        config: Atoms,
    ):
        """
        Make ASE calculator (``Atoms``)

        When using ASE+KIM potential, calculations are not done externally,
        but instead directly in the python environment from the ``Atoms`` data
        structure. Thus no input file is actually written, but instead the
        internal atoms attribute is set.

        :param run_path: directory path where the file is written, not used
        :type run_path: str
        :param input_args: dictionary of input arguments for ``Atoms`` (not
            currently used, constructed internally)
        :type input_args: dict
        :param config: ASE Atoms object of the configuration, containing
            position, cell, and atom type information at minimum
        :type config: Atoms
        :returns: name of written input file
        :rtype: str
        """
        # we only want to keep the structural information from the object
        pos = config.get_positions()
        cell = config.get_cell()[:]
        atype = config.get_chemical_symbols()
        self.atoms = Atoms(atype, positions=pos, cell=cell, pbc=[1, 1, 1])
        self.logger.info(
            f'ASE Atoms constructed for calculation with {self.potential}')
        return None

    def get_run_command(
        self,
        run_path: Union[str, PathLike],
        **unused_job_details,
    ):
        """
        assign the KIM calculator to the ``Atoms`` attribute

        this method bypasses the actual submission of a job to any scheduler,
        instead using the internal python environment/KIM to calculate ground
        truth values. Calculation outputs are saved in the ``run_path``
        specified in the ``args`` dict.

        :param run_path: output location of the calculation
        :type run_path: str
        :returns: None
        :rtype: None
        """
        self.logger.info(('KIMOracle works internally, no need for execution'
                          ' of any command'))
        self.atoms.calc = self.calculator

        # keys need to be manually assigned; not auto-updated in get_*()
        try_loading_ase_keys(self.atoms)
        self.atoms.calc = None

        out_file = f'{run_path}/{self.output_filename}'
        safe_write(out_file, self.atoms, format='extxyz')
        self.logger.info(f'Wrote KIM/ASE calculation output to {out_file}')

        return None

    def parse_for_storage(
        self,
        run_path: str = '',
        calc_id: Union[int, str] = None,
        scheduler: Scheduler = None,
    ) -> Atoms:
        """
        process calculation output to extract data in the format for Storage

        Parse the output from the KIM calculation into ASE Atoms objects. The
        resulting Atoms will include the atomic configurations, cell info, and
        possibly energies, forces, and stresses. Units are: total system energy
        in eV, forces on each atom in eV/A, and stress on the system in eV/A^3

        :param run_path: directory where the oracle output file resides.
            If not provided, will be extracted from the scheduler using calc_id
        :param calc_id: Calculation ID to look up via scheduler.get_job_path().
            Can be int or str depending on scheduler implementation.
        :param scheduler: Scheduler object from orchestrator that has attached
            metadata.
        :returns: Atoms of the configuration and attached properties and a
            dictionary of metadata that should be stored with the
            configuration.
        """
        if not run_path:
            run_path = scheduler.get_job_path(calc_id)
        data_file = f'{run_path}/{self.output_filename}'
        # safe_read outputs list but we expect sinlge config
        atoms = safe_read(data_file, format='extxyz')[0]
        atoms.info[METADATA_KEY] = {'data_source': abspath(data_file)}
        return atoms
