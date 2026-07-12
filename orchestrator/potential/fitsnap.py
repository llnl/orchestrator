import numpy as np
from fitsnap3lib.fitsnap import FitSnap
import os
from ase import Atoms
from typing import Optional, Union
from .potential_base import Potential, ModelHyperparameters
from .kim_mixins import KIMKitHandler
from ..storage import Storage
from ..scheduler import Scheduler
from dataclasses import dataclass, asdict


@dataclass
class SNAPBispectrumModelHyperparameters(ModelHyperparameters):
    """
    Hyperparameters for the SNAP Bispectrum Potential model.

    :param two_j_max: Maximum band for bispectrum components (2*j_max)
    :param rfac0: Parameter controlling cutoff radius scaling
    :param quadratic: Whether to use quadratic SNAP features
    :param rmin0: Minimum cutoff radius
    :param wj: Weight parameter for atoms or species
    :param radelem: Atomic radius parameter for species
    :param use_zbl: Whether to use ZBL potential for short-range interactions
    :param wselfallflag: Self-weight flag (0 or 1)
    :param chemflag: Chemical environment flag (0 or 1)
    :param bzeroflag: B-zero flag (0 or 1)
    """
    two_j_max: int
    rfac0: float
    quadratic: bool
    rmin0: float
    wj: Union[float, list[float]]
    radelem: Union[float, list[float]]
    use_zbl: bool
    wselfallflag: int
    chemflag: int
    bzeroflag: int


class SNAPPotential(Potential, KIMKitHandler):
    """
    Spectral Neighbor Analysis Potential (SNAP) implementation

    The SNAPPotential class implements the SNAP potential, which uses
    bispectrum components to represent the local atomic environment. This
    implementation is designed to work with FitSNAP and LAMMPS.

    The potential can be:
    1. Initialized with explicit hyperparameters for training
    2. Loaded from a template file
    3. Loaded from existing parameter files (snapcoeff, snapparam, etc.)

    SNAP potentials use a spectral decomposition of the neighbor density
    function into 4D hyperspherical harmonics.
    """

    default_cutoff_radius = 4.67
    default_rfac0 = 0.99363
    default_two_j_max = 6
    default_quadratic = False
    default_rmin0 = 0.0
    default_wj = 1.0
    default_radelem = 0.5
    default_use_zbl = False
    default_wselfallflag = 0
    default_chemflag = 0
    default_bzeroflag = 0
    training_script_name = "training_script.py"

    # Define the required and optional files for SNAP potential
    _required_files = [
        "snap_potential.snapcoeff",  # coeffs
        "snap_potential.snapparam",  # params
        "snap_potential.mod",  # import lines in lammps
    ]
    _optional_files = [
        "snap_potential.md",  # training metrics
        "snap_potential.in",  # training input
        training_script_name,  # training script
    ]

    # KIMkit settings - static for SNAP
    # only support as simulator model for now
    # important for KIM API integration but set as metadata in kimkit
    _potential_type: str = 'SNAP'  # internal metadata classification tag
    _kim_item_type: str = "simulator-model"
    _kim_simulator_name: str = 'lammps'  # simulation code support
    _model_type: str = 'snap'  # pair style string for use in simulation code
    _model_driver: str = None  # only support as simulator model for now, model
    # driver remains unset
    # static definition, zbl not supported at this time
    _model_definition: list = [
        "pair_style snap",
        # snapcoeff, snapparam, species list
        ("pair_coeff * * @<parameter-file-1>@ @<parameter-file-2>@ "
         "@<atom-type-sym-list>@"),
    ]

    # Set some dummy fields just to pacify kimkit - will be overwritten for
    # KIM API handler
    _kim_api_version: str = 'kim-api-not-supported-for-this-model'

    # TODO: will need CMake file for KIM API

    def __init__(
        self,
        species: list[str],
        # hyperparameters explicit
        cutoff_radius: float = default_cutoff_radius,
        rfac0: float = default_rfac0,
        two_j_max: int = default_two_j_max,
        quadratic: bool = default_quadratic,
        rmin0: float = default_rmin0,
        wj: Union[float, list[float]] = default_wj,
        radelem: Union[float, list[float]] = default_radelem,
        use_zbl: bool = default_use_zbl,
        wselfallflag: int = default_wselfallflag,
        chemflag: int = default_chemflag,
        bzeroflag: int = default_bzeroflag,
        # template that will be trained
        template: Optional[str] = None,
        # potential name for KIM ID generation and saving for external usage
        potential_name: str = 'snap_potential',
        **kwargs,
    ):
        """
        Initialize the SNAP potential

        This constructor provides several ways to initialize a SNAP potential:
        1. With explicit hyperparameters (cutoff_radius, rfac0, etc.)
        2. From a template file (template)
        To import from existing files, use the :meth:`initialize_from_files`
        class method.

        :param species: List of element symbols used in the potential
        :param cutoff_radius: Cutoff radius for atomic interactions
        :param rfac0: Scaling factor for distance (0-1, controls smoothing)
        :param two_j_max: Maximum 2j value for bispectrum components (even
            integer)
        :param quadratic: Whether to use quadratic terms in the potential
        :param rmin0: Minimum cutoff radius
        :param wj: Weight parameter for atoms/species (scalar or list)
        :param radelem: Atomic radius parameter (scalar or list per species)
        :param use_zbl: Whether to use ZBL potential for short-range
            interactions
        :param wselfallflag: Self-weight flag (0 or 1)
        :param chemflag: Chemical environment flag (0 or 1)
        :param bzeroflag: B-zero flag (0 or 1)
        :param template: Path to FitSNAP template input file
        :param potential_name: Name of the potential, used as filename prefix
        :param kwargs: Additional keyword arguments
        """
        # Initialize with base class
        super().__init__(
            species=species,
            potential_name=potential_name,
            **kwargs,
        )

        # Track input files as a dictionary mapping file types to file paths
        self.potential_files = {
            k: None
            for k in self._required_files + self._optional_files
        }
        if template:
            if os.path.exists(template):
                self.template = os.path.abspath(template)
            else:
                raise ValueError(
                    'A template file is supplied but it does not exist!')
        else:
            self.template = None

        # Initialize based on the available inputs
        # Initialization can come from two different sources:
        # 1. Template input file
        # 2. Explicit hyperparameters
        self._initialize_from_inputs(
            cutoff_radius,
            rfac0,
            two_j_max,
            quadratic,
            rmin0,
            wj,
            radelem,
            use_zbl,
            wselfallflag,
            chemflag,
            bzeroflag,
        )

    @classmethod
    def initialize_from_files(
        cls,
        param_file: str,  # snapparam file
        coeff_file: str,  # snapcoeff file
        mod_file: str,  # mod file for lammps file
        input_file: Optional[str] = None,  # training input
        training_file: Optional[str] = None,  # training script
        md_file: Optional[str] = None,  # metrics file
        potential_name: str = 'snap_potential',
    ) -> Potential:
        """
        :param param_file: Path to existing snapparam file
        :param coeff_file: Path to existing snapcoeff file
        :param mod_file: Path to LAMMPS mod file with potential commands
        :param md_file: Path to markdown metrics file
        """
        species, hyperparameters = cls._load_hyperparameters_from_files(
            param_file, coeff_file)
        instance = cls(
            species=species,
            **asdict(hyperparameters),
            potential_name=potential_name,
        )

        instance.potential_files = {
            "snap_potential.snapcoeff": coeff_file,
            "snap_potential.snapparam": param_file,
            "snap_potential.mod": mod_file,
            "snap_potential.md": md_file,
            "snap_potential.in": input_file,
            "training_script.py": training_file,
        }
        if input_file:
            instance.template = input_file

        # files are provided, but make sure they exist. Exception raised
        # if any missing
        instance._has_required_files = instance._check_files_set_and_exist(
            check_exist=True, )

        return instance

    @classmethod
    def initialize_from_kim(cls, kim_id: str) -> Potential:
        """
        Initialize the model from an existing potential in KIMkit

        :param kim_id: The KIM ID of the saved potential (e.g.,
            snap_potential__SM_...)
        """
        instance = super().initialize_from_kim(kim_id)

        for file_name in instance.potential_files.keys():
            full_path = os.path.join(kim_id, file_name)
            if os.path.exists(full_path):
                instance.potential_files[file_name] = full_path

        species, hyperparameters = instance._load_hyperparameters_from_files(
            instance.potential_files["snap_potential.snapparam"],
            instance.potential_files["snap_potential.snapcoeff"],
        )
        if instance.potential_files["snap_potential.in"]:
            instance.template = instance.potential_files["snap_potential.in"]
        instance.species = species
        instance.hyperparameters = hyperparameters

        # files are provided, but make sure they exist. Exception raised
        # if any missing
        instance._has_required_files = instance._check_files_set_and_exist(
            check_exist=True)

        return instance

    def _initialize_from_inputs(
        self,
        cutoff_radius,
        rfac0,
        two_j_max,
        quadratic,
        rmin0,
        wj,
        radelem,
        use_zbl,
        wselfallflag,
        chemflag,
        bzeroflag,
    ):
        """Initialize the potential based on available inputs"""
        self._has_required_files = False
        # Priority 1: Check if template is provided
        if self.template:
            # Parse template for hyperparameters
            set_hyperparams = self._parse_template_file(self.template)
            self.hyperparameters = SNAPBispectrumModelHyperparameters(
                cutoff_radius=set_hyperparams.get('cutoff_radius',
                                                  cutoff_radius),
                rfac0=set_hyperparams.get('rfac0', rfac0),
                two_j_max=set_hyperparams.get('two_j_max', two_j_max),
                quadratic=set_hyperparams.get('quadratic', quadratic),
                rmin0=set_hyperparams.get('rmin0', rmin0),
                wj=set_hyperparams.get('wj', wj),
                radelem=set_hyperparams.get('radelem', radelem),
                use_zbl=set_hyperparams.get('use_zbl', use_zbl),
                wselfallflag=set_hyperparams.get('wselfallflag', wselfallflag),
                chemflag=set_hyperparams.get('chemflag', chemflag),
                bzeroflag=set_hyperparams.get('bzeroflag', bzeroflag),
            )
        # Use default template and provided hyperparameters
        else:
            self.hyperparameters = SNAPBispectrumModelHyperparameters(
                cutoff_radius=cutoff_radius,
                rfac0=rfac0,
                two_j_max=two_j_max,
                quadratic=quadratic,
                rmin0=rmin0,
                wj=wj,
                radelem=radelem,
                use_zbl=use_zbl,
                wselfallflag=wselfallflag,
                chemflag=chemflag,
                bzeroflag=bzeroflag,
            )
            source_file_location = os.path.dirname(os.path.abspath(__file__))
            self.template = (f'{source_file_location}/'
                             'default_templates/snap_potential.in')

    @classmethod
    def _load_hyperparameters_from_files(
        cls,
        param_file: str,
        coeff_file: str,
    ) -> tuple[list[str], SNAPBispectrumModelHyperparameters]:
        """
        Load potential from existing parameter, coefficient, and model files

        This method reads the files specified in self.potential_files and
        extracts hyperparameters from the parameter file to be used later
        in the initialization process.

        :return: tuple of species list and hyperparameters
        """
        # Extract hyperparameters from snapparam file
        if not os.path.exists(param_file):
            raise FileNotFoundError(
                f"Required snapparam file not found: {param_file}")

        (
            cutoff_radius,
            rfac0,
            two_j_max,
            quadratic,
            rmin0,
            use_zbl,
            wselfallflag,
            chemflag,
            bzeroflag,
        ) = cls._extract_hyperparams_from_snapparam(param_file)

        # Extract species information from snapcoeff file if available
        if not os.path.exists(coeff_file):
            raise FileNotFoundError(
                f"Required snapcoeff file not found: {coeff_file}")

        (
            species,
            radelem,
            wj,
        ) = cls._extract_hyperparams_from_snapcoeff(coeff_file)

        # Store the extracted hyperparameters
        hyperparameters = SNAPBispectrumModelHyperparameters(
            cutoff_radius=cutoff_radius,
            rfac0=rfac0,
            two_j_max=two_j_max,
            quadratic=quadratic,
            rmin0=rmin0,
            wj=wj,
            radelem=radelem,
            use_zbl=use_zbl,
            wselfallflag=wselfallflag,
            chemflag=chemflag,
            bzeroflag=bzeroflag,
        )
        return species, hyperparameters

    @staticmethod
    def _parse_config_file(file_path, sections_to_parse=None) -> dict:
        """
        Generic helper to parse a configuration file with sections

        :param file_path: Path to the configuration file
        :param sections_to_parse: List of section names to parse (None means
            all)
        :return: Dictionary of extracted parameters by section
        """
        result = {}

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            current_section = None

            for line in lines:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Check for section headers
                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1]  # Remove brackets
                    if (sections_to_parse is None
                            or current_section in sections_to_parse):
                        result[current_section] = {}
                    continue

                # Skip if not in a section we care about
                if current_section not in result:
                    continue

                # Parse key-value pairs
                if "=" in line:
                    parts = line.split("=", 1)  # Split on first = only
                    if len(parts) >= 2:
                        key = parts[0].strip()
                        value = parts[1].strip()

                        # Process value based on key type
                        if key in [
                                "twojmax", "wselfallflag", "chemflag",
                                "bzeroflag", "quadraticflag"
                        ]:
                            int_val = int(value)
                            # Validate flag values
                            if key in [
                                    "wselfallflag", "chemflag", "bzeroflag",
                                    "quadraticflag"
                            ] and int_val not in [0, 1]:
                                raise ValueError(
                                    f"Invalid {key}: {int_val}, must be 0 or 1"
                                )
                            # Validate twojmax
                            if key == "twojmax" and int_val <= 0:
                                raise ValueError(f"Invalid twojmax: {int_val},"
                                                 " must be positive")
                            result[current_section][key] = int_val

                        elif key in ["rcutfac", "rfac0", "rmin0"]:
                            float_val = float(value)
                            # Validate positive values
                            if key == "rcutfac" and float_val <= 0:
                                raise ValueError(
                                    f"Invalid rcutfac: {float_val}, must be "
                                    "positive")
                            # Validate rfac0 range
                            if key == "rfac0" and (float_val <= 0
                                                   or float_val > 1.0):
                                raise ValueError(
                                    f"Invalid rfac0: {float_val}, must be "
                                    "between 0 and 1")
                            # Validate rmin0
                            if key == "rmin0" and float_val < 0:
                                raise ValueError(
                                    f"Invalid rmin0: {float_val}, must be "
                                    "non-negative")
                            result[current_section][key] = float_val

                        elif key in ["wj", "radelem"]:
                            # Handle possible list of values
                            values = value.split()
                            if len(values) > 1:
                                float_vals = [float(v) for v in values]
                                # Validate all values are positive
                                if any(v <= 0 for v in float_vals):
                                    raise ValueError(
                                        f"Invalid {key} values: {float_vals}, "
                                        "all must be positive")
                                result[current_section][key] = float_vals
                            else:
                                float_val = float(value)
                                # Validate the value is positive
                                if float_val <= 0:
                                    raise ValueError(
                                        f"Invalid {key}: {float_val}, must be "
                                        "positive")
                                result[current_section][key] = float_val
                        else:
                            # Default to string
                            result[current_section][key] = value

                # Special handling for ZBL in REFERENCE section
                if (current_section == "REFERENCE" and "pair_style" in line
                        and "zbl" in line):
                    result[current_section]["use_zbl"] = True

        except Exception as e:
            raise RuntimeError(
                f'Could not parse config file {file_path} due to {e}')

        return result

    @staticmethod
    def _extract_hyperparams_from_snapcoeff(
        coeff_file: str
    ) -> tuple[
            list[str],
            Union[list[float], float],
            Union[list[float], float],
    ]:
        """
        Extract species information from snapcoeff file

        This method parses a SNAP coefficient file to extract species names,
        atomic radii (radelem), and weight factors (wj). The format of
        snapcoeff files has a header line with number of species and
        coefficients, followed by species information lines and coefficient
        values.

        :param coeff_file: Path to the snapcoeff file
        :type coeff_file: str
        :return: Tuple of (species_list, radelem_values, wj_values)
        :rtype: tuple[list[str], Union[float, list[float]], Union[float, list
            [float]]]
        """
        species_list = []
        radelem_values = []
        wj_values = []

        with open(coeff_file, 'r') as f:
            lines = f.readlines()

            # Skip initial comment lines
            start_line = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith('#'):
                    start_line = i
                    break

            # Parse header line with number of species and coefficients
            header_parts = lines[start_line].strip().split()
            if len(header_parts) >= 2:
                num_species = int(header_parts[0])
                num_coeffs_per_species = int(header_parts[1])
            else:
                raise RuntimeError("Invalid header format in snapcoeff file")

            # Find and extract species information
            line_index = start_line + 1
            while line_index < len(lines) and len(species_list) < num_species:
                line = lines[line_index].strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            # Check if this looks like a species line
                            # (name followed by two numbers)
                            species_name = parts[0]
                            radelem = float(parts[1])
                            wj = float(parts[2])

                            species_list.append(species_name)
                            radelem_values.append(radelem)
                            wj_values.append(wj)

                            # Skip the coefficients for this species
                            line_index += num_coeffs_per_species + 1
                        except (ValueError, IndexError):
                            # Not a species line, continue to next line
                            line_index += 1
                    else:
                        line_index += 1
                else:
                    line_index += 1

        # convert to floats if length 1
        if len(wj_values) == 1:
            wj = wj_values[0]
        else:
            wj = wj_values
        if len(radelem_values) == 1:
            radelem = radelem_values[0]
        else:
            radelem = radelem_values

        # Validate wj - weights should be positive
        if isinstance(wj, float) and wj <= 0:
            raise ValueError(f"Invalid wj: {wj}, must be positive")
        elif isinstance(wj, list):
            if any(w <= 0 for w in wj):
                raise ValueError(f"Invalid wj values in list: {wj}, all "
                                 "weights must be positive")

        # Validate radelem - atomic radii should be positive
        if isinstance(radelem, float) and radelem <= 0:
            raise ValueError(f"Invalid radelem: {radelem}, must be positive")
        elif isinstance(radelem, list):
            if any(r <= 0 for r in radelem):
                raise ValueError(f"Invalid radelem values in list: {radelem}, "
                                 "all radii must be positive")

        return species_list, radelem, wj

    @classmethod
    def _extract_hyperparams_from_snapparam(
        cls,
        param_file: str,
    ) -> tuple[float, float, int, bool, float, bool, int, int, int]:
        """
        Extract hyperparameters from a snapparam file

        This method parses a SNAP parameter file to extract all listed
        hyperparameters. It supports the simple space-separated format without
        section headers. Parameters include cutoff radius, rfac0, twojmax,
        quadratic flag, etc.

        :param param_file: Path to the snapparam file
        :type param_file: str
        :return: Tuple of all hyperparameters (cutoff_radius, rfac0,
            two_j_max, quadratic, rmin0, use_zbl, wselfallflag,
            chemflag, bzeroflag)
        :rtype: tuple[float, float, int, bool, float, bool, int, int, int]
        """
        # Default values
        cutoff_radius = cls.default_cutoff_radius
        rfac0 = cls.default_rfac0
        two_j_max = cls.default_two_j_max
        quadratic = cls.default_quadratic
        rmin0 = cls.default_rmin0
        use_zbl = cls.default_use_zbl
        wselfallflag = cls.default_wselfallflag
        chemflag = cls.default_chemflag
        bzeroflag = cls.default_bzeroflag

        # First try the simple format:
        with open(param_file, 'r') as f:
            param_lines = f.readlines()

        # Check if this is simply formatted file (no section headers)
        for line in param_lines:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                # This looks like a section-based file
                raise RuntimeError('SNAP param file is incorrectly formatted')

            if not line:  # Skip empty lines
                continue

            # Split by whitespace
            parts = line.split()
            if len(parts) < 2:
                continue

            if len(parts) == 2:
                key = parts[0]
                value = parts[1]

                # Parse according to key
                if key == "rcutfac":
                    cutoff_radius = float(value)
                elif key == "rfac0":
                    rfac0 = float(value)
                elif key == "twojmax":
                    two_j_max = int(value)
                elif key == "quadraticflag":
                    quadratic = int(value) == 1
                elif key == "rmin0":
                    rmin0 = float(value)
                elif key == "wselfallflag":
                    wselfallflag = int(value)
                elif key == "chemflag":
                    chemflag = int(value)
                elif key == "bzeroflag":
                    bzeroflag = int(value)
            if len(parts) > 2:
                # reference section will contain zbl in pair_style line
                if parts[0] == '#' and 'zbl' in parts:
                    use_zbl = True

        # Validate the parsed values
        if cutoff_radius <= 0:
            raise ValueError(
                f"Invalid cutoff_radius: {cutoff_radius}, must be positive")

        if rfac0 <= 0 or rfac0 > 1.0:
            raise ValueError(
                f"Invalid rfac0: {rfac0}, must be between 0 and 1")

        if two_j_max <= 0:
            raise ValueError(
                f"Invalid two_j_max: {two_j_max}, must be positive")

        if rmin0 < 0:
            raise ValueError(f"Invalid rmin0: {rmin0}, must be non-negative")

        # Validate flag parameters (should be 0 or 1)
        for flag, flag_name in zip([wselfallflag, chemflag, bzeroflag],
                                   ['wselfallflag', 'chemflag', 'bzeroflag']):
            if flag not in [0, 1]:
                raise ValueError(
                    f"Invalid {flag_name}: {flag}, must be 0 or 1")

        return (cutoff_radius, rfac0, two_j_max, quadratic, rmin0, use_zbl,
                wselfallflag, chemflag, bzeroflag)

    @classmethod
    def _parse_template_file(cls, template_path: str) -> dict:
        """
        Parse a FitSNAP template file to extract hyperparameters

        This method reads a template file (which may include Jinja formatting)
        and extracts hardcoded hyperparameters.

        :param template_path: Path to the template file
        :return: Dictionary of extracted hyperparameters
        """
        extracted_params = {}

        try:
            # First we'll need to filter out Jinja template variables
            with open(template_path, 'r') as f:
                template_lines = []
                for line in f:
                    parts = line.strip().split('=', 1)
                    if len(parts
                           ) > 1 and '{{' in parts[1] and '}}' in parts[1]:
                        # Skip lines with Jinja variables
                        continue
                    template_lines.append(line)

            # Create a temporary file with non-Jinja lines
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w',
                                             delete=False) as temp_file:
                temp_file.writelines(template_lines)
                temp_path = temp_file.name

            # Parse the cleaned file using our generic parser
            try:
                config = cls._parse_config_file(temp_path,
                                                ["BISPECTRUM", "REFERENCE"])
            finally:
                os.unlink(temp_path)  # Clean up the temporary file

            # Convert parsed config to our parameter naming convention
            if "BISPECTRUM" in config:
                for param_name, param_value in config["BISPECTRUM"].items():
                    # Convert to expected parameter names
                    if param_name == "rcutfac":
                        extracted_params['cutoff_radius'] = param_value
                    elif param_name == "rfac0":
                        extracted_params['rfac0'] = param_value
                    elif param_name == "twojmax":
                        extracted_params['two_j_max'] = param_value
                    elif param_name == "quadraticflag":
                        extracted_params['quadratic'] = param_value == 1
                    elif param_name == "rmin0":
                        extracted_params['rmin0'] = param_value
                    elif param_name == "wselfallflag":
                        extracted_params['wselfallflag'] = param_value
                    elif param_name == "chemflag":
                        extracted_params['chemflag'] = param_value
                    elif param_name == "bzeroflag":
                        extracted_params['bzeroflag'] = param_value
                    elif param_name == "wj":
                        extracted_params['wj'] = param_value
                    elif param_name == "radelem":
                        extracted_params['radelem'] = param_value

            # Check for ZBL in REFERENCE section
            if "REFERENCE" in config and config["REFERENCE"].get("use_zbl"):
                extracted_params['use_zbl'] = True

        except Exception as e:
            raise RuntimeError("No hyperparameters could be extracted from "
                               f"template due to {e}")

        return extracted_params

    def load_potential(self, path: str):
        """
        Parameterize the potential by loading the potential files from a path.

        Note that this is specifically intended to be used in conjunction with
        save_potential, as both files will assume hard-coded file names. If you
        wish to load a potential using files generated external to the
        orchestrator, you should use `initialize_from_files`.

        :param path: Path to directory containing potential files
        :type path: str
        """
        from pathlib import Path

        # Ensure the path exists
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found: {path}")

        path_obj = Path(path)
        potential_name_found = None

        file_exts = [s.split('.')[1] for s in self._required_files]
        # Look for required files first to determine potential name
        for file_ext, file_name in zip(file_exts, self._required_files):
            # Find files with the expected extension
            matching_files = list(path_obj.glob(f"*.{file_ext}"))
            if not matching_files:
                raise FileNotFoundError(f"No {file_ext} file found in {path}")

            # Extract potential name from the first matching file
            if potential_name_found is None:
                potential_name_found = matching_files[0].stem

            # Check if all required files have consistent names
            expected_file = path_obj / f"{potential_name_found}.{file_ext}"
            if not expected_file.exists():
                raise FileNotFoundError(
                    f"Required file {expected_file} not found in {path}. "
                    f"Found {matching_files[0]} instead.")

            # Update potential_files with the found file
            self.potential_files[file_name] = str(expected_file)

        # Check if potential name needs updating
        if potential_name_found != self.potential_name:
            self.logger.info(
                f"Warning: Found potential with name '{potential_name_found}' "
                "which differs from Orchestrator standard: 'snap_potential'.")

        file_exts = [s.split('.')[1] for s in self._optional_files]
        # Check for optional files
        for file_ext, file_name in zip(file_exts, self._optional_files):
            if file_ext == "py":
                expected_file = path_obj / self.training_script_name
            else:
                expected_file = path_obj / f"{potential_name_found}.{file_ext}"
            if expected_file.exists():
                self.potential_files[file_name] = str(expected_file)

        # Mark that we have all required files
        self._has_required_files = True

        # Extract hyperparameters from the parameter file
        # these should all be set already, but ensure consistency
        species, hyperparameters = self._load_hyperparameters_from_files(
            self.potential_files["snap_potential.snapparam"],
            self.potential_files["snap_potential.snapcoeff"],
        )
        self.species = species
        self.hyperparameters = hyperparameters

    def _read_lammps_commands_from_mod(self) -> list[str]:
        """
        Read LAMMPS commands from .mod file and rewrite paths to be absolute.

        This helper method reads all non-comment, non-empty lines from the
        snap_potential.mod file and rewrites relative paths to coefficient
        and parameter files to be absolute paths for robustness.

        :returns: List of LAMMPS commands with absolute paths
        :rtype: list[str]
        :raises RuntimeError: If required files are not available
        :raises FileNotFoundError: If mod file is not found
        """
        from pathlib import Path

        mod_path = self.potential_files.get("snap_potential.mod")
        if not mod_path:
            raise RuntimeError(
                "snap_potential.mod is missing from potential_files")

        coeff_path = Path(self.potential_files.get("snap_potential.snapcoeff"))
        coeff_path_abs = coeff_path.resolve().as_posix()
        param_path = Path(self.potential_files.get("snap_potential.snapparam"))
        param_path_abs = param_path.resolve().as_posix()

        def _rewrite_cmd_paths(cmd: str) -> str:
            # Replace coeff and param file references with absolute paths
            parts = cmd.split()
            for i, part in enumerate(parts):
                # Check if this part looks like a file path for coeff or param
                if coeff_path.name in part:
                    parts[i] = coeff_path_abs
                elif param_path.name in part:
                    parts[i] = param_path_abs
            return ' '.join(parts)

        lmpcmds: list[str] = []
        with open(mod_path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                lmpcmds.append(_rewrite_cmd_paths(line))

        if not lmpcmds:
            raise RuntimeError(
                f"snap_potential.mod at {mod_path} contains no LAMMPS commands"
            )

        return lmpcmds

    def _initialize_calculator(self):
        """
        Set up the potential calculator based on current hyperparameters

        This method initializes the ASE calculator for SNAP potentials
        using the current hyperparameters and potential files. It imports
        the LAMMPS calculator from ASE and configures it with the appropriate
        pair style and coefficients for SNAP potentials.

        :raises RuntimeError: If required files are not available
        :raises Exception: If calculator setup fails
        """
        if self._has_required_files:
            try:
                # Import here to avoid dependency requirement when not needed
                from ase.calculators.lammpslib import LAMMPSlib

                # Get LAMMPS commands from mod file
                lmpcmds = self._read_lammps_commands_from_mod()

                # Create the calculator object
                self._potential_calculator = LAMMPSlib(lmpcmds=lmpcmds)
            except Exception as e:
                self.logger.info(
                    f"Warning: Failed to set up LAMMPS calculator: {e}")
                raise e
        else:
            raise RuntimeError("Required files are not available, "
                               "potential calculator not initialized.")

    def save_potential(
        self,
        path: str,
        makedirs: bool = False,
    ) -> Union[str, list[str]]:
        """
        Save the potential and other necessary data to disk.

        Note that this uses a strict naming convention for integration with
        load_potential().

        :param path: Directory path where the model should be saved
        :param makedirs: If True, creates folder if it doesn't exist. Default
            is False.
        :returns: List of paths to the saved model files
        """
        import shutil
        from pathlib import Path

        if not self._has_required_files:
            raise FileNotFoundError("The Potential is missing required files!"
                                    " It probably needs to be trained first.")

        # Create the directory if it doesn't exist and makedirs is True
        if makedirs:
            os.makedirs(path, exist_ok=True)
        elif not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found: {path}")

        path_obj = Path(path)
        saved_files = []

        # Copy all required and optional files to the destination
        all_files = self._required_files + self._optional_files

        for file_name in all_files:
            file_ext = file_name.split('.')[1]
            source_path = self.potential_files.get(file_name)

            # Skip if file doesn't exist
            if not source_path:
                continue

            # Destination path with standardized naming
            dest_file = path_obj / f"{self.potential_name}.{file_ext}"

            # Copy the file
            try:
                shutil.copy2(source_path, dest_file)
                saved_files.append(str(dest_file))
            except Exception as e:
                self.logger.info(
                    f"Warning: Failed to copy {file_name} file: {e}")

        return saved_files

    def _write_training_script(
        self,
        save_path: str,
        dataset_list: list,
        storage: Storage,
        energy_weight: float,
        force_weight: float,
        stress_weight: float,
        train_frac: float,
        test_frac: float,
        val_frac: float,
        per_atom_weights: Optional[Union[list[np.ndarray], str]] = None,
    ) -> str:
        """
        Write a script to run the potential training outside of memory

        This is a helper function for generating a script, training_script.py,
        which can be executed via a scheduler or offline. It additionally saves
        needed additional files with it, such as a weights.txt data file.

        :param save_path: path where the training script will be written
        :type save_path: str
        :param dataset_list: list of dataset handles which should be used for
            the training procedure
        :type dataset_list: list of str
        :param storage: an instance of the storage class, which contains the
            datasets in dataset_list
        :type storage: Storage
        :param energy_weight: Weight for energy terms in the loss function
        :type energy_weight: float
        :param force_weight: Weight for force terms in the loss function
        :type force_weight: float
        :param stress_weight: Weight for stress terms in the loss function
        :type stress_weight: float
        :param train_frac: Fraction of data to use for training
        :type train_frac: float
        :param test_frac: Fraction of data to use for testing
        :type test_frac: float
        :param val_frac: Fraction of data to use for validation
        :type val_frac: float
        :param per_atom_weights: Controls per-atom weighting
        :type per_atom_weights: Union[list[np.ndarray], str]
        :returns: name of the script that is generated (training_script.py)
        :rtype: str
        """
        # Make sure save_path is absolute
        full_save_path = os.path.abspath(save_path)

        # Create the settings file if it doesn't exist
        settings_path = os.path.join(full_save_path, "snap_potential.in")
        if not os.path.exists(settings_path):
            settings_path = self.write_settings_file(
                full_save_path,
                # use these values if weights are greater than 0
                energy_weight > 0,
                force_weight > 0,
                stress_weight > 0,
            )
        # Import lines
        import_lines = ('from orchestrator.utils.setup_input import '
                        'init_and_validate_module_type\n')

        # Potential dictionary with all hyperparameters
        potential_dict = {
            'potential_type': 'SNAP',
            'potential_args': {
                'species': self.species,
                'cutoff_radius': self.hyperparameters.cutoff_radius,
                'rfac0': self.hyperparameters.rfac0,
                'two_j_max': self.hyperparameters.two_j_max,
                'quadratic': self.hyperparameters.quadratic,
                'rmin0': self.hyperparameters.rmin0,
                'wj': self.hyperparameters.wj,
                'radelem': self.hyperparameters.radelem,
                'use_zbl': self.hyperparameters.use_zbl,
                'wselfallflag': self.hyperparameters.wselfallflag,
                'chemflag': self.hyperparameters.chemflag,
                'bzeroflag': self.hyperparameters.bzeroflag,
                'template': settings_path,
            }
        }

        # Initialize potential
        init_potential = ('potential = init_and_validate_module_type('
                          f'"potential", {potential_dict}, '
                          'single_input_dict=True)\n')

        # Storage dictionary
        storage_dict = {
            'storage_type':
            storage.factory_token if hasattr(storage, 'factory_token') else
            storage.__class__.__name__,
            'storage_args':
            storage.storage_init_args
            if hasattr(storage, 'storage_init_args') else {}
        }

        # Initialize storage
        init_storage = ('storage = init_and_validate_module_type("storage", '
                        f'{storage_dict}, single_input_dict=True)')

        # Handle per-atom weights
        # If per_atom_weights is a numpy array or list, save it to a file
        if isinstance(per_atom_weights, list):
            np.savez(f'{full_save_path}/weights.npz', *per_atom_weights)
            per_atom_weights_for_script = f"{full_save_path}/weights.npz"
        # If per_atom_weights is a string (path), ensure it's an absolute path
        elif isinstance(per_atom_weights, str):
            if not os.path.isabs(per_atom_weights):
                per_atom_weights_for_script = os.path.join(
                    full_save_path, per_atom_weights)
            else:
                per_atom_weights_for_script = per_atom_weights
        elif per_atom_weights is not None:
            raise ValueError('per_atom_weights must be a list or str')
        else:
            per_atom_weights_for_script = None

        # Construct the training call
        # this will construct the settings file based on the potential args
        construct_and_train = (
            f'model_path, error = potential.train('
            f'dataset_list={dataset_list},'
            f'storage=storage,'
            f'scheduler=None,'  # Don't use a scheduler
            f'energy_weight={energy_weight},'
            f'force_weight={force_weight},'
            f'stress_weight={stress_weight},'
            f'train_frac={train_frac},'
            f'test_frac={test_frac},'
            f'val_frac={val_frac},'
            'write_training_script=False,')

        if per_atom_weights_for_script:
            construct_and_train += ('per_atom_weights='
                                    f'"{per_atom_weights_for_script}")')
        else:
            construct_and_train += 'per_atom_weights=None)'

        # Combine the script components
        script = '\n'.join([
            import_lines, init_storage, init_potential, construct_and_train,
            '\nprint(f"Training complete. Model saved at: {model_path}")',
            'print(f"Training error: {error}")'
        ])

        # Write the script to file
        script_path = os.path.join(full_save_path, self.training_script_name)
        with open(script_path, 'w') as fout:
            fout.write(script)

        self.logger.info(f"Created training script: {script_path}")
        return self.training_script_name

    def train(
        self,
        dataset_list: list[str],
        storage: Storage,
        scheduler: Scheduler,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        stress_weight: float = 1.0,
        train_frac: float = 1.0,
        test_frac: float = 0.0,
        val_frac: float = 0.0,
        per_atom_weights: Optional[Union[list[np.ndarray], str]] = None,
        **kwargs,
    ) -> tuple[str, float]:
        """
        Train the potential using the provided data.

        Note that this function will automatically apply per-atom masking (set
        loss contribution to zero for masked atoms)if SELECTION_MASK_KEY is set
        in the atoms.info dictionaries.

        :param dataset_list: list of dataset handles to use for training
        :type dataset_list: list[str]
        :param storage: Storage object to access training data
        :type storage: Storage
        :param scheduler: Scheduler object for job management
        :type scheduler: Scheduler
        :param energy_weight: Weight for energy terms in the loss function
        :type energy_weight: float
        :param force_weight: Weight for force terms in the loss function
        :type force_weight: float
        :param stress_weight: Weight for stress terms in the loss function
        :type stress_weight: float
        :param train_frac: Fraction of data to use for training
        :type train_frac: float
        :param test_frac: Fraction of data to use for testing
        :type test_frac: float
        :param val_frac: Fraction of data to use for validation
        :type val_frac: float
        :param per_atom_weights: Controls per-atom weighting
        :type per_atom_weights: Union[list[np.ndarray], str]
        :param kwargs: Additional potential-specific training parameters

        :returns: Tuple containing (path to trained potential, training error)
        :rtype: tuple[str, float]
        """
        # Check if required inputs are provided
        if dataset_list is None or storage is None:
            raise ValueError(
                'A storage object and list of dataset handles are required!')

        if train_frac < 1:
            raise ValueError(
                '`train_frac` < 1 is not supported for FitSNAP yet, '
                f'but was set to {train_frac}')
        if test_frac > 0:
            raise ValueError('`test_frac` is not supported for FitSNAP yet, '
                             f'but was set to {test_frac}')
        if val_frac > 0:
            raise ValueError('`val_frac` is not supported for FitSNAP yet, '
                             f'but was set to {val_frac}')

        # Create working directory
        if scheduler is None:
            # this is the case when running from a training script to avoid
            # making nested dirs
            working_path = '.'
            # in this case the settings file has already been generated
            settings_file = 'snap_potential.in'
        else:
            working_path = scheduler.make_path(self.__class__.__name__,
                                               'training')
            settings_file = self.write_settings_file(
                working_path,
                # use these values if weights are greater than 0
                energy_weight > 0,
                force_weight > 0,
                stress_weight > 0,
            )

        snap = FitSnap(settings_file, arglist=["--overwrite"])

        # Convert dataset list to a list if it isn't already
        if not isinstance(dataset_list, list):
            dataset_list = [dataset_list]

        # Collect all configurations from the dataset
        self.logger.info('Reading training data from storage')
        combined_dataset = []
        for dataset_handle in dataset_list:
            configs = storage.get_data(dataset_handle)
            combined_dataset.extend(configs)

        # Format data for FitSnap
        snap.data = [
            self._collate_fitsnap_data(
                atoms,
                energy_weight,
                force_weight,
                stress_weight,
            ) for atoms in combined_dataset
        ]
        self.logger.info(f"Found {len(snap.data)} configurations")

        # Handle per-atom weighting - convert to concatenated list
        if per_atom_weights is not None:
            if isinstance(per_atom_weights, str):
                weights_path = per_atom_weights
                npzfile = np.load(per_atom_weights)
                per_atom_weights = [npzfile[name] for name in npzfile.files]
            elif isinstance(per_atom_weights, list):
                weights_path = None
            else:
                raise TypeError('per_atom_weights not a supported type!')

            for atoms, weights in zip(combined_dataset,
                                      per_atom_weights,
                                      strict=True):
                assert len(atoms) == len(weights), (
                    "Per-atom weight array "
                    f"length ({len(weights)}) does not match atoms length "
                    f"({len(atoms)}). Maybe they were provided in the wrong "
                    "order?")
            weights = np.concatenate(per_atom_weights)
            per_atom_fit = True
        else:
            weights_path = None
            per_atom_fit = False

        # Process the configurations
        snap.process_configs()

        # Apply per-atom weights if enabled
        if per_atom_fit:
            row_types = snap.pt.fitsnap_dict['Row_Type']
            manually_created_w_array = np.zeros(
                len(snap.pt.shared_arrays['w'].array))
            force_rows = [
                True if row == 'Force' else False for row in row_types
            ]

            assert (len(weights) * 3) == sum(force_rows), \
                f"{len(weights)} weights given, need {sum(force_rows) / 3}"

            energy_rows = [
                True if row == 'Energy' else False for row in row_types
            ]
            stress_rows = [
                True if row == 'Stress' else False for row in row_types
            ]

            # Modify energy weights based on per-atom weights
            if energy_weight > 1 and np.any(energy_rows):
                force_row_counter = 0
                energy_idxs = np.flatnonzero(energy_rows)
                for config, energy_idx in zip(combined_dataset, energy_idxs):
                    num_atoms = len(config)
                    use_all_atoms = np.all(
                        weights[force_row_counter:force_row_counter
                                + num_atoms])
                    force_row_counter += num_atoms
                    if use_all_atoms:
                        manually_created_w_array[energy_idx] = energy_weight
                    else:
                        manually_created_w_array[energy_idx] = 0
            else:
                manually_created_w_array[energy_rows] = energy_weight

            manually_created_w_array[force_rows] = force_weight * \
                np.array([val for val in weights.tolist() for _ in range(3)])
            manually_created_w_array[stress_rows] = stress_weight

            snap.pt.shared_arrays['w'].array = manually_created_w_array

        # Perform the fit
        snap.solver.perform_fit()

        # Analyze error metrics
        snap.solver.error_analysis()

        # Save the trained model
        self._write_trained_files(snap, working_path)

        # Write a training script for documentation and reproducibility
        # Only write it if not explicitly disabled in kwargs
        if kwargs.get('write_training_script', True):
            if weights_path:
                # revert value back to input string to avoid re-saving
                per_atom_weights = weights_path
            self._write_training_script(
                working_path,
                dataset_list,
                storage,
                energy_weight,
                force_weight,
                stress_weight,
                train_frac,
                test_frac,
                val_frac,
                per_atom_weights,
            )

        # Initialize the calculator with the trained model
        self._initialize_calculator()

        # Return trained model path and error metric
        return working_path, snap.solver.errors.get('MAE_Energy', 0.0)

    def _collate_fitsnap_data(
        self,
        atoms: Atoms,
        energy_weight: float,
        force_weight: float,
        stress_weight: float,
    ) -> dict:
        """
        Function to organize fitting data for FitSNAP from ASE atoms objects.

        :param atoms: ASE atoms object for a single configuration of atoms.
        :param energy_weight: Weight for energy terms in the loss function
        :param force_weight: Weight for force terms in the loss function
        :param stress_weight: Weight for stress terms in the loss function
        :returns: data dictionary in FitSNAP format for a single configuration.
        :rtype: dict
        """
        from ..utils.data_standard import ENERGY_KEY, FORCES_KEY, STRESS_KEY
        from fitsnap3lib.scrapers.ase_funcs import get_apre

        # Transform ASE cell to be appropriate for LAMMPS
        apre = get_apre(cell=atoms.cell)
        r = np.dot(np.linalg.inv(atoms.cell), apre)
        positions = np.matmul(atoms.get_positions(), r)
        cell = apre.T

        # Make a data dictionary for this config
        data = {}
        data['Group'] = None
        data['File'] = None

        # Handle stress tensor
        if STRESS_KEY in atoms.info:
            data['Stress'] = np.array(atoms.info[STRESS_KEY])
            if data['Stress'].shape[0] == 6:
                data['Stress'] = self._convert_to_3x3_stress_tensor(
                    data['Stress'])
            elif data['Stress'].shape != (3, 3):
                raise ValueError(
                    'Stress tensor not supplied as 6, or 3x3 formats')
        else:
            # Default to zeros if no stress data is available
            data['Stress'] = np.zeros((3, 3))

        data['Positions'] = positions
        data['Energy'] = atoms.info.get(ENERGY_KEY, 0.0)
        data['AtomTypes'] = atoms.get_chemical_symbols()
        data['NumAtoms'] = len(atoms)

        # Get forces or set to zeros if not available
        if FORCES_KEY in atoms.arrays:
            data['Forces'] = atoms.arrays[FORCES_KEY]
        else:
            data['Forces'] = np.zeros((len(atoms), 3))

        data['QMLattice'] = cell
        data['test_bool'] = 0
        data['Lattice'] = cell
        data['Rotation'] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        data['Translation'] = np.zeros((len(atoms), 3))

        # Inject the weights
        data['eweight'] = energy_weight
        data['fweight'] = force_weight
        data['vweight'] = stress_weight

        return data

    def _convert_to_3x3_stress_tensor(
        self,
        stress_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Convert the (6,) stress vector to 3x3 expected by FitSNAP

        :param stress_vector: 6 stress components (Voigt notation)
        :type stress_vector: np.ndarray
        :returns: transformed matrix in full 3x3 format
        :rtype: np.ndarray
        """
        return np.array([
            [stress_vector[0], stress_vector[5], stress_vector[4]],
            [stress_vector[5], stress_vector[1], stress_vector[3]],
            [stress_vector[4], stress_vector[3], stress_vector[2]],
        ])

    def write_settings_file(
        self,
        output_dir: str,
        use_energy: bool = True,
        use_force: bool = True,
        use_stress: bool = True,
    ) -> str:
        """
        Generate a FitSNAP input file from template using class attributes.

        This method creates a FitSNAP input file by filling in a template with
        values from the SNAPPotential class hyperparameters and other
        attributes. The generated file follows the FitSNAP configuration format
        with sections for BISPECTRUM parameters, CALCULATOR settings, SOLVER
        options, OUTFILE specifications, and REFERENCE information.

        The template is rendered using Jinja2 templating with replacements for
        various parameters like number of types, bispectrum components, cutoff
        radius, etc. The method automatically handles the conversion of single
        values to lists when appropriate for parameters like wj and radelem.

        :param output_dir: Directory where the file should be written
        :type output_dir: str
        :param use_energy: Whether to include energy terms in training
            (energy_flag=1)
        :type use_energy: bool
        :param use_force: Whether to include force terms in training
            (force_flag=1)
        :type use_force: bool
        :param use_stress: Whether to include stress terms in training
            (stress_flag=1)
        :type use_stress: bool
        :returns: Path to the generated settings file
        :rtype: str
        """
        import periodictable
        from ..utils.templates_jinja import render_template_to_file

        # Get atomic numbers for species (used for ZBL potential)
        species_atomic_numbers = []
        for element in self.species:
            try:
                atomic_number = getattr(periodictable.elements, element).number
                species_atomic_numbers.append(atomic_number)
            except AttributeError:
                # Handle the case where element is not found in periodictable
                self.logger.warning(
                    f"Could not find atomic number for {element}")
                species_atomic_numbers.append(0)  # Placeholder

        # Get ZBL flag from hyperparameters
        use_zbl = self.hyperparameters.use_zbl

        # Process per-species parameters
        num_species = len(self.species)

        # Handle wj (weights for each species)
        if isinstance(self.hyperparameters.wj, list):
            wj_values = self.hyperparameters.wj
        else:
            wj_values = [self.hyperparameters.wj] * num_species

        # Handle radelem (radii for each species)
        if isinstance(self.hyperparameters.radelem, list):
            radelem_values = self.hyperparameters.radelem
        else:
            radelem_values = [self.hyperparameters.radelem] * num_species

        # set training flags
        energy_flag = 1 if use_energy else 0
        force_flag = 1 if use_force else 0
        stress_flag = 1 if use_stress else 0

        # Create replacements dictionary with values from class attributes
        replacements = {
            # BISPECTRUM section
            'num_types':
            num_species,
            'two_j_max':
            self.hyperparameters.two_j_max,
            'rcutfac':
            self.hyperparameters.cutoff_radius,
            'rfac0':
            self.hyperparameters.rfac0,
            'rmin0':
            self.hyperparameters.rmin0,
            'wj':
            wj_values if num_species > 1 else self.hyperparameters.wj,
            'radelem':
            radelem_values
            if num_species > 1 else self.hyperparameters.radelem,
            'types':
            ' '.join(self.species),
            'quadraticflag':
            1 if self.hyperparameters.quadratic else 0,
            'wselfallflag':
            self.hyperparameters.wselfallflag,
            'chemflag':
            self.hyperparameters.chemflag,
            'bzeroflag':
            self.hyperparameters.bzeroflag,

            # CALCULATOR section
            'train_energy':
            energy_flag,
            'train_force':
            force_flag,
            'train_stress':
            stress_flag,

            # OUTFILE section
            'metrics_file':
            'snap_potential.md',
            'potential_name':
            'snap_potential',

            # REFERENCE section - use generic defaults
            'units':
            'metal',
            'atom_style':
            'atomic',
            'use_zbl':
            use_zbl,
            'species_atomic_numbers':
            species_atomic_numbers,
        }

        # Generate the output filename
        output_file_name = "snap_potential.in"

        # Render the template to a file
        rendered_file = render_template_to_file(
            template_path=self.template,
            output_dir=output_dir,
            replacements=replacements,
            output_file_name=output_file_name)

        # Return the full path to the generated file
        return os.path.join(output_dir, rendered_file)

    def _write_trained_files(self, snap_obj: FitSnap, path: str):
        """
        Save the trained SNAP potential files to disk.

        This method takes the trained FitSnap object and writes all required
        potential files to the specified directory. It sets the output paths
        in the FitSnap configuration, generates the potential files
        (snapcoeff, snapparam, mod, md), and updates the internal state to
        reflect the available files.

        :param snap_obj: The trained FitSnap object containing model
            coefficients and errors
        :type snap_obj: FitSnap
        :param path: Directory path where the trained model files should be
            saved
        :type path: str
        :return: None
        """
        self.logger.info(f'Saving model state in {path}')
        vars(snap_obj.config.sections['OUTFILE'])['potential_name'] = \
            path + '/snap_potential'
        vars(snap_obj.config.sections['OUTFILE'])['metric_file'] = \
            path + '/snap_potential.md'

        fit_coefficients = snap_obj.solver.fit
        errors = snap_obj.solver.errors
        snap_obj.output.output(fit_coefficients, errors)

        # Update potential_files with paths to all required and optional files
        self.potential_files = {
            "snap_potential.snapparam": f"{path}/snap_potential.snapparam",
            "snap_potential.snapcoeff": f"{path}/snap_potential.snapcoeff",
            "snap_potential.mod": f"{path}/snap_potential.mod",
            "snap_potential.md": f"{path}/snap_potential.md",
            "snap_potential.in": f"{path}/snap_potential.in",
            "training_script.py": f"{path}/{self.training_script_name}",
        }

        # Set _has_required_files to True since we now have all required files
        self._has_required_files = True

        self.training_hash = snap_obj.config.hash
        self.logger.info(
            f'Output fitsnap files with Hash: {self.training_hash}')

    def submit_train(
        self,
        dataset_list: list[str],
        storage: Storage,
        scheduler: Scheduler,
        job_details: Optional[dict] = None,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        stress_weight: float = 1.0,
        train_frac: float = 1.0,
        test_frac: float = 0.0,
        val_frac: float = 0.0,
        per_atom_weights: Optional[Union[list[np.ndarray], str]] = None,
        **kwargs,
    ) -> Union[str, int]:
        """
        Train the potential using the provided data via a submitted job

        :param dataset_list: list of dataset handles to use for training
        :type dataset_list: list[str]
        :param storage: Storage object to access training data
        :type storage: Storage
        :param scheduler: Scheduler object for job management
        :type scheduler: Scheduler
        :param job_details: information controlling job submission
        :type job_details: dict
        :param energy_weight: Weight for energy terms in the loss function
        :type energy_weight: float
        :param force_weight: Weight for force terms in the loss function
        :type force_weight: float
        :param stress_weight: Weight for stress terms in the loss function
        :type stress_weight: float
        :param train_frac: Fraction of data to use for training
        :type train_frac: float
        :param test_frac: Fraction of data to use for testing
        :type test_frac: float
        :param val_frac: Fraction of data to use for validation
        :type val_frac: float
        :param per_atom_weights: Controls per-atom weighting
        :type per_atom_weights: Union[list[np.ndarray], str]
        :param kwargs: Additional potential-specific training parameters
        :returns: calc id of the submitted job
        :rtype: tuple[str, float]
        """
        if dataset_list is None or storage is None:
            raise ValueError('A storage object and list of dataset handles'
                             ' are required!')
        if not isinstance(dataset_list, list):
            dataset_list = [dataset_list]

        if train_frac < 1:
            raise ValueError(
                '`train_frac` < 1 is not supported for FitSNAP yet, '
                f'but was set to {train_frac}')
        if test_frac > 0:
            raise ValueError('`test_frac` is not supported for FitSNAP yet, '
                             f'but was set to {test_frac}')
        if val_frac > 0:
            raise ValueError('`val_frac` is not supported for FitSNAP yet, '
                             f'but was set to {val_frac}')

        save_path = scheduler.make_path(self.__class__.__name__, 'training')

        script_filename = self._write_training_script(
            save_path,
            dataset_list,
            storage,
            energy_weight,
            force_weight,
            stress_weight,
            train_frac,
            test_frac,
            val_frac,
            per_atom_weights,
        )

        if job_details is None:
            job_details = {}

        job_details['custom_preamble'] = 'python'

        calc_id = scheduler.submit_job(
            script_filename,
            save_path,
            job_details=job_details,
        )

        return calc_id

    def load_from_submitted_training(
        self,
        calc_id: Union[str, int],
        scheduler: Scheduler,
    ):
        """
        Load a potential that was trained via a submitted job.

        This method waits for the training job to complete (if it hasn't
        already) and then loads the trained potential files from the job's
        output directory. It uses the load_potential method to initialize the
        SNAP potential from the trained files.

        :param calc_id: The calculation ID returned by submit_train
        :type calc_id: Union[str, int]
        :param scheduler: Scheduler object used to manage the job
        :type scheduler: Scheduler
        :return: None
        """
        # include checks that training finished appropriately
        scheduler.block_until_completed(calc_id)

        self.load_potential(scheduler.get_job_path(calc_id))

    def get_lammps_commands(self) -> str:
        """
        Extract required commands to inject into a LAMMPS input file to
        enable using external potentials with direct LAMMPS interface

        :returns: Commands to inject into LAMMPS input file to directly
            run a LAMMPS simulation with an external SNAP potential
        :rtype: str
        """
        if not self.potential_files:
            raise ValueError(
                "potential_files is not set on this Potential instance")

        # Use the helper function to get all LAMMPS commands from mod file
        lmpcmds = self._read_lammps_commands_from_mod()

        # Join all commands with newlines
        return "\n".join(lmpcmds)
