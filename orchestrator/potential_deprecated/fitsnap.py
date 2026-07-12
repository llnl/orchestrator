import os
import glob
import itertools
import periodictable
import configparser
from typing import List, Optional
from .potential_base import Potential
from fitsnap3lib.fitsnap import FitSnap
from ..utils.restart import restarter
from ..utils.exceptions import InstallPotentialError


class FitSnapPotential(Potential):
    """
    Build a potential using FitSnap

    All parameters defining the snap potential are defined in the
    settings file described in the trainer_args dict.

    NOTE: If building FitSNAP potentials as portable models, they will
    default to the OpenKIM model driver SNAP__MD_536750310735_000,
    which does not support explicit multielement potentials.
    Potentials that wish to use explicit multielement speicies should be
    saved as simulator-models instead.

    :param trainer_args: dict with the input parameters and their values as k,v
        pairs. Parameters include:
    :param settings_path:
    """

    def __init__(
        self,
        species: list[str],
        model_driver: str,
        settings_path: str,
        kim_api: str = 'kim-api-collections-management',
        kim_item_type: str = "simulator-model",
        parameter_path: str = None,
        kim_id: str = None,
        model_name_prefix: str = "FitSNAP_Potential_Orchestrator_Generated",
        param_files: Optional[list] = None,
        training_files: Optional[list] = None,
        potential_files: Optional[list] = None,
        checkpoint_file: Optional[str] = './orchestrator_checkpoint.json',
        checkpoint_name: Optional[str] = 'potential',
        **kwargs,
    ):
        """
        initialization of the FitSnap potential with trainer_args dict

        :param kim_id: kimcode to represent the item
        :type kim_id: str
        :param species: list of strings containing element symbols
        :type species: list[str]
        :param model_driver: driver needed to run the potential
        :type model_driver: str
        :param kim_api: path to the kim-api-collections-manager
            executable.
        :type kim_api: str
        :param kim_item_type: what type of kim object to create an ID for.
            For potentials, this should be either "portable-model" or
            "simulator-model", depending on whether the model uses a driver
            to implement its calculations, or runs commands in a simulator
            program (e.g. lammps, ASE) respectively.
        :type kim_item_type: str
        :param settings_path: FitSnap settings file that include
            parameters for various sections such as bispectrum,
            calculator, solver
        :type settings_path: str
        :param parameter_path: path where the potential's param_files
            will be written
        :type parameter_path: str
        :param param_files: list of file paths to the parameter files
            of the potential. May be order-sensitive.
        :type param_files: list[str]
        :param training_files: list of files associated with the
            training of the potential
        :type training_files: list[str]
        :param potential_files: list of all files associated with
            the potential, including the superset of param_files,
            training_files, and any other auxillary files.
        :type potential_files: list[str]
        :param checkpoint_file: file name to save checkpoints in
        :type checkpoint_file: str
        :param checkpoint_name: name of the checkpointed potential
        :type checkpoint_name: str
        """
        if settings_path:
            self.settings = settings_path
        else:
            n = len(species)
            # rough implementation of reasonable defaults
            # TODO: allow combining partial definitions with these defaults
            self.settings = {
                "BISPECTRUM": {
                    "numTypes": n,
                    "twojmax": [6] * n,
                    "wj": [1.0 - (0.1 * i / n) for i in range(0, n)],
                    "radelem": [0.5 - (0.1 * i / n) for i in range(0, n)],
                    "types": species
                },
                "CALCULATOR": {
                    "calculator": "LAMMPSSNAP",
                    "energy": 1,
                    "force": 1,
                    "stress": 1
                },
                "SOLVER": {
                    "solver": "SVD",
                    "compute_testerrs": 1,
                    "detailed_errors": 1
                },
                "OUTFILE": {
                    "metrics": "fitsnap_potential.md",
                    "potential": "fitsnap_potential"
                },
                "REFERENCE": {
                    "units": "metal"
                },
                # possibly add default zbl using logic below to order
            }

        if kim_id:
            self.kim_id = kim_id

        self.model = None

        self.model_type = "snap"

        if parameter_path is not None:
            root_exists = os.path.isdir(os.path.split(parameter_path)[0])
            # should we "if not root_exists: os.makedirs(parameter_path)" ?

            param_exists = os.path.isfile(f'{parameter_path}.snapparam')
            if root_exists and (os.path.isdir(parameter_path)
                                or not param_exists):
                raise ValueError(
                    'parameter_path should be of form path/potential_prefix')
        self.parameter_path = parameter_path

        self.kim_api_compatible = True

        if not model_driver:
            # default to openkim snap model driver if none supplied
            model_driver = "SNAP__MD_536750310735_000"

        self.checkpoint_name = checkpoint_name

        self.trainer_args = {
            "settings_path": settings_path,
            "parameter_path": parameter_path,
            "kim_api": kim_api
        }

        self.name = "fitsnap_potential"

        super().__init__(
            kim_id,
            species,
            model_driver,
            model_name_prefix=model_name_prefix,
            param_files=param_files,
            training_files=training_files,
            potential_files=potential_files,
            kim_api=kim_api,
            kim_item_type=kim_item_type,
            checkpoint_file=checkpoint_file,
            checkpoint_name=self.checkpoint_name,
        )
        self.logger.info('Finished instantiating FitSnap potential')

    def checkpoint_potential(self):
        """
        checkpoint the potential module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """

        save_dict = {
            self.checkpoint_name: {
                'parameter_path': self.parameter_path,
            }
        }
        try:
            if self.kim_id is not None:
                save_dict[self.checkpoint_name]['kim_id'] = self.kim_id
        except AttributeError:
            pass
        restarter.write_checkpoint_file(self.checkpoint_file, save_dict)

    def restart_potential(self):
        """
        restart the potential module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        restart_dict = restarter.read_checkpoint_file(
            self.checkpoint_file,
            self.checkpoint_name,
        )
        self.parameter_path = restart_dict.get('parameter_path',
                                               self.parameter_path)
        self.kim_id = restart_dict.get('kim_id', self.kim_id)
        self.build_potential()

    def build_potential(self) -> FitSnap:
        """
        Build a snap potential using FitSnap

        The settings file that includes parameters for the snap potential
        were passed to the object in __init__ as the trainer_args dict.
        In addition to returning the model,
        this method also sets it as the objects self.model attribute.

        :returns: fitsnap model parameterized by parameters in the settings
            file
        :rtype: fitsnap instance
        """

        self.logger.info('Building potential using given parameters')

        snap = FitSnap(self.settings, arglist=["--overwrite"])

        self.logger.info('Finished constructing FitSnap potential')
        self.model = snap

        if self.kim_item_type == "portable-model":
            if self.model_driver == "SNAP__MD_536750310735_000":
                if (snap.config.sections['BISPECTRUM'].wselfallflag != 0
                        or snap.config.sections['BISPECTRUM'].chemflag != 0
                        or snap.config.sections['BISPECTRUM'].bnormflag != 0
                        or snap.config.sections['BISPECTRUM'].switchinnerflag
                        != 0):
                    raise ValueError("wselfallflag, chemflag, bnormflag,"
                                     " and switchinnerflag are not supported"
                                     " by the KIM Model driver version 000"
                                     " for SNAP, which is the current KIM "
                                     "default. Version 001 will become the "
                                     "new default once it is released.")

            elif self.model_driver == "SNAP__MD_536750310735_001":
                # remove check if this gets added to the driver
                if snap.config.sections['BISPECTRUM'].switchinnerflag != 0:
                    raise ValueError("switchinnerflag is not supported "
                                     "by the KIM Model driver version 001"
                                     " for SNAP.")
                # check to be removed once development 001 driver fixes bug
                if snap.config.sections['BISPECTRUM'].quadraticflag != 0:
                    raise ValueError("The developmental KIM Model driver "
                                     "version 001 for SNAP currently produces"
                                     " incorrect forces for quadratic models."
                                     " Please use version 000 for quadratic.")
        else:
            pass

        return self.model

    def load_potential(self, path: str):
        """
        parameterize the potential based on an existing potential at path

        :param path: path string including filename where potential resides
        """
        raise NotImplementedError

    def get_potential_files(
        self,
        destination_path: str,
        kim_id: str,
        include_dependencies: bool = False,
    ) -> str:
        """
        Load a KIM model from a kimkit repository using the KIM ID

        :param destination_path: path to save the resulting .txz file
        :type destination_path: str
        :param kim_id: kimcode of the item to be retrieved
        :type kim_id: str
        :param include_dependencies: switch to include drivers of portable
            models, tests, |default| ``False``
        :type include_dependencies: bool
        :returns: path to the tar archive at destination_path
        :rtype: str
        """
        tarfile_name = super(FitSnapPotential, self).get_potential_files(
            destination_path=destination_path,
            kim_id=kim_id,
            include_dependencies=include_dependencies,
        )

        return tarfile_name

        # TODO: unpack tarfile to location, set as parameter_path

    def _write_potential_to_file(self, path):
        """
        save the current potential path to a file in a specified path

        :param path: path including filename where the potential is to be
            written
        :type path: str
        """

        if self.parameter_path is not None:
            param_path, name = os.path.split(self.parameter_path)
            if (os.path.abspath(param_path) != os.path.abspath(path)):
                os.makedirs(path, exist_ok=True)
                os.system(f'cp -r {param_path}/* {path}')
                new_parameter_path = os.path.join(path, name)
                self.parameter_path = new_parameter_path
            else:
                pass  # files already exist in specified location

        else:
            raise InstallPotentialError(
                'Potential object has no .parameter_path specifying the'
                'location of saved files from a successful training.')

    def _save_potential_to_kimkit(
        self,
        kim_id: str = None,
        species: List[str] = None,
        model_name_prefix: str = None,
        param_files: List[str] = None,
        training_files: Optional[List[str]] = None,
        potential_files: Optional[List[str]] = None,
        model_driver: str = None,
        model_defn: Optional[str] = None,
        model_init: Optional[str] = None,
        work_dir: str = '.',
        previous_item_name: str = None,
    ) -> str:
        """
        Save a potential into KIMKit for storage

        Add a KIM Portable Model (conformant to KIM API 2.3+) to KIMkit
        with placeholder metadata, intended for temporary models

        AT LEAST ONE of either kim_id or model_name_prefix is REQUIRED
        to save a potential to KIMkit.

        This is because When saving a new model it must be assigned a
        kimcode, a structured unique id code of the form

        Human_Readable_Prefix__MO_000000999999_000

        Each kimcode begins With a human-readable prefix (containing
        letters, numbers, and underscores, and starting with a letter).
        Then, there's a 2-letter code corresponding to
        the type of model; MO for portable-models that implement
        their executable code in a standalone model-driver, and SM for
        simulator-models that wrap commands to a KIM-compatible simulator
        software like LAMMPS. Then, there's a unique 12 digit ID number
        that identifies the item, and finally a 3 digit version number.

        You can simply provide the human-readable prefix
        as "model_name_prefix" and this method will generate a new
        kimcode and assign it as this potential's kim_id, beginning
        with version 000.

        Otherwise, you can manually generate a kimcode yourself by
        passing the same human-readable prefix to
        kimkit.kimcodes.generate_new_kim_id(prefix)
        which will return a new unique kimcode. Then you can simply
        assign that as the item's kim_id.


        :param kim_id: Valid KIM Model ID, Alchemy_W__MO_000000999999_000
        :type kim_id: str
        :param species: List of supported species
        :type species: list(str)
        :param model_name_prefix: Human readable prefix to a KIM Model ID,
            must be provided if kim_id is not
        :type model_name_prefix: str
        :param param_files: List of paths to parameter files. If there is
            more than one parameter file, the order matters.
            For example, for SNAP, the `snapcoeff` file comes
            first, then `snapparam`. See the README of the
            corresponding KIM Model Driver on openkim.org for more info.
        :type param_files: list(str)
        :param training_files: files associated with the training of the
            potential.
        :type training_files: list(str)
        :param potential_files: list of all files to be included in the
            potenttial. A superset of param_files, training_files,
            and any other auxillary files to be included. If param_files
            and training_files are not included they will be
            added automatically.
        :type potential_files: list(str)
        :param model_driver: KIM ID of the corresponding KIM Model Driver.
            Must be in KIMkit
        :type model_driver: str
        :param model_defn: for simulator-models, commands needed to
            initialize the potential in the simulator (typically LAMMPS)
        :type model_defn: str
        :param model_init: for simulator-models, commands needed to
            initialize the model in the simulator (typically LAMMPS)
        :type model_init: str
        :param work_dir: where to make temporary files
        :type work_dir: str
        :param previous_item_name: any name the item was referred to
            before this, that may be lingering in makefiles. Used
            by KIMkit to do a regex search to attempt to update
            makefiles to refer to the item's kim_id
        :type previous_item_name: str
        :returns: id of the newly saved potential
        :rtype: str
        """

        if kim_id is None:
            try:
                kim_id = self.kim_id
            except AttributeError:
                pass
            if model_name_prefix is None:
                raise TypeError("One of kim_id or model_name_prefix"
                                "is required to initialize a potential")
            self.generate_new_kim_id(model_name_prefix)
        # don't overwrite the working model, if any
        if self.model is None:
            self.model = self.build_potential()
        param_files = self._init_param_files(dest_path=self.parameter_path)
        sorted_param_files = self._sort_param_files(param_files)

        try:
            self.potential_files += potential_files
        except AttributeError:
            self.potential_files = []
            self.potential_files += potential_files
        for file in param_files:
            if file not in sorted_param_files:
                self.potential_files.append(file)
        self.param_files = sorted_param_files

        if not model_driver:
            try:
                model_driver = self.model_driver
            except AttributeError:
                # default to openkim snap model driver if none supplied
                model_driver = "SNAP__MD_536750310735_000"
        if not species:
            try:
                species = self.species
            except AttributeError:
                raise AttributeError("""species must be specified in
                input if not an attribute of this potential instance.""")

        super(FitSnapPotential, self)._save_potential_to_kimkit(
            kim_id=kim_id,
            model_name_prefix=model_name_prefix,
            model_defn=model_defn,
            model_init=model_init,
            param_files=self.param_files,
            training_files=training_files,
            potential_files=self.potential_files,
            model_driver=model_driver,
            species=species,
            work_dir=os.path.split(self.parameter_path)[0],
            previous_item_name=previous_item_name,
        )

        return self.kim_id

    def _sort_param_files(
        self,
        param_files: str,
    ) -> list[str]:
        """
        Helper function to sort param files for FitSnap

        Sorts parameter files into *.snapcoeff, then
        *.snapparam, then all other auxillary files.

        :param param_files: paths to parameter files for fitsnap potential
        :type param_files: str
        :rtype: list of file path strings
        """
        param_files_sorted = []

        param_path = os.path.split(self.parameter_path)[0]

        snapcoeff_glob = os.path.join(param_path, "*.snapcoeff")
        snapcoeff_file = glob.glob(snapcoeff_glob)

        if len(snapcoeff_file) == 1:
            param_files_sorted.append(snapcoeff_file[0])
            param_files.remove(snapcoeff_file[0])
        else:
            raise RuntimeError("""
            .snapcoeff file required to define snap potential""")

        snapparam_glob = os.path.join(param_path, "*.snapparam")
        snapparam_file = glob.glob(snapparam_glob)

        if len(snapparam_file) == 1:
            param_files_sorted.append(snapparam_file[0])
            param_files.remove(snapparam_file[0])
        else:
            raise RuntimeError("""
            .snapparam file required to define snap potential""")

        snapmod_glob = os.path.join(param_path, "*.mod")
        snapmod_file = glob.glob(snapmod_glob)

        # check for a .mod file and if it exists
        # check if it uses zbl
        # in which case, create a .hybridparam file from it
        if len(snapmod_file) >= 1:
            for i in range(len(snapmod_file)):
                added_hybridparam = self._add_hybridparam_file_if_required(
                    snapmod_file[i])
                if added_hybridparam is not None:
                    param_files_sorted = param_files_sorted + added_hybridparam
                    break

        for file in param_files_sorted:
            if ".mod" in file:
                param_files_sorted.remove(file)
            if ".md" in file:
                param_files_sorted.remove(file)

        return param_files_sorted

    def _add_hybridparam_file_if_required(
        self,
        fitsnap_mod_file: str,
    ) -> str:
        """Parse the .mod file associated with this potential,
        and use it to create a .hybridparam file if required.

        :param fitsnap_mod_file: path to the *.mod file
            that fitsnap creates
        :type fitsnap_mod_file: str
        :returns: path to created *.hybridparam file
        :rtype: str
        """
        with open(fitsnap_mod_file, "r") as f:

            data = f.read()

        if "zbl" in data:

            (lower_cutoff, upper_cutoff, atomic_number_pairs,
             atomic_numbers) = self._get_zbl_cutoffs(fitsnap_mod_file)

            n = len(atomic_numbers)

            param_path = os.path.split(self.parameter_path)[0]
            hybridparam_file = os.path.join(param_path,
                                            "fitsnap_potential.hybridparam")
            zbl_pair_file = os.path.join(param_path, "zbl.pair")

            with open(hybridparam_file, "w") as f2:
                f2.write("# Number of elements for the hybrid style\n")
                f2.write(f"{n}\n")
                f2.write("\n")
                f2.write("# Element names\n")
                species_string = ""
                for number in atomic_numbers:
                    element = periodictable.elements[number].symbol
                    species_string += element
                    species_string += " "
                f2.write(species_string + "\n")
                f2.write("\n")
                f2.write("# zbl inner outer\n")
                f2.write("zbl " + str(lower_cutoff) + " " + str(upper_cutoff))
                f2.write("\n")
                f2.write("\n")
                f2.write("# Element_1 Element_2 zbl Z_1 Z_2\n")
                for pair in atomic_number_pairs:
                    atomic_number2 = str(pair[0])
                    atomic_number1 = str(pair[1])
                    element1 = periodictable.elements[pair[0]].symbol
                    element2 = periodictable.elements[pair[1]].symbol
                    pair_line = ""
                    pair_line += element1
                    pair_line += " "
                    pair_line += element2
                    pair_line += " zbl "
                    pair_line += atomic_number1
                    pair_line += " "
                    pair_line += atomic_number2
                    f2.write(pair_line)
                    f2.write("\n")

                f2.write("\n")
            f2.close()
            if self.kim_item_type == "portable-model":
                return [hybridparam_file]

            if self.kim_item_type == "simulator-model":
                with open(zbl_pair_file, "w") as f:
                    for pair in atomic_number_pairs:
                        atomic_number2 = str(pair[0])
                        atomic_number1 = str(pair[1])
                        element1 = periodictable.elements[pair[0]].symbol
                        element2 = periodictable.elements[pair[1]].symbol
                        f.write(element1 + " " + element2 + " zbl "
                                + atomic_number1 + " " + atomic_number2)
                return [hybridparam_file, zbl_pair_file]

            else:
                raise TypeError("kim_item_type must be either"
                                "'portable-model' or 'simulator-model'")

        else:
            return None

    def _get_zbl_cutoffs(self, fitsnap_mod_file):
        """
        Helper function to read required zbl parameters from the
        fitsnap param_files when using zbl.

        :param fitsnap_mod_file: path to the .mod parameter file
            that specifies the zbl cutoffs
        :type fitsnap_mod_file: str
        """

        with open(fitsnap_mod_file, "r") as f:
            data = f.read()

        data = data.split("\n")
        limit_line = "pair_style hybrid/overlay zbl"
        pair_line = "pair_coeff * * zbl"
        lower_cutoff = None
        upper_cutoff = None

        atomic_numbers = set()

        for line in data:
            if limit_line in line:
                words = line.split(" ")
                for word in words:
                    try:
                        num = float(word)
                        if not lower_cutoff:
                            lower_cutoff = num
                        else:
                            upper_cutoff = num
                    except ValueError:
                        pass
            elif pair_line in line:
                words = line.split(" ")
                for word in words:
                    try:
                        num = int(word)
                        atomic_numbers.add(num)
                    except ValueError:
                        pass

        atomic_number_pairs = []

        for result in itertools.combinations_with_replacement(
                atomic_numbers, 2):
            atomic_number_pairs.append(result)

        return lower_cutoff, upper_cutoff, atomic_number_pairs, atomic_numbers

    def install_potential_in_kim_api(
        self,
        potential_name='kim_potential',
        model_defn=None,
        model_init=None,
        install_locality='user',
        save_path='.',
        import_into_kimkit=True,
    ) -> None:
        """
        set up potential so it can be used externally

        For a KIM model, this entails installing the potential into the KIM API

        :param potential_name: name of the potential.,
            |default| 'kim_potential'
        :type potential_name: str
        :param model_defn: for simulator-models, commands needed to
            initialize the potential in the simulator (typically LAMMPS)
        :type model_defn: str
        :param install_locality: kim-api-collections-management collection
            to install into. Options include "user", "system", "CWD",
            and "environment" |default| "user"
        :type install_locality: str
        :param save_path: location where the files associated with the
            potential are on disk. The files should already be written
            to save_path. |default| "."
        """
        param_files = self._init_param_files(dest_path=self.parameter_path)
        sorted_param_files = self._sort_param_files(param_files)
        potential_files = []
        for file in param_files:
            if file not in sorted_param_files:
                potential_files.append(file)
        self.potential_files = potential_files
        self.param_files = sorted_param_files

        if self.kim_item_type == "portable-model":
            try:
                self.model_driver
            except AttributeError:
                # default to openkim snap model driver if none supplied
                self.model_driver = "SNAP__MD_536750310735_000"
        return super().install_potential_in_kim_api(
            potential_name=potential_name,
            model_defn=model_defn,
            model_init=model_init,
            install_locality=install_locality,
            save_path=save_path,
            import_into_kimkit=import_into_kimkit)

    def _init_param_files(self, dest_path) -> None:
        """
        Write out the potential's current parameters to dest_path,
        record the paths to all of the parameter files and set them
        as members of self.param_files.

        :param dest_path: where to save parameter files
        :type dest_path: str
        """

        # why is this an input argument if we overwrite it?
        dest_path = self.parameter_path

        try:
            param_path = os.path.split(dest_path)[0]
            self._write_potential_to_file(path=param_path)
        except TypeError:
            return []

        try:
            param_files = [
                os.path.join(param_path, file)
                for file in os.listdir(param_path)
            ]

            # only these file extensions
            # should be in the fitsnap param files
            good_extensions = ("snapparam", "snapcoeff", "hybridparam", "pair")

            filtered_param_files = []

            for file in param_files:
                for extension in good_extensions:
                    if extension in file:
                        filtered_param_files.append(file)

            self.param_files = filtered_param_files
        except Exception:
            raise

        return filtered_param_files

    def _write_smspec(self,
                      potential_type='snap',
                      model_defn=None,
                      model_init=None,
                      work_dir="."):
        """
        Helper method to write the auxillary file smspec.edn,
        which is used by the KIM_API to build simulator-models.

        :param potential_type: what type of potential object this is,
            (e.g. fitsnap, dnn, etc.)
        :type potential_type: str
        :param model_defn: for simulator-models, commands needed to
            define the potential in the simulator (typically LAMMPS)
        :type model_defn: str
        :param model_init: optional for simulator-models, commands needed to
            initialize the potential in the simulator (typically LAMMPS)
        :type model_init: str
        :param work_dir: where to save the file
        :type work_dir: str
        """
        if model_defn is None:

            # if a model_defn is provided, let it override default behavior
            param_path = os.path.split(self.parameter_path)[0]
            all_files = os.listdir(param_path)
            for file in all_files:
                if ".mod" in file:
                    snapmod_file = os.path.join(param_path, file)

            model_defn = None
            for file in self.param_files:
                if "zbl.pair" in file:
                    lower_cutoff, upper_cutoff, __, __ = self._get_zbl_cutoffs(
                        snapmod_file)
                    model_defn = [
                        ("pair_style hybrid/overlay "
                         f"zbl {lower_cutoff} {upper_cutoff} snap"),
                        ("pair_coeff * *"
                         " snap @<parameter-file-1>@"
                         " @<parameter-file-2>@ @<atom-type-sym-list>@"),
                        ("KIM_SET_TYPE_PARAMETERS"
                         " pair @<parameter-file-3>@ @<atom-type-sym-list>@")
                    ]
                    break

        super()._write_smspec(potential_type, model_defn, model_init, work_dir)

    def convert_input_file_to_dict(self, path) -> dict:
        """
        Reads a fitsnap input file and creates a dictionary of the contents

        :param path: path to the input file to be read
        :type path: str
        :returns input_settings_dict: settings read from FitSNAP input file
        :rtype input_settings_dict: dict
        """
        c = configparser.ConfigParser()
        c.optionxform = str
        c.read(path)
        input_settings_dict = {s: dict(c.items(s)) for s in c.sections()}

        return input_settings_dict

    def create_fitsnap_input_file(self, settings, path) -> None:
        """
        Creates a FitSNAP input file from a settings dictionary

        See https://fitsnap.github.io/Run/Run_input.html for fitsnap input
        documentation. Dictionary should follow the same hierarchical format.
        See sister function convert_input_file_to_dict().

        :param settings: dictionary of FitSNAP settings
        :type settings: dict
        :param path: location to save the FitSNAP input file
        :type path: str
        """
        c = configparser.ConfigParser()
        c.optionxform = str
        for key, val in settings.items():
            c[key] = val

        with open(path, 'w') as f:
            c.write(f)

        return None

    def _check_hashes(self) -> bool:
        """
        Checks the hashes in the training files against the saved one.

        Checks *.mod, *.snapcoeff, and *.snapparam at the parameter_path
        to see if they have the hash value from the last training of the
        potential.
        """
        for file_suffix in ['/*.mod', '/*.snapcoeff', '/*.snapparam']:
            for filepath in glob.glob(self.parameter_path + file_suffix):
                with open(filepath, 'r') as f:
                    for line in f.readlines():
                        if "Hash:" in line.split():
                            f_hash = line.split().index("Hash:")
                            if f_hash != self.training_hash:
                                return False
        return True

    def get_params(self):
        """
        return the parameters of the potential in a human readable format

        :returns: parameters read from the .snapcoeff file
        :rtype: dict
        """
        try:
            # *.snapcoeff is sorted to be the first param_file
            with open(self.param_files[0], 'r') as f:
                lines = f.readlines()
                # TODO: check how this is formatted for different 2J max
                num_species, num_coeff_each = [
                    int(a) for a in lines[2].split()
                ]
                species = []
                radelem = []
                wj = []
                coeffs = []
                coeff_labels = []
                for i in range(num_species + 1):
                    a, b, c = lines[i * (num_coeff_each + 1) + 3].split()
                    species.append(a)
                    radelem.append(float(b))
                    wj.append(float(c))
                    data = []
                    labels = []
                    data_start = i * (num_coeff_each + 1) + 3 + 1
                    data_end = (i + 1) * (num_coeff_each + 1) + 3
                    for k in range(data_start, data_end + 1):
                        split_line = lines[k].split()
                        data.append(float(split_line[0]))
                        labels.append(split_line[-1])
                    coeffs.append(data)
                    coeff_labels.append(labels)

                snapcoeff_data = {
                    'species': species,
                    'radelems': radelem,
                    'wjs': wj,
                    'coeffs': coeffs,
                    'coeff_labels': coeff_labels
                }
        except Exception:
            raise

        return snapcoeff_data

    def get_metadata(self):
        """
        return the relevant metadata about the potential
        """
        raise NotImplementedError

    def get_hyperparameters(self):
        """
        return the relevant hyperparameters of the potential
        """
        # untrained can report back the values from settings_path,
        # or possibly directly from potential.model if built
        # trained can exist with no settings; just the param_files
        # SNAP param files sufficient but not exhaustive
        raise NotImplementedError
