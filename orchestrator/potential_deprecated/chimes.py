import os
from typing import List, Optional
from .potential_base import Potential
from ..utils.restart import restarter
from ..utils.exceptions import InstallPotentialError


class ChIMES:

    def __init__(
        self,
        polynomial_orders: str,
        cutoff_distances: str,
    ):
        self.polynomial_orders = polynomial_orders
        self.cutoff_distances = cutoff_distances


class ChIMESPotential(Potential):
    """
    Build a potential using ChIMES

    All parameters defining the ChIMES potential are defined in the
    settings file described in the trainer_args dict.

    :param trainer_args: dict with the input parameters and their values as k,v
        pairs. Parameters include:
    :param settings_path:
    """

    def __init__(
        self,
        species: list[str],
        model_driver: str,
        kim_api: str,
        polynomial_orders: list[int],
        cutoff_distances: list[float],
        kim_item_type: str = 'simulator-model',
        parameter_path: str = None,
        kim_id: str = None,
        model_name_prefix: str = "ChIMES_Potential_Orchestrator_Generated",
        param_files: Optional[list] = None,
        training_files: Optional[list] = None,
        potential_files: Optional[list] = None,
        checkpoint_file: Optional[str] = './orchestrator_checkpoint.json',
        checkpoint_name: Optional[str] = 'potential',
        **kwargs,
    ):
        """
        initialization of the ChIMES potential with trainer_args dict

        :param species: list of strings containing element symbols
        :type species: list[str]
        :param model_driver: driver needed to run the potential
        :type model_driver: str
        :param kim_api: path to the kim-api-collections-manager
            executable.
        :type kim_api: str
        :param polynomial_orders: list of polynomial orders to define the
            ChIMES potential
        :type polynomial_orders: list[int]
        :param cutoff_distances: list of cutoff distances to define the ChIMES
            potential
        :type cutoff_distances: list[float]
        :param kim_item_type: what type of kim object to create an ID for.
            For potentials, this should be either "portable-model" or
            "simulator-model", depending on whether the model uses a driver
            to implement its calculations, or runs commands in a simulator
            program (e.g. lammps, ASE) respectively.
        :type kim_item_type: str
        :param parameter_path: path where the potential's param_files
            will be written
        :type parameter_path: str
        :param kim_id: kimcode to represent the item
        :type kim_id: str
        :param model_name_prefix: human readable prefix to make a kim ID
        :type model_name_prefix: str
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

        if kim_id:
            self.kim_id = kim_id

        self.model = None

        self.model_type = "ChIMES"

        if parameter_path is not None:
            root_exists = os.path.isdir(os.path.split(parameter_path)[0])
            if not root_exists:
                raise ValueError('parameter_path does not exist!')
        self.parameter_path = parameter_path

        self.kim_api_compatible = True

        if not model_driver:
            # default to no-driver if none supplied
            model_driver = "no-driver"

        if kim_item_type != 'simulator-model':
            raise ValueError('ChIMES Potentials are only supported as '
                             'simulator-models in Orchestrator')

        self.checkpoint_name = checkpoint_name

        self.trainer_args = {
            "parameter_path": parameter_path,
            "polynomial_orders": polynomial_orders,
            "cutoff_distances": cutoff_distances,
            "kim_api": kim_api
        }

        self.name = "chimes_potential"

        super().__init__(
            kim_id,
            species,
            model_driver,
            kim_item_type=kim_item_type,
            model_name_prefix=model_name_prefix,
            param_files=param_files,
            training_files=training_files,
            potential_files=potential_files,
            kim_api=kim_api,
            checkpoint_file=checkpoint_file,
            checkpoint_name=self.checkpoint_name,
        )
        self.logger.info('Finished instantiating ChIMES potential')

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
        self.build_potential()

    def build_potential(self) -> ChIMES:
        """
        Build a chimes potential using ChIMES

        The settings file that includes parameters for the ChIMES potential
        were passed to the object in __init__ as the trainer_args dict.
        In addition to returning the model,
        this method also sets it as the objects self.model attribute.

        :returns: chimes model parameterized by parameters in the settings
            file
        :rtype: ChIMES instance
        """

        self.logger.info('Building potential using given parameters')

        polynomial_orders = self.trainer_args['polynomial_orders']
        cutoff_distances = self.trainer_args['cutoff_distances']
        chimes = ChIMES(polynomial_orders, cutoff_distances)

        self.logger.info('Finished constructing ChIMES potential')
        self.model = chimes

        return self.model

    def load_potential(self, path: str):
        """
        parameterize the potential based on an existing potential at path

        :param path: path string including filename where potential resides
        """
        raise NotImplementedError

    def _write_potential_to_file(self, path: str):
        """
        save the current potential path to a file in a specified path

        :param path: path including filename where the potential is to be
            written
        :type path: str
        """

        if self.parameter_path is not None:
            if (os.path.abspath(self.parameter_path) != os.path.abspath(path)):
                os.system(f'ln -s `realpath {self.parameter_path}`* {path}')
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

        if param_files is None:
            param_files = self._init_param_files(dest_path=self.parameter_path)
        self.potential_files = potential_files

        if not model_driver:
            try:
                model_driver = self.model_driver
            except AttributeError:
                raise AttributeError("""model_driver must be specified in
                input if not an attribute of this potential instance.""")

        if not species:
            try:
                species = self.species
            except AttributeError:
                raise AttributeError("""species must be specified in
                input if not an attribute of this potential instance.""")

        super(ChIMESPotential, self)._save_potential_to_kimkit(
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
            previous_item_name=previous_item_name)

        return self.kim_id

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
            and "environment-variable" |default| "user"
        :type install_locality: str
        :param save_path: location where the files associated with the
            potential are on disk. The files should already be written
            to save_path. |default| "."
        """
        param_files = self._init_param_files(dest_path=self.parameter_path)
        sorted_param_files = self._sort_param_files(param_files)
        self.param_files = sorted_param_files

        model_defn = [
            ("pair_style chimesFF"),
            ("variable kim_atom_type_sym_list string "
             "\"@<atom-type-sym-list>@\""),
            ("include @<parameter-file-1>@"),
            ("variable kim_atom_type_sym_list delete"),
            ("pair_coeff * * @<parameter-file-2>@"),
        ]

        if self.kim_item_type == "simulator-model":
            try:
                self.model_driver
            except AttributeError:
                # default to openkim no-driver if none supplied
                self.model_driver = "no-driver"
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
        except TypeError:
            return []

        try:
            param_files = [
                os.path.join(param_path, file)
                for file in os.listdir(param_path)
            ]

            self.param_files = param_files
        except Exception:
            raise

        return param_files

    def _sort_param_files(self, param_files):

        sorted_param_files = []

        for file in param_files:
            if 'masses.lammps' in file:
                sorted_param_files.append(os.path.abspath(file))
                break

        for file in param_files:
            if 'chimes_potential' in file:
                sorted_param_files.append(os.path.abspath(file))
                break

        return sorted_param_files

    def get_params(self):
        """
        return the parameters of the potential in a human readable format
        """
        raise NotImplementedError

    def get_metadata(self):
        """
        return the relevant metadata about the potential
        """
        raise NotImplementedError

    def get_hyperparameters(self):
        """
        return the relevant hyperparameters of the potential
        """
        raise NotImplementedError
