import os
from typing import List, Optional
from kliff.models import KIMModel
from kliff.models.kim import KIMModelError
from .potential_base import Potential
from ..utils.restart import restarter


# KMB note - I think this class may be redundant if more "well described"
# potentials are implemented with the proper "load" functions. I.e. it is
# more desireable to load a KIM DNN as a KliffBPPotential object with well
# defined parameters. Keep this as a catch-all for now until all desired
# potential types are implemented properly
class KIMPotential(Potential):
    """
    initialization of an arbitrary KIM potential with trainer_args dict

    the KIM ID of an existing KIMModel is provided as key-value pairs in
    the trainer_args dict.

    :param potential_args: dict with the input parameters and their values as
        k,v pairs. Parameters include:
    :param kim_id: ID of an existing KIM potential
    """

    def __init__(
        self,
        kim_id: str,
        species: list[str],
        model_driver: str,
        kim_api: str = 'kim-api-collections-management',
        kim_item_type: str = "portable-model",
        model_name_prefix: str = "KIM_Potential_Orchestrator_Generated",
        param_files: list = None,
        training_files: list = None,
        potential_files: list = None,
        checkpoint_file: str = './orchestrator_checkpoint.json',
        checkpoint_name: str = 'potential',
        **kwargs,
    ):
        """
        Load an existing KIM potential with Kliff

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
        self.kim_api_compatible = True

        self.kim_api = kim_api

        if checkpoint_name:
            self.checkpoint_name = checkpoint_name

        self.kim_id = kim_id

        # this could be set and used as a switch to wrap the proper class
        self.model_type = None
        self.trainer = 'kliff'
        self.is_torch = False  # is this always true?
        self.model: Optional[KIMModel] = None
        super().__init__(kim_id,
                         species,
                         model_driver,
                         model_name_prefix=model_name_prefix,
                         param_files=param_files,
                         training_files=training_files,
                         potential_files=potential_files,
                         kim_api=kim_api,
                         kim_item_type=kim_item_type,
                         checkpoint_file=checkpoint_file,
                         checkpoint_name=checkpoint_name)

    def checkpoint_potential(self):
        """
        checkpoint the potential module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        write = self._write_potential_to_file(self.checkpoint_potential_name)
        save_dict = {
            self.checkpoint_name: {
                'wrote_potential': write,
                'kim_id': self.kim_id
            }
        }
        restarter.write_checkpoint_file(self.checkpoint_file, save_dict)

    def restart_potential(self):
        """
        restart the potential module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        # see if any internal variables were checkpointed
        restart_dict = restarter.read_checkpoint_file(
            self.checkpoint_file,
            self.checkpoint_name,
        )
        wrote_potential = restart_dict.get('wrote_potential', False)
        if wrote_potential:
            self.logger.info(
                'KIMPotential cannot currently restart from a file!')

    def build_potential(self) -> KIMModel:
        """
        "Build" a KIM potential using KLIFF

        The existing KIM potential is loaded based on the KIM ID passed on
        class instantiation in the trainer_args dict.
        Return the model and set the elf.model attribute as well.

        :returns: pre-existing KIM model
        :rtype: KIMModel
        """
        model = self.load_potential(None)
        return model

    def _write_potential_to_file(
        self,
        path: str,
    ) -> bool:
        """
        save the current potential state to a file which can be loaded

        Not all KIM models have defined ways for writing to file, so this
        method will not always work. However, these models are likely not being
        trained and can just be loaded as KIMModels using their KIM IDs.

        :param path: path including filename where the potential is to be
            written
        :type path: str
        :returns: True if succesfully written, False otherwise
        :rtype: boolean
        """
        try:
            self.model.write_kim_model(path)
            return True
        except KIMModelError:
            self.logger.info('KIM does not support writing this potential'
                             'type to file')
            return False

    # this may be the more appropriate function to use? Call from build?
    def load_potential(self, path: str = None) -> KIMModel:
        """
        Load a KIM model using the KIM ID

        :param path: path including filename where potential resides, if not
            installed to the user namespace |default| ``None``
        :type path: str
        :returns: pre-existing KIM model
        :rtype: KIMModel
        """

        if path is not None:
            self.logger.warning(('Load from path not yet implemented, will '
                                 'attempt normal load from user namespace'))

        self.logger.info(f'Loading KIM model with ID: {self.kim_id}')
        try:
            model = KIMModel(model_name=self.kim_id)
        except Exception as e:
            # kim unit tests use this potential as a base
            # the kliff trainer wants to already have it in the KIM_API
            # and then uninstalls it once training is done
            # which can cause errors in the tests subsequently
            # so if that happens, just reinstall it
            test = "SW_StillingerWeber_1985_Si__MO_405512056662_006"
            if self.kim_id == test:
                os.system(f'{self.kim_api} install --force '
                          f'user {self.kim_id}; cd - 1> /dev/null')
                model = KIMModel(model_name=self.kim_id)
            else:
                raise e
        self.logger.info(f'Finished loading {self.kim_id}')
        model.is_torch = self.is_torch
        model.trainer = self.trainer
        self.model = model
        return model

    def get_potential_files(
        self,
        destination_path: str,
        kim_id: str = None,
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
        tarfile_name = super(KIMPotential, self).get_potential_files(
            destination_path=destination_path,
            kim_id=kim_id,
            include_dependencies=include_dependencies)

        return tarfile_name

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
        their executable code internally or in a standalone model-driver,
        and SM for simulator-models that wrap commands to a KIM-compatible
        simulator software like LAMMPS. Then, there's a unique 12-digit ID
        number that identifies the item, and finally a 3-digit version number.

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

        # if a previous item name is being passed in the metadata dict
        # from the kimkit integration test
        # read it out and remove it from the dict.
        tmp_kim_id = self.kim_id

        # temporarily use the old item ID to load it
        if previous_item_name:
            self.kim_id = previous_item_name

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

        # don't overwrite the working model, if any
        if self.model is None:
            self.model = self.load_potential()

        # re-set the current kim_id
        self.kim_id = tmp_kim_id

        kim_id = self.kim_id

        work_dir = os.path.join(work_dir, self.kim_id)

        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        super(KIMPotential, self)._save_potential_to_kimkit(
            kim_id=kim_id,
            model_name_prefix=model_name_prefix,
            model_defn=model_defn,
            model_init=model_init,
            param_files=param_files,
            training_files=training_files,
            potential_files=potential_files,
            model_driver=model_driver,
            species=species,
            work_dir=work_dir,
            previous_item_name=previous_item_name)

        return self.kim_id

    def get_params(self, **kwargs) -> dict:
        """
        Get potential parameter values.

        The format of the keyword argument dictionary is::

           kwargs = {
               param1: [idx1, idx2, idx3],
               param2: [idx1, idx2, idx3],
           }

        :returns: A dictionary of current potential parameter values
        :rtype: dict
        """
        try:
            self.model_calculator
        except AttributeError:
            self._init_model_calculator()

        return self.model_calculator.get_parameters(**kwargs)

    def get_metadata(self):
        """
        return the relevant metadata about the potential
        """
        print('this method is not yet implemented')

    def get_hyperparameters(self):
        """
        return the relevant hyperparameters of the potential
        """
        print('this method is not yet implemented')

    def set_params(self, **kwargs) -> dict:
        """
        Set potential parameter values.

        The format of the keyword argument dictionary is::

           kwargs = {
               param1: [[idx1, idx2, idx3], [val1, val2, val3]],
               param2: [[idx1, idx2, idx3], [val1, val2, val3]],
           }

        :param params: A dictionary of parameter values
        :type params: dict
        """
        try:
            self.model_calculator
        except AttributeError:
            self._init_model_calculator()

        self.model_calculator.set_parameters(**kwargs)
