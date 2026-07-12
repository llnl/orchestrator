import os
from typing import List, Optional
from kliff.legacy import nn
from kliff.legacy.descriptors import SymmetryFunction
from kliff.models import NeuralNetwork
from .potential_base import Potential
from ..utils.restart import restarter


class KliffBPPotential(Potential):
    """
    Build a Beller-Parrinello DNN using KLIFF

    any or all parameters defining the DNN can be
    provided as key value pairs in the trainer_args dict.
    Parameters that are not provided will be
    initialized to default values.
    Instantiating a KliffBPPotential object with an empty dict
    will therefore generate a "default" configuration.
    This DNN is built using KLIFF with a pytorch backend.

    :param potential_args: dict with the input parameters and
        their values as k,v pairs. Parameters include:
    :param cutoff_type: kliff - type of cutoff in descriptor
    :param cutoff: kliff - distance cutoff for descriptor
    :param hyperparams: kliff - hyperparameters of BPDNN
    :param norm: kliff - apply or not data normalization
    :param neurons: kliff - number of neurons in each layer of the BPDNN
    """

    def __init__(
        self,
        species: list[str],
        model_driver: str,
        cutoff_type,
        cutoff,
        hyperparams,
        norm,
        neurons,
        kim_api: str = 'kim-api-collections-management',
        kim_item_type: str = "simulator-model",
        kim_id: str = None,
        model_name_prefix="DNN_Potential_Orchestrator_Generated",
        param_files: list = None,
        training_files: list = None,
        potential_files: list = None,
        checkpoint_file: str = './orchestrator_checkpoint.json',
        checkpoint_name: str = 'potential',
        **kwargs,
    ):
        """
        initialization of the BPDNN with trainer_args dict

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
        :param cutoff_type: kliff - type of cutoff in descriptor
        :param cutoff: kliff - distance cutoff for descriptor
        :param hyperparams: kliff - hyperparameters of BPNN
        :param norm: kliff - apply or not data normalization
        :param neurons: kliff - number of neurons in each layer of the DNN
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

        # DNN-specific required arguments
        self.cutoff_type = cutoff_type
        self.cutoff = cutoff
        self.hyperparams = hyperparams
        self.norm = norm
        self.neurons = neurons

        self.checkpoint_potential_name = './checkpointed_potential.pkl'
        self.checkpoint_name = checkpoint_name

        if kim_id:
            self.kim_id = kim_id

        # set trainer_args that KLIFF needs to train
        self.trainer_args = {
            "cutoff_type": cutoff_type,
            "cutoff": cutoff,
            "hyperparams": hyperparams,
            "norm": norm,
            "neurons": neurons,
            "kim_api": kim_api
        }

        self.species = species
        self.model_driver = model_driver

        self.model_type = 'dnn'
        self.trainer = 'kliff'
        self.is_torch = True
        self.model = None

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
        self._write_potential_to_file(self.checkpoint_potential_name)
        save_dict = {
            self.checkpoint_name: {
                'wrote_potential': True,
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
        # see if any internal variables were checkpointed
        restart_dict = restarter.read_checkpoint_file(
            self.checkpoint_file,
            self.checkpoint_name,
        )
        wrote_potential = restart_dict.get('wrote_potential', False)
        if wrote_potential:
            self.load_potential(self.checkpoint_potential_name)

    def build_potential(self) -> NeuralNetwork:
        """
        Build DNN potential using KLIFF

        The parameters for the DNN potential were passed to the object in
        __init__ as the trainer_args dict. Unspecified parameters
        were set to default values (Si).
        In addition to returning the model, this method also sets
        it as the objects self.model attribute.

        :returns: model parameterized by self.trainer_args
        :rtype: NeuralNetwork
        """

        self.logger.info('Building potential using given parameters')

        descriptor = SymmetryFunction(cut_name=self.cutoff_type,
                                      cut_dists=self.cutoff,
                                      hyperparams=self.hyperparams,
                                      normalize=self.norm)
        model = NeuralNetwork(descriptor)

        layers = (
            nn.Linear(descriptor.get_size(), self.neurons[0]),
            nn.Tanh(),
        )

        for layer in range(len(self.neurons) - 1):
            layers += (
                nn.Linear(self.neurons[layer], self.neurons[layer + 1]),
                nn.Tanh(),
            )

        layers += (nn.Linear(self.neurons[-1], 1), )

        model.add_layers(*layers)
        model.is_torch = self.is_torch
        model.trainer = self.trainer

        self.logger.info('Finished constructing BPDNN potential')
        self.model = model

        return model

    def _write_potential_to_file(self, path: str):
        """
        save the current potential state to a file which can be loaded

        Use the ModelTorch save and load methods. The file is a serialized
        pickle file. KIM models are only written for install.

        :param path: path including filename where the potential is to be
            written
        :type path: str
        """
        if os.path.isdir(path):
            path = os.path.join(path, "DNN_Potential")
        if path[-4:] != '.pkl':
            self.logger.info(f'Appending .pkl to {path}')
            path = path + '.pkl'
        self.logger.info(f'Writing potential pkl file to {path}')
        self.model.save(path)

    def load_potential(self, path: str):
        """
        parameterize the potential based on an existing potential at path

        This method loads a potential saved by the
        :meth:`write_potential_to_file` method. The path should match what was
        used to save the potential. Filenames are automatically appended with
        .pkl if they do not include this extension. This method sets the
        internal model parameters according to the saved state.

        :param path: path string including filename where potential resides
        :type path: str
        """
        if path[-4:] != '.pkl':
            self.logger.info(f'Appending .pkl to {path}')
            path = path + '.pkl'
        if self.model is None:
            self.build_potential()
        self.logger.info(f'Loading potential pkl file from {path}')
        self.model.load(path)

    def get_potential_files(self,
                            destination_path: str,
                            kim_id: str,
                            include_dependencies: bool = False) -> str:
        """
        Load a DNN model from a kimkit repository using the KIM ID
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
        tarfile_name = super(KliffBPPotential, self).get_potential_files(
            destination_path=destination_path,
            kim_id=kim_id,
            include_dependencies=include_dependencies)

        return tarfile_name

    def _save_potential_to_kimkit(self,
                                  kim_id: str = None,
                                  species: List[str] = None,
                                  model_name_prefix: str = None,
                                  param_files: List[str] = None,
                                  training_files: Optional[List[str]] = None,
                                  potential_files: Optional[List[str]] = None,
                                  model_driver: str = "no-driver",
                                  model_defn: Optional[str] = None,
                                  model_init: Optional[str] = None,
                                  work_dir: str = '.',
                                  previous_item_name: str = None) -> str:
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
        if not kim_id:
            try:
                kim_id = self.kim_id
            except AttributeError:
                kim_id = self.generate_new_kim_id(model_name_prefix)

        work_dir = os.path.join(work_dir, kim_id)

        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        # don't overwrite the working model, if any
        if self.model is None:
            self.model = self.build_potential()

        try:
            param_files = self.param_files
            if not param_files:
                self._write_potential_to_file(
                    os.path.join(work_dir, "DNN_Potential"))
                param_files = []
                param_files.append(os.path.join(work_dir, "DNN_Potential.pkl"))
        except AttributeError:
            self._write_potential_to_file(
                os.path.join(work_dir, "DNN_Potential"))
            param_files = []
            param_files.append(os.path.join(work_dir, "DNN_Potential.pkl"))

        if not species:
            try:
                species = self.species
            except AttributeError:
                raise AttributeError("""species must be specified in
                input if not an attribute of this potential instance.""")

        super(KliffBPPotential, self)._save_potential_to_kimkit(
            kim_id=self.kim_id,
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
        if not os.path.isdir(f'{save_path}/{potential_name}'):
            self.model.write_kim_model(f'{save_path}/{potential_name}')

        return super().install_potential_in_kim_api(
            potential_name=potential_name,
            model_defn=model_defn,
            model_init=model_init,
            install_locality=install_locality,
            save_path=save_path,
            import_into_kimkit=import_into_kimkit)

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

    def evaluate(self, data):
        """
        evaluate the potential for given data

        :param data: configuration data (atomic positions) to obtain
            forces, energy, and/or stresses from the potential
        """
        raise NotImplementedError
