from pprint import pprint

from orchestrator.potential.nequip_allegro import NequIPAllegroPotential
from orchestrator.utils.setup_input import init_and_validate_module_type

print("From existing")
model = NequIPAllegroPotential.initialize_from_files(
    "existing_model/config.yaml",
    parameters_file="existing_model/parameters.ckpt")
pprint(model.hyperparameters)

print("From scratch")
model = NequIPAllegroPotential(
    species=["C", "H", "O"],
    model_type="allegro",
    device="cuda",
    cutoff_radius=4.5,
    num_layers=1,
    l_max=1,
    num_features=32,
)

pprint(model.hyperparameters)
pprint(f"{model.potential_files=}")

# Saving
print("Saving model")
try:
    model.save_potential("new_model", makedirs=True)
except FileNotFoundError as e:
    print(e)
    print("It is expected that this would fail, since the model "
          "was made from scratch and never trained.")

# Training
print("Training model")
workflow = init_and_validate_module_type(
    "workflow",
    {
        "workflow_type": "LOCAL",
        "workflow_args": {
            "root_directory": ".",
            "wait_freq": 20,
        },
    },
    single_input_dict=True,
)

storage = init_and_validate_module_type(
    "storage",
    {
        "storage_type": "COLABFIT",
        "storage_args": {
            "credential_file": "/PATH/TO/your_credentials.json"
        },
    },
    single_input_dict=True,
)

model.train(
    dataset_list=["DS_ID_FOR_CHIMES"],
    storage=storage,
    workflow=workflow,
    energy_weight=1.0,
    force_weight=1.0,
    stress_weight=0.0,
    train_frac=0.8,
    test_frac=0.1,
    val_frac=0.1,
    max_epochs=10,
    per_atom_weights=None,
    log_errors=True,
)

model.save_potential("from_train", makedirs=True)

# Re-build it, to avoid re-using previous results
model = NequIPAllegroPotential(
    species=["C", "H", "O"],
    model_type="allegro",
    device="cuda",
    cutoff_radius=4.5,
    num_layers=1,
    l_max=1,
    num_features=32,
)

calc_id = model.submit_train(
    dataset_list=["DS_ID_FOR_CHIMES"],
    storage=storage,
    workflow=workflow,
    job_details={},
    energy_weight=1.0,
    force_weight=1.0,
    stress_weight=0.0,
    train_frac=0.8,
    test_frac=0.1,
    val_frac=0.1,
    max_epochs=10,
    per_atom_weights=None,
)

model.load_from_submitted_training(calc_id, workflow)
model.save_potential("from_submit_train", makedirs=True)

print("Done!")
