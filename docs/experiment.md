# Experiment

`idrak.experiment.Experiment` is a utility class designed for tracking and managing your deep learning experimental runs. It provides a structured way to log hyperparameters, metrics, and artifacts, making your research reproducible and organized.

## Key Features

*   **Hyperparameter Logging:** Record all configuration parameters for your experiment.
*   **Metric Tracking:** Log performance metrics (e.g., loss, accuracy) over time or at specific steps.
*   **Artifact Management:** Keep track of important files generated during the experiment (e.g., saved models, plots).
*   **Serialization:** Save and load experiment data to/from YAML files.

## Initialization

An `Experiment` is initialized with a unique ID, a name, and an optional description.

```python
from idrak.experiment import Experiment
import uuid

# Generate a unique ID for the experiment
exp_id = str(uuid.uuid4())

# Initialize the experiment
exp = Experiment(id=exp_id, name="MyFirstTrainingRun", description="Training a simple linear model on MNIST.")
print(f"Experiment initialized with ID: {exp.id}")
```

## Logging Hyperparameters

Use `log_hyperparameter(key: str, value: Any)` to record any configuration setting relevant to your experiment.

```python
# ... (assuming exp is initialized as above) ...

hyperparams = {
    "learning_rate": 0.01,
    "batch_size": 32,
    "epochs": 10,
    "optimizer": "Adam",
    "model_architecture": "Linear(10, 1)" # Example architecture string
}

for key, value in hyperparams.items():
    exp.log_hyperparameter(key, value)

# Example of logging a dynamically generated model architecture string
model_architecture_str = "Linear -> Sigmoid"
exp.log_hyperparameter("model_architecture", model_architecture_str)

print("Logged hyperparameters:", exp.hyperparameters)
```

## Logging Metrics

Use `log_metric(key: str, value: Any, step: Optional[int] = None)` to record performance metrics. The `step` parameter is useful for tracking metrics over training iterations or epochs.

```python
# ... (assuming exp is initialized) ...

# Simulate a training loop
for epoch in range(hyperparams["epochs"]):
    # ... (perform training for the epoch) ...
    train_loss = 0.5 - epoch * 0.01 # Example loss
    val_accuracy = 0.7 + epoch * 0.02 # Example accuracy

    exp.log_metric("train_loss", train_loss, step=epoch)
    exp.log_metric("val_accuracy", val_accuracy, step=epoch)

print("Logged metrics:", exp.metrics)
```

## Logging Artifacts

Use `log_artifact(path: str, name: Optional[str] = None)` to record paths to important files generated during your experiment. This could include saved model weights, plots, or data samples.

```python
from idrak.experiment import Experiment
import uuid
from idrak.utils.model_io import save_model, load_model # Added import
from idrak.nn import Pipeline, Linear # Added for model example
from idrak.nn.activations import Sigmoid # Added for model example
import os # Added for path operations

# ... (assuming exp is initialized) ...

# Simulate a simple model for saving
model = Pipeline(Linear(10, 1), Sigmoid())

# Ensure the directory for saved models exists
os.makedirs("./saved_models", exist_ok=True)
model_path = os.path.join("./saved_models", "my_model_epoch_9.idrak")
save_model(model, model_path) # Now actively saving the model

exp.log_artifact(model_path, name="final_model_weights")

print("Logged artifacts:", exp.artifacts)
```

## Saving and Loading Experiments

Experiment data can be saved to a YAML file and loaded back later.

*   `save(directory: str = "experiments")`: Saves the experiment data to a YAML file within the specified directory.
*   `load(experiment_id: str, directory: str = "experiments") -> Experiment`: Loads an experiment from a YAML file.

```python
# ... (assuming exp has logged data and exp_id is defined) ...

# Save the experiment data
exp.save(directory="./my_experiments")

# Later, load the experiment
loaded_exp = Experiment.load(exp.id, directory="./my_experiments") # Use exp.id for loading
print(f"Loaded experiment ID: {loaded_exp.id}")
print(f"Loaded hyperparameters: {loaded_exp.hyperparameters}")
```

