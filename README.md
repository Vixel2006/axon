<h1 align="center">Idrak</h1>

<p align="center">
  This is not a deep learning framework. It's an argument.
</p>

<p align="center">
  <a href="#"><img alt="Build Status" src="https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge"></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge"></a>
  <a href="https://github.com/yushi2006/nawah/issues"><img alt="Issues" src="https://img.shields.io/github/issues/yushi2006/nawah?style=for-the-badge&color=orange"></a>
  <a href="#"><img alt="Stars" src="https://img.shields.io/github/stars/yushi2006/nawah?style=for-the-badge"></a>
</p>

---

## 🚀 Philosophy: The Unapologetic Truth of Data Transformations

Idrak challenges the status quo of deep learning frameworks. We believe in a world where experimentation is effortless, extending functionality is trivial, and maintenance is a breeze. Our core philosophy revolves around:

-   **Neural Networks as Pure Data Pipelines:** Forget monolithic blocks. A neural network is nothing more than a relentless, sequential pipeline of data transformations. Idrak embraces this brutal simplicity, allowing you to compose and manipulate these transformations with unprecedented clarity and control.
-   **Unrivaled Hackability:** Dive deep, inspect, and mutate anything at runtime. Idrak treats you like a hacker, not just a user. We expose the guts because you deserve to see them.
-   **Fast & Easy Experimentation:** Rapidly iterate on ideas without boilerplate. Focus on your unique logic, not framework intricacies. Your time is too valuable for abstraction layers that fight you.
-   **Effortless Extensibility & Maintainability:** The modular design ensures that adding new features or maintaining existing ones is straightforward and intuitive. No more archaeological digs through opaque codebases.
-   **AI Accelerator Startup Advantage:** For those building AI accelerators, Idrak significantly reduces the backend operational overhead. Achieve PyTorch-like results with drastically less effort in writing custom ops. Stop writing thousands of lines of backend code when you could be innovating.
-   **Laziness is Superior:** Embrace lazy evaluation for computations. This paradigm allows for powerful optimizations, efficient resource utilization, and a more intuitive way to define complex computational graphs. Why compute what you don't need, when you don't need it?

Idrak achieves this through a clear, functional approach to deep learning components:
-   **The Model (`Pipeline`):** Compose your model's layers into a flexible `Pipeline` object, managing learnable parameters and enabling dynamic architecture changes. It's your data's journey, laid bare.
-   **The Experiment (`Experiment`):** A dedicated tool for tracking hyperparameters, logging metrics, and managing artifacts, providing a clear overview of your experimental runs. No more guessing what worked and why.

This design promotes clean, composable systems over monolithic structures, giving you full control.

<br>

## The Idrak Way in 4 Steps

See how Idrak's design leads to a cleaner, faster, and more hackable workflow.

### Step 1: Define Hyperparameters & Initialize Experiment

```python
import idrak
from idrak.experiment import Experiment
import uuid

# Define your experiment's hyperparameters
hyperparams = {
    "in_channels": 3,
    "base_channels": 64,
    "block_type": 'ResBlock',
    "learning_rate": 1e-3,
    "batch_size": 128,
    "epochs": 10,
    "optimizer": 'Adam'
}

# Initialize an experiment to track your run
exp = Experiment(id=str(uuid.uuid4()), name="MyFirstIdrakExperiment")
for key, value in hyperparams.items():
    exp.log_hyperparameter(key, value)
```

### Step 2: Build the Model (`Pipeline`)

```python
import idrak.metrics as metrics
from idrak.nn import Pipeline, Conv2d, Linear

# Layers are composed using the >> operator or passed to the Pipeline constructor
model = Pipeline(
    Conv2d(hyperparams["in_channels"], hyperparams["base_channels"], kernel_size=7)) 
    >> Linear(hyperparams["base_channels"], 10)
)

# You can reset parameters at any time to start from scratch
# model.reset_parameters()
```

### Step 3: Define YOUR Logic (`training_step`)

```python
def my_training_step(pipeline_model, batch):
    x, y_true = batch
    y_pred = pipeline_model(x)
    loss = metrics.bce(y_pred, y_true)
    return {"loss": loss, "y_pred": y_pred}
```

### Step 4: Run the Training Loop & Log with Experiment

```python
# Assuming my_train_data and my_eval_data are defined
# Assuming an optimizer is initialized, e.g., optimizer = idrak.optim.Adam(model.params, lr=hyperparams["learning_rate"])

for epoch in range(hyperparams["epochs"]):
    total_loss = 0
    for i, batch in enumerate(my_train_data):
        # Forward pass, backward pass, and optimization
        results = my_training_step(model, batch)
        loss = results["loss"]
        # ... (backward pass and optimizer step would go here) ...

        total_loss += loss.item()
        exp.log_metric("train_loss", loss.item(), step=epoch * len(my_train_data) + i)

    avg_train_loss = total_loss / len(my_train_data)
    exp.log_metric("avg_epoch_train_loss", avg_train_loss, step=epoch)

    # Evaluation loop (simplified)
    eval_loss = 0
    for batch in my_eval_data:
        results = my_training_step(model, batch) # Using same step for simplicity
        eval_loss += results["loss"].item()
    avg_eval_loss = eval_loss / len(my_eval_data)
    exp.log_metric("avg_epoch_eval_loss", avg_eval_loss, step=epoch)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Eval Loss = {avg_eval_loss:.4f}")

exp.save() # Save experiment results
```

---

## 💣 Hackability Is Not a Feature. It's the Point.

Most libraries hide their internals like it’s some sacred artifact. Idrak doesn’t do that. Everything is an object you can poke, inspect, and mutate at runtime. We tear down the walls.

-   🔍 Access **all model parameters** with `pipeline.params`
-   🧱 Access **buffers** (non-trainable states) with `pipeline.buffers`
-   🧠 Access individual layers by index: `my_layer = pipeline[idx]`
-   💥 Inject any arbitrary PyFunc into a pipeline step
-   👁️‍🗨️ Pure functional `>>` pipelines — you trace, log, wrap, or fuse
-   🧬 Plug and play — it’s just Python dicts and functional calls

Frameworks treat you like a user. Idrak treats you like a hacker.

### Dynamic Architecture: Swap Layers, Reset State, or Keep Going. Your Call.

Idrak gives you the power to fundamentally alter your network's structure and state *mid-run*. This isn't just flexibility; it's a declaration of war on static, rigid architectures.

```python
from idrak.nn import Conv2d, Linear, Pipeline
import idrak.core.tensor as T

# Assume 'model' is an existing Pipeline
# Assume 'input_tensor' is a Tensor

# Original first layer
print(f"Original first layer: {model[0]}")

# Create a new layer with different parameters
new_first_layer = Conv2d(3, 128, kernel_size=3)

# --- Scenario 1: Swap and reset parameters of the *entire* pipeline ---
# This is for when you want a fresh start after a major architectural change.
print("\n--- Swapping layer and resetting ALL parameters ---")
model[0] = new_first_layer
model.reset_parameters() # Re-initializes ALL layers in the pipeline
print(f"New first layer after full reset: {model[0]}")
# All weights in the model are now re-randomized.

# --- Scenario 2: Swap and *only* reset parameters of the new layer ---
# Ideal for fine-tuning or targeted modifications without disturbing learned weights elsewhere.
print("\n--- Swapping layer without full pipeline reset ---")
another_new_layer = Linear(10, 5) # Assuming this fits the pipeline
original_layer_at_idx = model[1] # Save original to compare
model[1] = another_new_layer
if isinstance(model[1], T.Module): # Check if it's a Module to reset
    model[1].reset_parameters() # Only resets the new layer's parameters
print(f"Layer at index 1 after swap (only new layer reset): {model[1]}")
# Parameters of other layers in the pipeline remain unchanged.

# --- Scenario 3: Freezing layers ---
# Lock down parts of your network. Essential for transfer learning or architectural experiments.
print("\n--- Freezing a layer ---")
print(f"Requires grad for first layer before freeze: {model[0].params[0].requires_grad}")
model[0].freeze() # Freeze the first layer
print(f"Requires grad for first layer after freeze: {model[0].params[0].requires_grad}")
# This layer's parameters will no longer be updated during backpropagation.
```

You want to freeze weights? Detach the gradient. You want to mutate activations on the fly? Go ahead. Want to track internal outputs? Inject a hook or just override the fn inline. It's your engine. Drive it like a maniac.

---

## 🔥 The Vision: A Fully Fused Stack & Lazy Evaluation

Idrak's design enables our ultimate goal: bridging the gap between high-level expression and bare-metal performance, powered by lazy evaluation. This isn't just an optimization; it's a declaration of intent.

-   **JIT Compiler for CUDA:** The explicit pipeline (`>>`) is a parsable AST. We will trace and fuse it into high-performance CUDA kernels, leveraging lazy evaluation for optimal execution. This isn't just an optimization; it's a fundamental shift in how we execute deep learning.
-   **Lazy Evaluation for Efficiency:** Computations are only performed when their results are actually needed, leading to significant performance gains and reduced memory footprint, especially for complex graphs. Why waste cycles on intermediate results that might never be used?

---

## ✅ Core Features & Status

-   ✅ Core Components: Flexible `Pipeline` for model definition and composition, `Experiment` for tracking.
-   ✅ Transparent API: Index/slice-based layer access and explicit data flows.
-   ✅ Core Autograd Engine: Tape-based and fully functional.
-   🔧 JIT Compilation Engine: AST parsing and fusion (In Development)
-   ✅ Built-in Metrics: Accuracy, Precision, F1, etc.
-   ✅ Core Layers, Losses, Optimizers
-   ✅ CUDA Backend: Custom kernels for performance
-   ✅ Lazy Evaluation: Core to the computational graph and optimization strategy.
-   ✅ Runtime Layer Swapping & Flexible Parameter Initialization (`reset_parameters()`).
-   ✅ Layer Freezing (`freeze()`).

---

## 🚀 Getting Started

```bash
git clone https://github.com/yushi2006/idrak.git
cd idrak
pip install -e .
```

---

## 📄 License

MIT License — free to use, modify, and commercialize with attribution.

