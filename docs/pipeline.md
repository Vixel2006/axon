# Pipeline

`fajr.nn.pipeline.Pipeline` is the core class for composing neural network layers in Fajr. It allows you to chain `Module` instances together, treating your network as a sequential flow of data transformations. This design emphasizes clarity, hackability, and dynamic architecture manipulation.

## Key Features

*   **Sequential Composition:** Chain `Module`s using the `>>` operator.
*   **List-like Access:** Access and modify layers by index or slice.
*   **Dynamic Architecture:** Swap layers at runtime with ease.
*   **Automatic Parameter Management:** Automatically collects parameters and buffers from its constituent layers.

## Initialization and Composition

You can initialize a `Pipeline` by passing `Module` instances to its constructor or by using the `>>` operator.

```python
import fajr as nw
from fajr.nn import Pipeline, Linear, Conv2d
from fajr.nn.activations import ReLU

# Method 1: Pass layers to constructor
model1 = Pipeline(
    Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3)),
    ReLU(),
    Linear(in_features=16*5*5, out_features=10) # Example: assuming input size reduction to 5x5
)

# Method 2: Use the >> operator (preferred for readability and seen in mnist.py)
model2 = Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3)) \
       >> ReLU() \
       >> Linear(in_features=16*5*5, out_features=10)

# Pipelines can also be chained for modularity
feature_extractor = Conv2d(3, 32, (3,3)) >> ReLU()
classifier = Linear(32*5*5, 10) # Example: assuming input size reduction to 5x5
full_model = feature_extractor >> classifier

# Example from mnist.py: A simple linear model with Sigmoid
mnist_model = Linear(784, 1) >> Sigmoid()
```

## Layer Access and Manipulation

`Pipeline` behaves like a Python list, allowing you to inspect and modify its layers dynamically.

```python
import fajr as nw
from fajr.nn import Pipeline, Linear, Conv2d
from fajr.nn.activations import ReLU, Sigmoid

model = Linear(784, 1) >> Sigmoid()

# Access a layer by index
first_layer = model[0]
print(f"First layer: {first_layer.__class__.__name__}")

# Access a slice of layers (returns a new Pipeline)
middle_layers = model[0:1] # Example: just the first layer
print(f"Middle layers: {[layer.__class__.__name__ for layer in middle_layers.layers]}")

# Swap a layer at runtime (as seen in mnist.py)
print(f"Original second layer: {model[1].__class__.__name__}")
model[1] = ReLU() # Replace Sigmoid with ReLU
print(f"New second layer: {model[1].__class__.__name__}")

# After swapping, you might want to re-initialize parameters.
# As seen in mnist.py, calling reset_parameters() on the pipeline
# re-initializes all layers.
model.reset_parameters() # Re-initializes ALL parameters in the pipeline
print("Model parameters reset after layer swap.")

# If you only wanted to reset the new layer, you'd do it manually:
# new_relu_layer = ReLU()
# model[1] = new_relu_layer
# if isinstance(model[1], nw.nn.Module): model[1].reset_parameters()
```

## Methods

*   `forward(x: Tensor) -> Tensor`: Performs a forward pass through all layers in the pipeline.
*   `params`: A list of all learnable `Tensor` parameters within the pipeline.
*   `buffers`: A list of all non-trainable `Tensor` buffers within the pipeline.
*   `freeze()`: Sets `requires_grad=False` for all parameters in the pipeline, effectively freezing them.
*   `reset_parameters()`: Re-initializes all learnable parameters in the pipeline. Each `Module` implements its own `reset_parameters` logic.

## Example: Freezing Layers

```python
import fajr as nw
from fajr.nn import Pipeline, Linear, Conv2d
from fajr.nn.activations import ReLU

model = Linear(784, 1) >> Sigmoid()

# Check initial state of a parameter in the first layer
# (Assuming Linear layer has a 'W' parameter)
print(f"Initial requires_grad for first layer's weight: {model[0].W.requires_grad}")

# Freeze the entire model
model.freeze()
print(f"Requires_grad after freezing entire model: {model[0].W.requires_grad}")

# You can also freeze individual layers (though freezing the whole model already covers this)
# To demonstrate, let's unfreeze and then freeze just one layer
# (Note: there's no unfreeze() method, so this is conceptual for demonstration)
# For actual unfreezing, you'd need to manually set requires_grad=True for params

# Let's assume we want to freeze only the first layer after some training
# (This would typically be done before training starts)
model_selective_freeze = Linear(784, 1) >> Sigmoid()
print(f"\nBefore selective freeze: {model_selective_freeze[0].W.requires_grad}")
model_selective_freeze[0].freeze() # Freezes only the first layer
print(f"After selective freeze (first layer): {model_selective_freeze[0].W.requires_grad}")
print(f"After selective freeze (second layer): {model_selective_freeze[1].W.requires_grad}") # Sigmoid has no params, so this would be False or error
```

