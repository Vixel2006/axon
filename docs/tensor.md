# Tensor

`fajr.core.tensor.Tensor` is the fundamental data structure in Fajr. It represents multi-dimensional arrays and is the primary vehicle for all computations, supporting automatic differentiation and lazy evaluation.

## Key Features

*   **Multi-dimensional Arrays:** Handles data in various shapes and dimensions.
*   **Device Agnostic:** Can operate on CPU (and eventually CUDA).
*   **Automatic Differentiation:** Tracks computation history for gradient calculation (`requires_grad`).
*   **Lazy Evaluation:** Operations are recorded and executed only when `realize()` is called.

## Initialization

While `Tensor` can be directly instantiated, it's often created via functional APIs like `fajr.functions.zeros`, `fajr.functions.ones`, or `fajr.functions.from_data`.

```python
import fajr
import fajr.functions as F
import numpy as np

# Create a tensor of zeros
x = F.zeros((2, 3), requires_grad=True)
print(x)

# Create a tensor from a NumPy array
data = np.array([[1.0, 2.0], [3.0, 4.0]])
y = F.from_data(data.shape, data, requires_grad=True)
print(y)

# Accessing data (triggers realization if lazy)
print(y.data)
```

## Attributes

*   `shape`: A tuple representing the dimensions of the tensor.
*   `ndim`: The number of dimensions.
*   `device`: The device the tensor resides on (e.g., "cpu").
*   `requires_grad`: Boolean, if `True`, gradients will be computed for this tensor.
*   `data`: A NumPy array view of the tensor's data (triggers `realize()`).
*   `grad`: A NumPy array view of the tensor's gradient (triggers `realize()` and `backward()` if not already computed).

## Methods

*   `realize()`: Forces the computation of the tensor's value if it's part of a lazy graph.
*   `backward()`: Computes gradients for the tensor (if `requires_grad` is `True`).
*   `detach()`: Returns a new tensor that is detached from the current computation graph.

## Operator Overloads

`Tensor` supports standard arithmetic operations, which automatically build the computation graph:

```python
import fajr
import fajr.functions as F

a = F.ones((2, 2), requires_grad=True)
b = F.from_data((2, 2), np.array([[1, 2], [3, 4]]), requires_grad=True)

c = a + b       # Element-wise addition
d = a * 2.0     # Scalar multiplication
e = a @ b       # Matrix multiplication

print(c.data)
print(d.data)
print(e.data)

# Gradients can be computed after operations
e.backward()
print(a.grad)
print(b.grad)
```

