# Functions (Functional API)

`fajr.functions` provides a high-level, functional API for creating tensors and performing various operations (unary, binary, movement, reduction). These functions are the primary way to interact with Idrak's lazy computation graph and build complex expressions.

## Key Categories

### Initialization Operations

Functions to create new tensors with specific initial values, this is the only operation category that does use eager execution.

*   `zeros(shape: tuple, device: str = "cpu", requires_grad: bool = True) -> Tensor`
*   `ones(shape: tuple, device: str = "cpu", requires_grad: bool = True) -> Tensor`
*   `randn(shape: tuple, seed: int = 42, device: str = "cpu", requires_grad: bool = True) -> Tensor`
*   `uniform(shape: tuple, low: float = 0.0, high: float = 1.0, device: str = "cpu", requires_grad: bool = True) -> Tensor`
*   `from_data(shape: tuple, data: list | np.ndarray, device: str = "cpu", requires_grad: bool = True) -> Tensor`

```python
import fajr
import fajr.functions as F
import numpy as np

a = F.zeros((2, 2))
b = F.ones((1, 3), requires_grad=True)
c = F.from_data((2, 2), np.array([[1, 2], [3, 4]]))
print(a)
print(b)
print(c)
```

### Unary Operations

Functions that operate on a single tensor.

*   `relu(a: Tensor) -> Tensor`
*   `log(a: Tensor) -> Tensor`
*   `exp(a: Tensor) -> Tensor`
*   `abs(a: Tensor) -> Tensor`
*   `neg(a: Tensor) -> Tensor`
*   `clip(a: Tensor, min_val: float, max_val: float) -> Tensor`

```python
import fajr
import fajr.functions as F

x = F.from_data((2,2), [[-1.0, 0.5], [2.0, -3.0]], requires_grad=True)
y = F.relu(x)
z = F.log(F.abs(x))
print(y.data)
print(z.data)
```

### Binary Operations

Functions that operate on two tensors or a tensor and a scalar.

*   `add(a: Tensor | float, b: Tensor | float) -> Tensor`
*   `sub(a: Tensor | float, b: Tensor | float) -> Tensor`
*   `mul(a: Tensor | float, b: Tensor | float) -> Tensor`
*   `div(a: Tensor | float, b: Tensor | float) -> Tensor`
*   `pow(a: Tensor, b: Tensor | float) -> Tensor`
*   `matmul(a: Tensor, b: Tensor) -> Tensor` (Matrix multiplication)
*   `dot(a: Tensor, b: Tensor) -> Tensor` (Dot product)
*   `conv2d(a: Tensor, b: Tensor, kernel_size: tuple, stride: tuple, padding: int) -> Tensor` (2D Convolution)

```python
import fajr
import fajr.functions as F
import numpy as np

a = F.ones((2, 2), requires_grad=True)
b = F.from_data((2, 2), np.array([[1, 2], [3, 4]]), requires_grad=True)

c = F.add(a, b)
d = F.mul(a, 2.0)
e = F.matmul(a, b)

print(c.data)
print(d.data)
print(e.data)
```

### Movement Operations

Functions that change the shape or arrangement of tensor elements without changing their values.

*   `view(a: Tensor, shape: tuple) -> Tensor`
*   `unsqueeze(a: Tensor, dim: int = 0) -> Tensor`
*   `squeeze(a: Tensor, dim: int = 0) -> Tensor`
*   `expand(a: Tensor, shape: tuple) -> Tensor`
*   `broadcast(a: Tensor, shape: tuple) -> Tensor`
*   `transpose(a: Tensor, n: int, m: int) -> Tensor`
*   `concat(a: list[Tensor], axis: int = 0) -> Tensor`
*   `stack(a: list[Tensor], axis: int = 0) -> Tensor`

```python
import fajr
import fajr.functions as F

x = F.ones((1, 3))
y = F.unsqueeze(x, dim=0) # Shape becomes (1, 1, 3)
z = F.squeeze(y)         # Shape becomes (1, 3) again
print(y.shape)
print(z.shape)
```

### Reduction Operations

Functions that reduce the number of dimensions of a tensor by performing an operation (e.g., sum, mean, max) along one or more axes.

*   `sum(a: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor`
*   `mean(a: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor`
*   `max(a: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor`

```python
import fajr
import fajr.functions as F

x = F.from_data((2, 3), [[1, 2, 3], [4, 5, 6]])

sum_all = F.sum(x)          # Sum of all elements
sum_dim0 = F.sum(x, dim=0)   # Sum along dimension 0
mean_dim1 = F.mean(x, dim=1, keepdim=True) # Mean along dimension 1, keep dimension

print(sum_all.data)
print(sum_dim0.data)
print(mean_dim1.data)
```

