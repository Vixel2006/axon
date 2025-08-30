from py.core.tensor import Tensor
import numpy as np

# Create a sample tensor
x = Tensor(shape=(2, 3), data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=False)
print(f"Original Tensor:\n{x.data}")

# Perform transpose
y = x.transpose(0, 1)
print(f"Transposed Tensor:\n{y.data}")

# Verify the shape
print(f"Transposed Tensor Shape: {y.shape}")

# Verify the data
expected_data = np.array([[1., 4.], [2., 5.], [3., 6.]])
print(f"Expected Transposed Data:\n{expected_data}")
assert np.array_equal(y.data, expected_data), "Transposed data does not match expected data!"
print("Transpose test passed!")