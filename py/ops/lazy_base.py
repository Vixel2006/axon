from typing import Any, Tuple
from py.elnawah_bindings.ctypes_definitions import CTensor


class LazyOp:
    """
    Abstract base class for lazy tensor operations.

    Responsibilities:
        - Defines the interface for computing output shapes
          without executing the actual operation.

    Methods:
        - calculate_output_shape(...):
            Must be implemented by subclasses to infer the resulting
            tensor shape from input tensor(s) and parameters.
    """

    def calculate_output_shape(self, *args: Any, **kwargs: Any) -> Tuple[int, ...]:
        raise NotImplementedError
