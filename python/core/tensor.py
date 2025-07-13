from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from .init_strategies import InitStrategy, OneInit, RandnInit, ZeroInit


class Tensor:
    def __init__(
        self,
        data: Union[List, np.ndarray],
        requires_grad: bool = True,
        dtype: type = np.float32,
        _ctx: Optional[tuple] = None,
    ):
        self.data = (
            np.array(data, dtype=dtype)
            if not isinstance(data, np.ndarray)
            else data.astype(dtype)
        )
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.ndim = self.data.ndim
        self.shape = self.data.shape
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._ctx = _ctx

    @property
    def T(self) -> Tensor:
        from .ops import Ops

        return Ops.transpose(self)

    @classmethod
    def from_strategy(
        cls,
        shape: tuple,
        strategy: InitStrategy,
        requires_grad: bool = True,
        dtype: type = np.float32,
    ) -> Tensor:
        data = strategy.init(shape, dtype)
        return cls(data, requires_grad, dtype=dtype)

    @classmethod
    def ones(cls, shape: tuple, **kwargs) -> Tensor:
        return cls.from_strategy(shape, OneInit(), **kwargs)

    @classmethod
    def zeros(cls, shape: tuple, **kwargs) -> Tensor:
        return cls.from_strategy(shape, ZeroInit(), **kwargs)

    @classmethod
    def randn(cls, *shape: int, **kwargs) -> Tensor:
        return cls.from_strategy(shape, RandnInit(), **kwargs)

    def __add__(self, other: Union[float, int, Tensor]) -> Tensor:
        from .ops import Ops

        return Ops.add(self, other if isinstance(other, Tensor) else Tensor(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other: Union[float, int, Tensor]) -> Tensor:
        from .ops import Ops

        return Ops.sub(self, other if isinstance(other, Tensor) else Tensor(other))

    def __rsub__(self, other):
        from .ops import Ops

        return Ops.neg(self.__sub__(other))

    def __mul__(self, other: Union[float, int, Tensor]) -> Tensor:
        from .ops import Ops

        if isinstance(other, Tensor):
            return Ops.elementwisemul(self, other)
        else:
            return Ops.mul(self, float(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other: Union[float, int, Tensor]) -> Tensor:
        return self * (other**-1.0)

    def __rtruediv__(self, other):
        return (self**-1.0) * other

    def __matmul__(self, other: Tensor) -> Tensor:
        from .ops import Ops

        return Ops.matmul(self, other)

    def __neg__(self) -> Tensor:
        from .ops import Ops

        return Ops.neg(self)

    def __pow__(self, power: float) -> Tensor:
        from .ops import Ops

        return Ops.pow(self, power)

    def __rshift__(self, other):
        from ..nn.module import Module

        if isinstance(other, Module):
            return other(self)

        elif callable(other):
            return other(self)

        else:
            raise TypeError(
                f"Unsupported operand type for >>: 'Tensor' and '{type(other).__name__}'. "
                f"Right-hand side must be an nn.Module or a callable."
            )

    def view(self, *shape: int) -> Tensor:
        from .ops import Ops

        return Ops.view(self, shape)

    def squeeze(self, dim: Optional[int] = None) -> Tensor:
        from .ops import Ops

        return Ops.squeeze(self, dim)

    def unsqueeze(self, dim: int) -> Tensor:
        from .ops import Ops

        return Ops.unsqueeze(self, dim)

    def __getitem__(self, idx) -> Tensor:
        from .ops import Ops

        return Ops.getitem(self, idx)

    def conv2d(self, kernel: Tensor, **kwargs) -> Tensor:
        from .ops import Ops

        return Ops.conv2d(self, kernel, **kwargs)

    def conv2dTranspose(self, kernel: Tensor, **kwargs) -> Tensor:
        from .ops import Ops

        return Ops.conv2dTranspose(self, kernel, **kwargs)

    def broadcast_to(self, shape: tuple) -> "Tensor":
        from .ops import Ops

        return Ops.broadcast_to(self, shape)

    def sum(self, axis=None, keepdims=False) -> Tensor:
        from .ops import Ops

        return Ops.sum(self, axis, keepdims)

    def relu(self) -> Tensor:
        from .ops import Ops

        return Ops.relu(self)

    def abs(self) -> Tensor:
        from .ops import Ops

        return Ops.abs(self)

    def exp(self) -> Tensor:
        from .ops import Ops

        return Ops.exp(self)

    def log(self) -> Tensor:
        from .ops import Ops

        return Ops.exp(self)

    @staticmethod
    def build_topo(tensor: "Tensor") -> list["Tensor"]:
        topo, visited = [], set()

        def _visit(t):
            if t not in visited:
                visited.add(t)
                if t._ctx:
                    for parent in t._ctx[1].get("inputs", []):
                        if isinstance(parent, Tensor):
                            _visit(parent)
                topo.append(t)

        _visit(tensor)
        return topo

    def backward(self, grad: Optional[np.ndarray] = None):
        if not self.requires_grad:
            raise RuntimeError(
                "Called backward on a tensor that does not require gradients."
            )

        if grad is None:
            if self.data.size != 1:
                grad = np.ones_like(self.data, dtype=self.dtype)
            else:
                grad = np.array([1.0], dtype=self.dtype)

        self.grad = grad

        topo_sorted_graph = self.build_topo(self)

        for t in reversed(topo_sorted_graph):
            if t._ctx is None or t.grad is None:
                continue

            backward_fn, arg_ctx = t._ctx
            inputs = arg_ctx.get("inputs", [])

            input_grads = backward_fn(t.grad, arg_ctx)

            if not isinstance(input_grads, tuple):
                input_grads = (input_grads,)

            if len(inputs) != len(input_grads):
                raise ValueError(
                    f"Mismatch between number of inputs ({len(inputs)}) and gradients ({len(input_grads)}) for op."
                )

            for parent_tensor, grad_for_parent in zip(inputs, input_grads):
                if isinstance(parent_tensor, Tensor) and parent_tensor.requires_grad:
                    if grad_for_parent is None:
                        continue
                    if parent_tensor.grad is None:
                        parent_tensor.grad = np.zeros_like(
                            parent_tensor.data, dtype=self.dtype
                        )
                    parent_tensor.grad += grad_for_parent

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=self.dtype)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
