from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .tensor import Tensor


# Helper function to handle broadcasting in backward passes
def _reduce_grad(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    if grad.shape == target_shape:
        return grad

    # Sum across broadcasted dimensions
    # First, handle the case where grad has more dimensions than target
    axis_to_sum = []
    for i, dim in enumerate(grad.shape):
        if i >= len(target_shape) or dim != target_shape[i]:
            axis_to_sum.append(i)

    grad = grad.sum(axis=tuple(axis_to_sum), keepdims=True)

    # Remove leading dimensions if necessary
    while grad.ndim > len(target_shape):
        grad = grad.squeeze(axis=0)

    return grad


class Autograd:
    @staticmethod
    def add_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        a, b = ctx["inputs"]
        return (_reduce_grad(grad, a.shape), _reduce_grad(grad, b.shape))

    @staticmethod
    def sub_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        a, b = ctx["inputs"]
        return (_reduce_grad(grad, a.shape), _reduce_grad(-grad, b.shape))

    @staticmethod
    def mul_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        (a,) = ctx["inputs"]
        scalar = ctx["scalar"]
        return (grad * scalar,)

    @staticmethod
    def elementwisemul_backward(
        grad: np.ndarray, ctx: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        a, b = ctx["inputs"]
        da = _reduce_grad(grad * b.data, a.shape)
        db = _reduce_grad(grad * a.data, b.shape)
        return da, db

    @staticmethod
    def pow_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        (base,) = ctx["inputs"]
        power = ctx["power"]
        return (grad * (power * base.data ** (power - 1.0)),)

    @staticmethod
    def matmul_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        a, b = ctx["inputs"]
        da = grad @ b.data.T
        db = a.data.T @ grad
        return da, db

    @staticmethod
    def neg_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        return (-grad,)

    @staticmethod
    def transpose_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        axes = ctx["axes"]
        if axes is None:
            return (grad.T,)
        else:
            inv_axes = np.argsort(axes)
            return (np.transpose(grad, inv_axes),)

    @staticmethod
    def view_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        original_shape = ctx["original_shape"]
        return (grad.reshape(original_shape),)

    @staticmethod
    def squeeze_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        original_shape = ctx["original_shape"]
        return (grad.reshape(original_shape),)

    @staticmethod
    def unsqueeze_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        original_shape = ctx["original_shape"]
        return (grad.reshape(original_shape),)

    @staticmethod
    def getitem_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        original_shape = ctx["original_shape"]
        idx = ctx["idx"]
        full_grad = np.zeros(original_shape, dtype=grad.dtype)
        full_grad[idx] = grad
        return (full_grad,)

    @staticmethod
    def broadcast_to_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        original_shape = ctx["original_shape"]
        return (_reduce_grad(grad, original_shape),)

    @staticmethod
    def sum_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        (t,) = ctx["inputs"]
        original_shape = ctx["original_shape"]
        return (np.broadcast_to(grad, original_shape),)

    @staticmethod
    def relu_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        (t,) = ctx["inputs"]
        mask = t.data > 0
        return (grad * mask,)

    @staticmethod
    def abs_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        (t,) = ctx["inputs"]
        return (grad * np.sign(t.data),)

    @staticmethod
    def conv2d_backward(
        grad_output: np.ndarray, ctx: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        x, kernel = ctx["inputs"]
        padding = ctx["padding"]
        stride = ctx["stride"]

        Sh, Sw = (stride, stride) if isinstance(stride, int) else stride
        Ph, Pw = (padding, padding) if isinstance(padding, int) else padding

        N, C_in, H, W = x.data.shape
        C_out, _, Kh, Kw = kernel.data.shape

        x_padded = (
            np.pad(x.data, ((0, 0), (0, 0), (Ph, Ph), (Pw, Pw)), mode="constant")
            if Ph > 0 or Pw > 0
            else x.data
        )
        dx_padded = np.zeros_like(x_padded)
        dW = np.zeros_like(kernel.data)

        for n in range(N):
            for i in range(grad_output.shape[2]):
                for j in range(grad_output.shape[3]):
                    patch = x_padded[n, :, i * Sh : i * Sh + Kh, j * Sw : j * Sw + Kw]
                    for c_out in range(C_out):
                        grad_val = grad_output[n, c_out, i, j]
                        if kernel.requires_grad:
                            dW[c_out] += patch * grad_val
                        if x.requires_grad:
                            dx_padded[
                                n, :, i * Sh : i * Sh + Kh, j * Sw : j * Sw + Kw
                            ] += kernel.data[c_out] * grad_val

        dx = (
            dx_padded[:, :, Ph : H + Ph, Pw : W + Pw] if Ph > 0 or Pw > 0 else dx_padded
        )
        return dx, dW

    @staticmethod
    def conv2dTranspose_backward(
        grad_output: np.ndarray, ctx: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        x, kernel = ctx["inputs"]
        padding = ctx["padding"]
        stride = ctx["stride"]

        from .ops import Ops

        dx_op = Ops.conv2d(
            Tensor(grad_output, requires_grad=False),
            kernel.T,
            padding=padding,
            stride=stride,
        )
        dx = dx_op.data

        dw = np.zeros_like(kernel.data)

        return dx, dw

    @staticmethod
    def exp_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        output_data = ctx["output_data"]

        return (grad * output_data,)

    @staticmethod
    def log_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        (t,) = ctx["inputs"]

        return (grad * (1.0 / (t.data + 1e-8)),)
