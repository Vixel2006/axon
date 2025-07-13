import numpy as np

from .tensor import Tensor


class Ops:
    @staticmethod
    def add(a: Tensor, b: Tensor) -> Tensor:
        data = a.data + b.data
        requires_grad = a.requires_grad or b.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (a, b)}
            _grad_fn = Autograd.add_backward
        return Tensor(data, requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def sub(a: Tensor, b: Tensor) -> Tensor:
        data = a.data - b.data
        requires_grad = a.requires_grad or b.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (a, b)}
            _grad_fn = Autograd.sub_backward
        return Tensor(data, requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def mul(a: Tensor, scalar: float) -> Tensor:
        data = a.data * scalar
        requires_grad = a.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (a,), "scalar": scalar}
            _grad_fn = Autograd.mul_backward
        return Tensor(data, requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def elementwisemul(a: Tensor, b: Tensor) -> Tensor:
        data = a.data * b.data
        requires_grad = a.requires_grad or b.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (a, b)}
            _grad_fn = Autograd.elementwisemul_backward
        return Tensor(data, requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def pow(base: Tensor, power: float) -> Tensor:
        data = base.data**power
        requires_grad = base.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (base,), "power": power}
            _grad_fn = Autograd.pow_backward
        return Tensor(data, requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def matmul(a: Tensor, b: Tensor) -> Tensor:
        data = a.data @ b.data
        requires_grad = a.requires_grad or b.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (a, b)}
            _grad_fn = Autograd.matmul_backward
        return Tensor(data, requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def neg(t: Tensor):
        data = -t.data
        requires_grad = t.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (t,)}
            _grad_fn = Autograd.neg_backward
        return Tensor(data, requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def relu(t: Tensor) -> Tensor:
        data = np.maximum(t.data, 0)
        requires_grad = t.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (t,)}
            _grad_fn = Autograd.relu_backward
        return Tensor(data, requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def transpose(t: Tensor, axes=None) -> Tensor:
        data = np.transpose(t.data, axes)
        requires_grad = t.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (t,), "axes": axes}
            _grad_fn = Autograd.transpose_backward
        return Tensor(data, requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def view(t: Tensor, shape: tuple) -> Tensor:
        data = t.data.reshape(shape)
        requires_grad = t.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (t,), "original_shape": t.shape}
            _grad_fn = Autograd.view_backward
        return Tensor(data, requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def squeeze(t: Tensor, dim: int = None) -> Tensor:
        data = np.squeeze(t.data, axis=dim)
        requires_grad = t.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (t,), "original_shape": t.shape}
            _grad_fn = Autograd.squeeze_backward
        return Tensor(data, requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def unsqueeze(t: Tensor, dim: int) -> Tensor:
        data = np.expand_dims(t.data, axis=dim)
        requires_grad = t.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (t,), "original_shape": t.shape}
            _grad_fn = Autograd.unsqueeze_backward
        return Tensor(data, requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def getitem(t: Tensor, idx) -> Tensor:
        data = t.data[idx]
        requires_grad = t.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (t,), "original_shape": t.shape, "idx": idx}
            _grad_fn = Autograd.getitem_backward
        return Tensor(data, requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def broadcast_to(t: Tensor, shape: tuple) -> Tensor:
        data = np.broadcast_to(t.data, shape)
        requires_grad = t.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (t,), "original_shape": t.shape}
            _grad_fn = Autograd.broadcast_to_backward
        return Tensor(data, requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def sum(t: Tensor, axis=None, keepdims=False) -> Tensor:
        data = t.data.sum(axis=axis, keepdims=keepdims)
        requires_grad = t.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (t,), "original_shape": t.shape}
            _grad_fn = Autograd.sum_backward
        return Tensor(data, requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def conv2d(
        t: Tensor, kernel: Tensor, padding: tuple = (0, 0), stride: tuple = (1, 1)
    ) -> Tensor:
        N, C_in, H, W = t.shape
        C_out, C_in_k, Kh, Kw = kernel.shape
        assert C_in == C_in_k, "Input channels must match kernel channels"

        Sh, Sw = (stride, stride) if isinstance(stride, int) else stride
        Ph, Pw = (padding, padding) if isinstance(padding, int) else padding

        H_out = (H + 2 * Ph - Kh) // Sh + 1
        W_out = (W + 2 * Pw - Kw) // Sw + 1
        assert H_out > 0 and W_out > 0, "Invalid output dimensions"

        t_padded = (
            np.pad(t.data, ((0, 0), (0, 0), (Ph, Ph), (Pw, Pw)), mode="constant")
            if Ph > 0 or Pw > 0
            else t.data
        )
        output = np.zeros((N, C_out, H_out, W_out), dtype=t.dtype)

        for i in range(H_out):
            for j in range(W_out):
                patch = t_padded[:, :, i * Sh : i * Sh + Kh, j * Sw : j * Sw + Kw]
                for c_out in range(C_out):
                    output[:, c_out, i, j] = np.sum(
                        patch * kernel.data[c_out], axis=(1, 2, 3)
                    )

        requires_grad = t.requires_grad or kernel.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (t, kernel), "padding": padding, "stride": stride}
            _grad_fn = Autograd.conv2d_backward
        return Tensor(output, requires_grad=requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def conv2dTranspose(
        t: Tensor, kernel: Tensor, padding: tuple = (0, 0), stride: tuple = (1, 1)
    ) -> Tensor:
        N, C_in, H, W = t.shape
        C_out, _, Kh, Kw = kernel.shape

        Sh, Sw = (stride, stride) if isinstance(stride, int) else stride
        Ph, Pw = (padding, padding) if isinstance(padding, int) else padding

        Oh = (H - 1) * Sh + Kh - 2 * Ph
        Ow = (W - 1) * Sw + Kw - 2 * Pw
        if Oh <= 0 or Ow <= 0:
            raise ValueError(f"Invalid output dimensions: Oh={Oh}, Ow={Ow}.")

        output = np.zeros((N, C_out, Oh, Ow), dtype=t.dtype)

        for i in range(H):
            for j in range(W):
                for c_out in range(C_out):
                    for c_in in range(C_in):
                        input_val = t.data[:, c_in, i, j]
                        kernel_patch = kernel.data[c_out, c_in, :, :]
                        for kh in range(Kh):
                            for kw in range(Kw):
                                out_i, out_j = i * Sh + kh - Ph, j * Sw + kw - Pw
                                if 0 <= out_i < Oh and 0 <= out_j < Ow:
                                    output[:, c_out, out_i, out_j] += (
                                        input_val * kernel_patch[kh, kw]
                                    )

        requires_grad = t.requires_grad or kernel.requires_grad
        _grad_fn, ctx = None, None
        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (t, kernel), "padding": padding, "stride": stride}
            _grad_fn = Autograd.conv2dTranspose_backward
        return Tensor(output, requires_grad=requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def abs(t: Tensor) -> Tensor:
        data = np.abs(t.data)
        requires_grad = t.requires_grad
        _grad_fn, ctx = None, None

        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (t,)}
            _grad_fn = Autograd.abs_backward

        return Tensor(data, requires_grad=requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def exp(t: Tensor) -> Tensor:
        data = np.exp(t.data)
        requires_grad = t.requires_grad
        _grad_fn, ctx = None, None

        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (t,), "output_data": data}
            _grad_fn = Autograd.exp_backward

        return Tensor(data, requires_grad=requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def log(t: Tensor) -> Tensor:
        data = np.log(t.data)
        requires_grad = t.requires_grad
        _grad_fn, ctx = None, None

        if requires_grad:
            from .autograd import Autograd

            ctx = {"inputs": (t,)}
            _grad_fn = Autograd.log_backward

        return Tensor(data, requires_grad=requires_grad, _ctx=(_grad_fn, ctx))
