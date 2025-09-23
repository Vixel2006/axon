from __future__ import annotations
import math
from typing import Any
import ctypes
from .op import LazyOp
from idrak.idrak_bindings.ctypes_definitions import CTensor
from idrak.idrak_bindings.c_wrapper_functions import (
    c_concat_grad_op,
    c_view,
    c_unsqueeze,
    c_squeeze,
    c_expand,
    c_broadcast,
    c_transpose,
    c_concat,
)

class ViewOp(LazyOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        raise NotImplementedError("ViewOp.calc_out_shape must be implemented by subclasses.")

    @staticmethod
    def create_ctx_struct(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
        return {}, None

    @staticmethod
    def forward(out: "Tensor", *args: Any, **kwargs: Any):
        raise NotImplementedError("ViewOp.forward must be implemented by subclasses.")

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        pass


class View(ViewOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        from idrak.core.tensor import Tensor
        a: Tensor = args[0]
        shape: tuple[int, ...] = kwargs.get("shape") or args[1]

        if not isinstance(a, Tensor):
            raise TypeError("View requires a Tensor as the first argument.")
        if not isinstance(shape, (tuple, list)):
            raise TypeError("View requires shape to be a tuple or list.")

        target_numel_fixed_dims = 1
        has_negative_one = False
        negative_one_idx = -1

        for idx, dim in enumerate(shape):
            if dim == -1:
                if has_negative_one:
                    raise ValueError("Can only specify one -1 in shape.")
                has_negative_one = True
                negative_one_idx = idx
            elif dim <= 0:
                raise ValueError(f"Invalid dimension {dim} in shape. Dimensions must be positive or -1.")
            else:
                target_numel_fixed_dims *= dim

        a_numel = a.numel()

        if has_negative_one:
            if target_numel_fixed_dims == 0:
                raise ValueError("Cannot infer -1 dimension if product of other dimensions is 0.")
            if a_numel % target_numel_fixed_dims != 0:
                raise ValueError(
                    f"Unable to infer -1 dimension: total elements {a_numel} "
                    f"not divisible by product of other dimensions {target_numel_fixed_dims}."
                )
            inferred_dim = a_numel // target_numel_fixed_dims
            final_shape_list = list(shape)
            final_shape_list[negative_one_idx] = inferred_dim
            final_shape = tuple(final_shape_list)
        else:
            final_shape = tuple(shape)

        if a_numel != math.prod(final_shape):
            raise RuntimeError(
                f"Unable to view as numel mismatch: {a_numel} != {math.prod(final_shape)}"
            )

        return final_shape

    @staticmethod
    def create_ctx_struct(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
        if len(args) < 2:
            raise ValueError("View.create_ctx_struct requires a Tensor and a shape.")

        shape = args[1]
        if not isinstance(shape, (tuple, list)):
            raise TypeError("Shape must be a tuple or list.")

        forward_kwargs = {"shape": tuple(shape)}
        return forward_kwargs, None

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", shape: tuple[int, ...]):
        c_view(a_tensor.c_tensor_ptr, out.c_tensor_ptr, (ctypes.c_int * len(shape))(*shape), len(shape))


class Unsqueeze(ViewOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        from idrak.core.tensor import Tensor
        a: Tensor = args[0]
        dim: int = kwargs.get("dim") if "dim" in kwargs else args[1]

        if not isinstance(a, Tensor):
            raise TypeError("Unsqueeze requires a Tensor as the first argument.")
        if not isinstance(dim, int):
            raise TypeError("Unsqueeze requires dim to be an integer.")

        if dim < 0:
            dim = a.ndim + dim + 1
        if not (0 <= dim <= a.ndim):
            raise ValueError(f"Dimension {dim} out of range for tensor with {a.ndim} dimensions.")

        new_shape = list(a.shape)
        new_shape.insert(dim, 1)
        return tuple(new_shape)

    @staticmethod
    def create_ctx_struct(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
        if len(args) < 2:
            raise ValueError("Unsqueeze.create_ctx_struct requires a Tensor and a dim.")

        dim = args[1]
        if not isinstance(dim, int):
            raise TypeError("Dim must be an integer.")

        if dim < 0:
            dim = args[0].ndim + dim
        
        forward_kwargs = {"dim": dim}
        return forward_kwargs, dim

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", dim: int):
        c_unsqueeze(a_tensor.c_tensor_ptr, out.c_tensor_ptr, dim)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        pass


class Squeeze(ViewOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        from idrak.core.tensor import Tensor
        a: Tensor = args[0]
        dim: Optional[int] = kwargs.get("dim")
        if dim is None and len(args) > 1 and isinstance(args[1], int):
            dim = args[1]

        if not isinstance(a, Tensor):
            raise TypeError("Squeeze requires a Tensor as the first argument.")

        new_shape = list(a.shape)

        if dim is not None:
            if not isinstance(dim, int):
                raise TypeError("Squeeze requires dim to be an integer or None.")
            if dim < 0:
                dim = a.ndim + dim
            if not (0 <= dim < a.ndim):
                raise IndexError(f"Dimension out of range (expected to be in the range of [-{a.ndim}, {a.ndim-1}], but got {dim})")

            if new_shape[dim] == 1:
                new_shape.pop(dim)
        else:
            new_shape = [s for s in new_shape if s != 1]

        return tuple(new_shape)

    @staticmethod
    def create_ctx_struct(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
        if len(args) < 1:
            raise ValueError("Squeeze.create_ctx_struct requires a Tensor.")

        dim: Optional[int] = None
        if "dim" in kwargs:
            dim = kwargs["dim"]
        elif len(args) > 1 and isinstance(args[1], int):
            dim = args[1]

        if dim < 0:
            dim = args[0].ndim + dim
        
        forward_kwargs = {"dim": dim}
        return forward_kwargs, dim

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", dim: Optional[int] = None):
        c_squeeze(a_tensor.c_tensor_ptr, out.c_tensor_ptr, dim if dim is not None else -1)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        pass


class Transpose(ViewOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        from idrak.core.tensor import Tensor
        a: Tensor = args[0]
        dim0: int = kwargs.get("dim0") if "dim0" in kwargs else args[1]
        dim1: int = kwargs.get("dim1") if "dim1" in kwargs else args[2]

        if not isinstance(a, Tensor):
            raise TypeError("Transpose requires a Tensor as the first argument.")
        if not isinstance(dim0, int) or not isinstance(dim1, int):
            raise TypeError("Transpose requires dim0 and dim1 to be integers.")

        new_shape = list(a.shape)

        if dim0 < 0:
            dim0 = a.ndim + dim0
        if dim1 < 0:
            dim1 = a.ndim + dim1

        if not (0 <= dim0 < a.ndim and 0 <= dim1 < a.ndim):
            raise IndexError(f"Dimensions out of range for transpose: {dim0}, {dim1} for tensor with {a.ndim} dims.")

        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        return tuple(new_shape)

    @staticmethod
    def create_ctx_struct(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
        if len(args) < 3:
            raise ValueError("Transpose.create_ctx_struct requires a Tensor and two dimensions.")

        dim0, dim1 = args[1], args[2]
        if not isinstance(dim0, int) or not isinstance(dim1, int):
            raise TypeError("Dim0 and Dim1 must be integers.")

        if dim0 < 0:
            dim0 = args[0].ndim + dim0
        if dim1 < 0:
            dim1 = args[0].ndim + dim1

        forward_kwargs = {"dim0": dim0, "dim1": dim1}
        return forward_kwargs, (dim0, dim1)

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", dim0: int, dim1: int):
        c_transpose(a_tensor.c_tensor_ptr, out.c_tensor_ptr, dim0, dim1)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        pass


class Expand(ViewOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        from idrak.core.tensor import Tensor
        a: Tensor = args[0]
        shape: tuple[int, ...] = kwargs.get("shape") or args[1]

        if not isinstance(a, Tensor):
            raise TypeError("Expand requires a Tensor as the first argument.")
        if not isinstance(shape, (tuple, list)):
            raise TypeError("Expand requires shape to be a tuple or list.")

        if len(shape) < a.ndim:
            raise ValueError(f"Expand target shape {shape} must have at least as many dimensions as input {a.shape}.")

        expanded_shape = list(shape)
        padded_a_shape = (1,) * (len(expanded_shape) - a.ndim) + a.shape

        for i in range(len(expanded_shape)):
            input_dim = padded_a_shape[i]
            target_dim = expanded_shape[i]

            if target_dim == -1:
                expanded_shape[i] = input_dim
            elif input_dim != 1 and input_dim != target_dim:
                raise ValueError(
                    f"Can't expand dimension {i} from size {input_dim} to {target_dim}. "
                    "Input dimension must be 1 or equal to target dimension."
                )
        return tuple(expanded_shape)

    @staticmethod
    def create_ctx_struct(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
        if len(args) < 2:
            raise ValueError("Expand.create_ctx_struct requires a Tensor and a shape.")

        shape = args[1]
        if not isinstance(shape, (tuple, list)):
            raise TypeError("Shape must be a tuple or list.")

        forward_kwargs = {"shape": tuple(shape)}
        return forward_kwargs, None

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", shape: tuple[int, ...]):
        c_expand(a_tensor.c_tensor_ptr, out.c_tensor_ptr, (ctypes.c_int * len(shape))(*shape))

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        pass


class Broadcast(ViewOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        from idrak.core.tensor import Tensor
        a: Tensor = args[0]
        shape: tuple[int, ...] = kwargs.get("shape") or args[1]

        if not isinstance(a, Tensor):
            raise TypeError("Broadcast requires a Tensor as the first argument.")
        if not isinstance(shape, (tuple, list)):
            raise TypeError("Broadcast requires shape to be a tuple or list.")

        max_ndim = max(a.ndim, len(shape))
        padded_a_shape = (1,) * (max_ndim - a.ndim) + a.shape
        padded_target_shape = (1,) * (max_ndim - len(shape)) + shape

        result_shape = []
        for dim_a, dim_target in zip(padded_a_shape, padded_target_shape):
            if dim_a == dim_target:
                result_shape.append(dim_target)
            elif dim_a == 1:
                result_shape.append(dim_target)
            elif dim_target == 1:
                raise ValueError(f"Cannot broadcast dimension from size {dim_a} to {dim_target}.")
            else:
                raise ValueError(f"Shapes are not broadcastable: {a.shape} to {shape}. Mismatch at dim ({dim_a} vs {dim_target})")

        return tuple(result_shape)

    @staticmethod
    def create_ctx_struct(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
        if len(args) < 2:
            raise ValueError("Broadcast.create_ctx_struct requires a Tensor and a shape.")

        shape = args[1]
        if not isinstance(shape, (tuple, list)):
            raise TypeError("Shape must be a tuple or list.")

        forward_kwargs = {"shape": tuple(shape)}
        return forward_kwargs, None

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", shape: tuple[int, ...]):
        c_broadcast(a_tensor.c_tensor_ptr, out.c_tensor_ptr, (ctypes.c_int * len(shape))(*shape), len(shape))

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        pass

class Concat(LazyOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        from idrak.core.tensor import Tensor
        if not args:
            raise ValueError("Concat requires at least a list of Tensors.")

        tensors: List[Tensor] = args[0]
        if not isinstance(tensors, (list, tuple)) or not all(isinstance(t, Tensor) for t in tensors):
            raise TypeError("First argument to Concat must be a list/tuple of Tensors.")
        if not tensors:
            raise ValueError("Concat requires at least one tensor.")

        axis = kwargs.get("axis", 0)
        if not isinstance(axis, int):
            raise TypeError("Concat axis must be an integer.")

        first_tensor_ndim = tensors[0].ndim
        if axis < 0:
            axis = first_tensor_ndim + axis

        if not (0 <= axis < first_tensor_ndim):
            raise IndexError(f"Axis {axis} out of bounds for tensor with {first_tensor_ndim} dimensions.")

        shape = list(tensors[0].shape)

        for i in range(1, len(tensors)):
            if tensors[i].ndim != first_tensor_ndim:
                raise ValueError("All tensors for concat must have the same number of dimensions.")
            shape[axis] += tensors[i].shape[axis]
            for j in range(first_tensor_ndim):
                if j != axis and tensors[i].shape[j] != tensors[0].shape[j]:
                    raise ValueError(f"Can't concat: dimension {j} mismatch ({tensors[i].shape[j]} vs {tensors[0].shape[j]}).")
        return tuple(shape)

    @staticmethod
    def create_ctx_struct(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
        from idrak.core.tensor import Tensor
        axis = kwargs.get("axis", 0)
        forward_kwargs = {"axis": axis}
        input_shapes = [t.shape for t in args[0] if isinstance(t, Tensor)]
        backward_ctx = {"axis": axis, "input_shapes": input_shapes}
        return forward_kwargs, backward_ctx

    @staticmethod
    def forward(out: "Tensor", *a_tensors: "Tensor", axis: int):
        inputs_c_ptrs = []
        for t in a_tensors:
            inputs_c_ptrs.append(t.c_tensor_ptr)

        c_inputs_array = (ctypes.POINTER(CTensor) * len(inputs_c_ptrs))(*inputs_c_ptrs)

        c_concat(c_inputs_array, out.c_tensor_ptr, len(inputs_c_ptrs), axis)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_concat_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Stack(LazyOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        from idrak.core.tensor import Tensor
        if not args:
            raise ValueError("Stack requires at least a list of Tensors.")

        tensors: List[Tensor] = args[0]
        if not isinstance(tensors, (list, tuple)) or not all(isinstance(t, Tensor) for t in tensors):
            raise TypeError("First argument to Stack must be a list/tuple of Tensors.")
        if not tensors:
            raise ValueError("Stack requires at least one tensor.")

        first_shape = tensors[0].shape
        for i in range(1, len(tensors)):
            if tensors[i].shape != first_shape:
                raise ValueError("All input tensors to stack must have the same shape.")

        axis = kwargs.get("axis", 0)
        if not isinstance(axis, int):
            raise TypeError("Stack axis must be an integer.")

        if axis < 0:
            axis = tensors[0].ndim + axis + 1

        if not (0 <= axis <= tensors[0].ndim):
            raise IndexError(f"Axis {axis} out of bounds for tensor with {tensors[0].ndim} dimensions.")

        shape = list(first_shape)
        shape.insert(axis, len(tensors))

        return tuple(shape)

    @staticmethod
    def create_ctx_struct(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
        axis = kwargs.get("axis", 0)
        forward_kwargs = {"axis": axis}
        input_shape = args[0][0].shape if args[0] else ()
        backward_ctx = {"axis": axis, "input_shape": input_shape}
        return forward_kwargs, backward_ctx

    @staticmethod
    def forward(out: "Tensor", *a_tensors: "Tensor", axis: int):
        from idrak.core.tensor import Tensor
        temp_unsqueezed_ptrs: List[ctypes.POINTER(CTensor)] = []

        temp_unsqueezed_tensors: List[Tensor] = []

        for t in a_tensors:
            if axis < 0:
                normalized_axis_for_unsqueeze = t.ndim + axis + 1
            else:
                normalized_axis_for_unsqueeze = axis

            unsqueezed_shape = list(t.shape)
            unsqueezed_shape.insert(normalized_axis_for_unsqueeze, 1)
            unsqueezed_shape_tuple = tuple(unsqueezed_shape)

            temp_t = Tensor(shape=unsqueezed_shape_tuple, requires_grad=t.requires_grad)
            c_unsqueeze(t.c_tensor_ptr, temp_t.c_tensor_ptr, normalized_axis_for_unsqueeze)
            
            temp_unsqueezed_ptrs.append(temp_t.c_tensor_ptr)
            temp_unsqueezed_tensors.append(temp_t)

        c_inputs_array = (ctypes.POINTER(CTensor) * len(temp_unsqueezed_ptrs))(*temp_unsqueezed_ptrs)
        c_concat(c_inputs_array, out.c_tensor_ptr, len(temp_unsqueezed_ptrs), axis)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_concat_grad_op(out_ptr, prev_ptrs, n_prev, extras)

