from __future__ import annotations
from os import wait
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

    # idrak/ops/view_ops.py (continued)

# Assuming c_view, c_unsqueeze, etc. are imported
from idrak.idrak_bindings.c_wrapper_functions import (
    c_view, c_unsqueeze, c_squeeze, c_expand, c_broadcast, c_transpose,
    # c_view_grad_op, c_unsqueeze_grad_op, etc. (you'll need these)
)

class ViewOp(LazyOp):
    """
    Base class for operations that create a view of an existing tensor,
    modifying its shape/strides but sharing the underlying data storage.
    """
    @staticmethod
    def calc_out_shape(*args, **kwargs) -> tuple[int, ...]:
        # View ops typically take one input tensor (args[0]) and then shape/dim args
        raise NotImplementedError("ViewOp.calc_out_shape must be implemented by subclasses.")

    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        # Capture all non-tensor arguments for the forward pass
        forward_kwargs = {k: v for k, v in kwargs.items()}
        # Also capture relevant positional arguments (e.g., shape, dim, n, m)
        if len(args) > 1: # args[0] is the input tensor, subsequent are parameters
            if isinstance(args[1], (tuple, list)): # For shape
                forward_kwargs["shape"] = tuple(args[1])
            elif isinstance(args[1], int): # For dim/n/m
                forward_kwargs["dim"] = args[1]
            if len(args) > 2 and isinstance(args[2], int): # For m in transpose
                forward_kwargs["m"] = args[2]

        # For simple view ops, backward_ctx might be None, or it could store info
        # needed for proper gradient reshaping (e.g., original shape).
        # For now, let's keep it None unless needed.
        backward_ctx = None
        return forward_kwargs, backward_ctx

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", **kwargs):
        raise NotImplementedError("ViewOp.forward must be implemented by subclasses.")

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        # For view ops, the backward pass typically involves reshaping or broadcasting
        # the incoming gradient (grad_out) back to the shape of the input tensor (a_tensor).
        # The C function `c_view_grad_op` (or similar) would perform this.
        # This will need to be implemented for each specific view op if it's not generic.
        pass


# --- View ---
class View(ViewOp): # Inherit from ViewOp
    @staticmethod
    def calc_out_shape(a: "Tensor", shape: tuple[int, ...]) -> tuple[int, ...]: # <- CHANGED signature to match *args, **kwargs
        if not isinstance(a, CTensor):
            raise TypeError("View requires a Tensor as the first argument.")
        if not isinstance(shape, (tuple, list)):
            raise TypeError("View requires shape to be a tuple or list.")

        # Tensor.safe_c_numel needs to be a static method or global helper
        # Ensure your Tensor class has this, or use a helper from C bindings directly.
        from idrak.core.tensor import Tensor # Re-import if needed
        # Assuming Tensor.numel() now correctly calculates Python numel
        # or use a dedicated C function like `c_numel_from_shape(shape_ptr, ndim)`
        
        target_numel = 1
        for dim in shape:
            if dim == -1:
                # Calculate -1 dimension based on total elements
                # This logic is tricky: must be exactly one -1
                if target_numel == 0: # Avoid division by zero
                    raise ValueError("Cannot infer -1 dimension if remaining size is 0.")
                target_numel = a.numel() / target_numel # Temporarily calculate
            else:
                target_numel *= dim
        
        if a.numel() != int(target_numel):
            raise RuntimeError(
                f"Unable to view as numel mismatch: {a.numel()} != {int(target_numel)}"
            )
        
        # Now replace the -1 in shape with the actual calculated dimension
        final_shape = list(shape)
        if -1 in final_shape:
            idx = final_shape.index(-1)
            remaining_prod = 1
            for i, dim in enumerate(final_shape):
                if i != idx:
                    remaining_prod *= dim
            final_shape[idx] = a.numel() // remaining_prod

        return tuple(final_shape)

    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]: # <- OVERRIDE from ViewOp
        if len(args) < 2:
            raise ValueError("View.create_ctx_struct requires a Tensor and a shape.")
        
        shape = args[1] # The shape is the second positional argument
        if not isinstance(shape, (tuple, list)):
            raise TypeError("Shape must be a tuple or list.")
        
        forward_kwargs = {"shape": tuple(shape)}
        return forward_kwargs, None

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", shape: tuple[int, ...]): # <- CHANGED signature
        # This assumes c_view updates out.c_tensor_ptr to be a view of a_tensor.c_tensor_ptr->data
        # and increments the underlying Storage's ref count.
        c_view(a_tensor.c_tensor_ptr, out.c_tensor_ptr, (ctypes.c_int * len(shape))(*shape), len(shape))
        # The 'out' tensor was already allocated with the target shape by LazyOp.create_node.
        # c_view should then internally update out.c_tensor_ptr->data, strides, etc.

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        # You'll need a C function for view backward
        # It typically reshapes grad_out back to the input's original shape.
        # c_view_grad_op(out_ptr, prev_ptrs[0], original_shape_ptr, original_ndim, extras)
        # For now, it's a no-op placeholder.
        pass # raise NotImplementedError("View backward not implemented.")


# --- Unsqueeze ---
class Unsqueeze(ViewOp):
    @staticmethod
    def calc_out_shape(a: "Tensor", dim: int) -> tuple[int, ...]: # <- CHANGED signature
        if not isinstance(a, CTensor):
            raise TypeError("Unsqueeze requires a Tensor as the first argument.")
        if not isinstance(dim, int):
            raise TypeError("Unsqueeze requires dim to be an integer.")

        if dim < 0:
            dim = a.ndim + dim + 1
        if dim > a.ndim: # Allows appending at the end (dim == a.ndim)
            raise ValueError(f"Can't unsqueeze dim {dim} for tensor with {a.ndim} dimensions.")
        new_shape = list(a.shape)
        new_shape.insert(dim, 1)
        return tuple(new_shape)

    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]: # <- OVERRIDE
        if len(args) < 2:
            raise ValueError("Unsqueeze.create_ctx_struct requires a Tensor and a dim.")
        
        dim = args[1]
        if not isinstance(dim, int):
            raise TypeError("Dim must be an integer.")
        
        forward_kwargs = {"dim": dim}
        return forward_kwargs, None

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", dim: int): # <- CHANGED signature
        c_unsqueeze(a_tensor.c_tensor_ptr, out.c_tensor_ptr, dim)
    
    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        # You'll need a C function for unsqueeze backward (typically a squeeze)
        # c_unsqueeze_grad_op(out_ptr, prev_ptrs[0], dim, extras)
        pass # raise NotImplementedError("Unsqueeze backward not implemented.")


# --- Squeeze ---
class Squeeze(ViewOp):
    @staticmethod
    def calc_out_shape(a: "Tensor", dim: Optional[int] = None) -> tuple[int, ...]: # <- CHANGED signature
        if not isinstance(a, CTensor):
            raise TypeError("Squeeze requires a Tensor as the first argument.")
        
        new_shape = list(a.shape)
        
        if dim is not None:
            if not isinstance(dim, int):
                raise TypeError("Squeeze requires dim to be an integer or None.")
            if dim < 0:
                dim = a.ndim + dim
            if dim < 0 or dim >= a.ndim:
                raise IndexError(f"Dimension out of range (expected to be in the range of [-{a.ndim}, {a.ndim-1}], but got {dim})")
            
            if new_shape[dim] == 1:
                new_shape.pop(dim)
            else:
                # If the specified dimension is not 1, it's a no-op for that dim, or raise error.
                # PyTorch allows it to be a no-op, NumPy raises error if axis not 1.
                # Let's align with PyTorch (no-op).
                pass
        else: # Squeeze all dimensions of size 1
            new_shape = [s for s in new_shape if s != 1]
            
        return tuple(new_shape)

    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]: # <- OVERRIDE
        if len(args) < 1:
            raise ValueError("Squeeze.create_ctx_struct requires a Tensor.")
        
        dim = args[1] if len(args) > 1 else None # Dim is optional
        
        forward_kwargs = {"dim": dim} # Store None if no dim specified
        return forward_kwargs, None

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", dim: Optional[int] = None): # <- CHANGED signature
        # Your c_squeeze likely needs the calculated out_shape or similar context
        # to correctly set strides, etc.
        # Assuming c_squeeze can infer from `out.c_tensor_ptr`'s already set shape.
        c_squeeze(a_tensor.c_tensor_ptr, out.c_tensor_ptr, dim if dim is not None else -1) # -1 might mean 'all' in C

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        # c_squeeze_grad_op(out_ptr, prev_ptrs[0], dim, extras) (typically an unsqueeze)
        pass # raise NotImplementedError("Squeeze backward not implemented.")


# --- Transpose ---
class Transpose(ViewOp):
    @staticmethod
    def calc_out_shape(a: "Tensor", dim0: int, dim1: int) -> tuple[int, ...]: # <- CHANGED signature
        if not isinstance(a, CTensor):
            raise TypeError("Transpose requires a Tensor as the first argument.")
        if not isinstance(dim0, int) or not isinstance(dim1, int):
            raise TypeError("Transpose requires dim0 and dim1 to be integers.")

        new_shape = list(a.shape)

        # Normalize negative dimensions
        if dim0 < 0:
            dim0 = a.ndim + dim0
        if dim1 < 0:
            dim1 = a.ndim + dim1

        if not (0 <= dim0 < a.ndim and 0 <= dim1 < a.ndim):
            raise IndexError(f"Dimensions out of range for transpose: {dim0}, {dim1} for tensor with {a.ndim} dims.")

        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        return tuple(new_shape)

    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]: # <- OVERRIDE
        if len(args) < 3:
            raise ValueError("Transpose.create_ctx_struct requires a Tensor and two dimensions.")
        
        dim0, dim1 = args[1], args[2]
        if not isinstance(dim0, int) or not isinstance(dim1, int):
            raise TypeError("Dim0 and Dim1 must be integers.")
        
        forward_kwargs = {"dim0": dim0, "dim1": dim1}
        return forward_kwargs, None

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", dim0: int, dim1: int): # <- CHANGED signature
        c_transpose(a_tensor.c_tensor_ptr, out.c_tensor_ptr, dim0, dim1)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        # c_transpose_grad_op(out_ptr, prev_ptrs[0], dim0, dim1, extras) (transpose again)
        pass # raise NotImplementedError("Transpose backward not implemented.")


# --- Expand ---
class Expand(ViewOp):
    @staticmethod
    def calc_out_shape(a: "Tensor", shape: tuple[int, ...]) -> tuple[int, ...]: # <- CHANGED signature
        if not isinstance(a, CTensor):
            raise TypeError("Expand requires a Tensor as the first argument.")
        if not isinstance(shape, (tuple, list)):
            raise TypeError("Expand requires shape to be a tuple or list.")

        # Expand rules:
        # 1. Output shape must have same or more dimensions than input.
        # 2. For each dimension, it must be either:
        #    a. Equal to input dim.
        #    b. -1 (meaning it's inferred from input)
        #    c. 1 (meaning input dim must be 1)
        # This is more complex than simple `Tensor.broadcast` for two operands.
        # This is `tensor.expand(target_shape)`
        
        if len(shape) < a.ndim:
            raise ValueError(f"Expand target shape {shape} must have at least as many dimensions as input {a.shape}.")

        expanded_shape = list(shape)
        
        # Align shapes from the right
        padded_a_shape = (1,) * (len(expanded_shape) - a.ndim) + a.shape

        for i in range(len(expanded_shape)):
            input_dim = padded_a_shape[i]
            target_dim = expanded_shape[i]

            if target_dim == -1: # Infer from input
                expanded_shape[i] = input_dim
            elif input_dim != 1 and input_dim != target_dim:
                raise ValueError(
                    f"Can't expand dimension {i} from size {input_dim} to {target_dim}. "
                    "Input dimension must be 1 or equal to target dimension."
                )
        return tuple(expanded_shape)

    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]: # <- OVERRIDE
        if len(args) < 2:
            raise ValueError("Expand.create_ctx_struct requires a Tensor and a shape.")
        
        shape = args[1]
        if not isinstance(shape, (tuple, list)):
            raise TypeError("Shape must be a tuple or list.")
        
        forward_kwargs = {"shape": tuple(shape)}
        return forward_kwargs, None

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", shape: tuple[int, ...]): # <- CHANGED signature
        # c_expand should handle setting strides for broadcasting.
        # The `ndim` argument in your original `Broadcast.create_node` seems odd here for expand.
        # `c_expand` likely needs `a.c_tensor_ptr`, `out.c_tensor_ptr`, and the target `shape`.
        c_expand(a_tensor.c_tensor_ptr, out.c_tensor_ptr, (ctypes.c_int * len(shape))(*shape), len(shape)) # Assuming C expects ndim
    
    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        # c_expand_grad_op(out_ptr, prev_ptrs[0], original_shape_ptr, original_ndim, extras) (typically a sum over expanded dims)
        pass # raise NotImplementedError("Expand backward not implemented.")


# --- Broadcast (Note: This is different from Tensor.broadcast method used internally) ---
# This Broadcast Op seems to be for creating a new tensor that is an explicit broadcasted version.
# If your Tensor.broadcast method exists, this operation might be redundant as an exposed LazyOp.
# However, if your c_broadcast directly takes source and target, we can keep it.
class Broadcast(ViewOp): # Inherit from ViewOp
    @staticmethod
    def calc_out_shape(a: "Tensor", shape: tuple[int, ...]) -> tuple[int, ...]: # <- CHANGED signature
        from idrak.core.tensor import Tensor
        if not isinstance(a, Tensor):
            raise TypeError("Broadcast requires a Tensor as the first argument.")
        if not isinstance(shape, (tuple, list)):
            raise TypeError("Broadcast requires shape to be a tuple or list.")
        
        # Broadcasting logic as per numpy rules for two shapes to get result shape.
        # This is often used for binary ops. For a single tensor, `expand` is usually used.
        # If this `Broadcast` operation is intended to be the generic broadcasting engine
        # that `BOp.compute_broadcasted_shape` uses, then its role is different.
        # Assuming this is to create a new tensor explicitly broadcasted to `shape`.
        
        # Check if 'a' is broadcastable to 'shape'
        max_ndim = max(a.ndim, len(shape))
        padded_a_shape = (1,) * (max_ndim - a.ndim) + a.shape
        padded_target_shape = (1,) * (max_ndim - len(shape)) + shape
        
        result_shape = []
        for dim_a, dim_target in zip(padded_a_shape, padded_target_shape):
            if dim_a == dim_target:
                result_shape.append(dim_target)
            elif dim_a == 1:
                result_shape.append(dim_target)
            elif dim_target == 1: # Can't broadcast input_dim > 1 to target_dim = 1
                raise ValueError(f"Cannot broadcast dimension from size {dim_a} to {dim_target}.")
            else: # Mismatch and neither is 1
                raise ValueError(f"Shapes are not broadcastable: {a.shape} to {shape}. Mismatch at dim ({dim_a} vs {dim_target})")
        
        return tuple(result_shape)


    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]: # <- OVERRIDE
        if len(args) < 2:
            raise ValueError("Broadcast.create_ctx_struct requires a Tensor and a shape.")
        
        shape = args[1]
        if not isinstance(shape, (tuple, list)):
            raise TypeError("Shape must be a tuple or list.")
        
        forward_kwargs = {"shape": tuple(shape)}
        return forward_kwargs, None

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", shape: tuple[int, ...]): # <- CHANGED signature
        c_broadcast(a_tensor.c_tensor_ptr, out.c_tensor_ptr, shape, len(shape)) 

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        # c_broadcast_grad_op(out_ptr, prev_ptrs[0], original_shape_ptr, original_ndim, extras)
        pass # raise NotImplementedError("Broadcast backward not implemented.")


# --- Concat ---
# Concat is NOT a view op; it creates new data by copying.
class Concat(LazyOp): # Stays as LazyOp base
    @staticmethod
    def calc_out_shape(*args, **kwargs) -> tuple[int, ...]: # <- CHANGED signature
        if not args:
            raise ValueError("Concat requires at least a list of Tensors.")
        
        tensors = args[0]
        if not isinstance(tensors, (list, tuple)) or not all(isinstance(t, CTensor) for t in tensors):
            raise TypeError("First argument to Concat must be a list/tuple of Tensors.")
        if not tensors:
            raise ValueError("Concat requires at least one tensor.")

        axis = kwargs.get("axis", 0)
        if not isinstance(axis, int):
            raise TypeError("Concat axis must be an integer.")
            
        if axis < 0:
            axis = tensors[0].ndim + axis
        
        if not (0 <= axis < tensors[0].ndim):
            raise IndexError(f"Axis {axis} out of bounds for tensor with {tensors[0].ndim} dimensions.")

        shape = list(tensors[0].shape)

        for i in range(1, len(tensors)):
            if tensors[i].ndim != tensors[0].ndim:
                raise ValueError("All tensors for concat must have the same number of dimensions.")
            shape[axis] += tensors[i].shape[axis] # Sum along the concat axis
            for j in range(len(tensors[0].shape)):
                if j != axis and tensors[i].shape[j] != tensors[0].shape[j]:
                    raise ValueError(f"Can't concat: dimension {j} mismatch ({tensors[i].shape[j]} vs {tensors[0].shape[j]}).")
        return tuple(shape)

    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]: # <- CHANGED signature and return type
        axis = kwargs.get("axis", 0)
        forward_kwargs = {"axis": axis}
        # For backward, you'll need the original shapes of the input tensors to split the gradient
        # This means `extras` should probably store a list of (shape, numel) for each input tensor.
        # For simplicity, we'll keep it simple for now, but this is a point of expansion.
        backward_ctx = {"axis": axis, "input_shapes": [t.shape for t in args[0]]} # Store shapes for backward
        return forward_kwargs, backward_ctx
    
    @staticmethod
    def forward(out: "Tensor", a_tensors: List["Tensor"], axis: int): # <- CHANGED signature
        inputs = []
        for t in a_tensors: # a_tensors is now the list of input Tensors
            inputs.append(t.c_tensor_ptr)
        
        # C function needs `ctypes.POINTER(ctypes.POINTER(CTensor))` for the list of input tensors
        c_inputs_array = (ctypes.POINTER(CTensor) * len(inputs))(*inputs)
        
        c_concat(c_inputs_array, out.c_tensor_ptr, len(inputs), axis)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any): # <- CHANGED signature for clarity and type hints
        # For concat backward, prev_ptrs refers to the *input* tensors to concat.
        # We need to split `out_ptr`'s gradient back according to original input shapes and axis.
        # `extras` should contain the input shapes and axis.
        c_concat_grad_op(out_ptr, prev_ptrs, n_prev, extras)


# --- Stack ---
# Stack is also NOT a view op, it creates new data. It's often implemented via unsqueeze + concat.
class Stack(LazyOp): # Stays as LazyOp base
    @staticmethod
    def calc_out_shape(*args, **kwargs) -> tuple[int, ...]: # <- CHANGED signature
        if not args:
            raise ValueError("Stack requires at least a list of Tensors.")
        
        tensors = args[0]
        if not isinstance(tensors, (list, tuple)) or not all(isinstance(t, CTensor) for t in tensors):
            raise TypeError("First argument to Stack must be a list/tuple of Tensors.")
        if not tensors:
            raise ValueError("Stack requires at least one tensor.")

        # All tensors must have the same shape for stacking
        first_shape = tensors[0].shape
        for i in range(1, len(tensors)):
            if tensors[i].shape != first_shape:
                raise ValueError("All input tensors to stack must have the same shape.")

        axis = kwargs.get("axis", 0)
        if not isinstance(axis, int):
            raise TypeError("Stack axis must be an integer.")

        if axis < 0:
            axis = tensors[0].ndim + axis + 1 # +1 because stack adds a new dimension

        if not (0 <= axis <= tensors[0].ndim): # Inclusive of ndim for appending
            raise IndexError(f"Axis {axis} out of bounds for tensor with {tensors[0].ndim} dimensions.")

        shape = list(first_shape)
        shape.insert(axis, len(tensors)) # Insert the new dimension
        
        return tuple(shape)

    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]: # <- CHANGED signature and return type
        axis = kwargs.get("axis", 0)
        forward_kwargs = {"axis": axis}
        # For backward, you'll need the axis and potentially the original shapes (which are all the same)
        backward_ctx = {"axis": axis, "input_shape": args[0][0].shape} # Assuming first tensor's shape is representative
        return forward_kwargs, backward_ctx

    @staticmethod
    def forward(out: "Tensor", a_tensors: List["Tensor"], axis: int): # <- CHANGED signature
        # Stack is often implemented by unsqueezing each input tensor at `axis`
        # and then concatenating them along that `axis`.
        # Your c_concat function needs to handle this.
        # This implies c_concat will get a list of already-unsqueezed-tensors.
        
        # This logic needs to be careful:
        # 1. Create temporary unsqueezed views of each a_tensor.
        # 2. Collect their c_tensor_ptr.
        # 3. Call c_concat.
        
        # This is more complex than just passing `a_tensors` directly to c_concat.
        # A simple `c_stack` C function would be better, or manually perform unsqueeze+concat if no `c_stack`.
        
        # If your c_concat is designed to handle "stacking" by implicitly unsqueezing:
        inputs_c_ptrs = []
        for t in a_tensors:
            inputs_c_ptrs.append(t.c_tensor_ptr)
        
        c_inputs_array = (ctypes.POINTER(CTensor) * len(inputs_c_ptrs))(*inputs_c_ptrs)
        c_concat(c_inputs_array, out.c_tensor_ptr, len(inputs_c_ptrs), axis) # Assuming c_concat handles the effective unsqueeze

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any): # <- CHANGED signature
        # Similar to concat, but the backward involves splitting and then squeezing each part.
        c_concat_grad_op(out_ptr, prev_ptrs, n_prev, extras) # Assuming concat_grad_op can handle the stack context
        # (This implies concat_grad_op needs to understand the difference between concat and stack for its gradient calculation).
        # A dedicated `c_stack_grad_op` might be clearer if the gradient logic is distinct.
