import pytest
import numpy as np
import ctypes
from typing import Any, Dict, List, Tuple, Optional

# Import actual classes and functions from your idrak library
from idrak.core.tensor import Tensor
from idrak.core.buffer import LazyBuffer
from idrak.ops.op import LazyOp

# Import specific operations that Tensor uses internally (e.g., for .T)
from idrak.ops.mop import Transpose, View, Unsqueeze, Squeeze, Expand, Broadcast, Concat, Stack

# Import binary, unary, and initialization operations
from idrak.ops.bop import Add, Sub, Mul, Div, Pow, MatMul, RSub, RDiv, Dot, Conv2D
from idrak.ops.uop import ReLU, Log, Exp, Abs, Neg, Clip
from idrak.ops.iop import Zeros, Ones, Uniform, Randn, FromData

# Import actual C-binding functions for verification if needed,
# though the Python Tensor methods should abstract this away.
# We assume these are correctly linked and work.
from idrak.idrak_bindings.ctypes_definitions import CTensor, CStorage, Conv2DBackwardExtras, ClipExtras
from idrak.idrak_bindings.c_wrapper_functions import (
    c_tmalloc, c_tfree, c_gmalloc, c_numel, c_compute_strides,
    c_add, c_sub, c_mul, c_div, c_pow_scalar, c_pow, c_div_scalar, c_add_scalar,
    c_sub_scalar, c_rsub_scalar, c_mul_scalar, c_conv, c_rdiv_scalar,
    c_rdiv_scalar, c_add_grad_op, c_sub_grad_op, c_mul_grad_op, c_pow_grad_op,
    c_matmul_grad_op, c_div_grad_op, c_rdiv_grad_op, c_rsub_grad_op,
    c_conv_grad_op, c_dot, c_dot_grad_op, c_relu, c_log, c_exp, c_abs, c_neg,
    c_relu_grad_op, c_log_grad_op, c_abs_grad_op, c_exp_grad_op, c_neg_grad_op,
    c_clip, c_clip_grad_op, c_concat_grad_op, c_view, c_unsqueeze, c_squeeze,
    c_expand, c_broadcast, c_transpose, c_concat, c_ones, c_zeros, c_uniform,
    c_randn, c_from_data
)

# Helper to check if a C pointer is valid (not NULL)
def is_c_ptr_valid(c_ptr):
    return bool(c_ptr) and c_ptr.contents is not None

class TestTensorCore:

    def test_tensor_init(self):
        t = Tensor((2, 3))
        assert t.shape == (2, 3)
        assert t.ndim == 2
        assert t.device == "cpu"
        assert t.requires_grad is True
        assert is_c_ptr_valid(t.c_tensor_ptr)
        assert not is_c_ptr_valid(t.c_tensor_ptr.contents.data)
        assert is_c_ptr_valid(t.c_tensor_ptr.contents.grad)
        assert np.all(t.grad == 0.0)

    def test_tensor_init_no_grad(self):
        t = Tensor((2, 2), requires_grad=False)
        assert t.requires_grad is False
        assert t.c_tensor_ptr.contents.grad is None
        # Attempting to access .grad property should raise AttributeError
        with pytest.raises(AttributeError, match="grad storage is not allocated for this tensor"):
            _ = t.grad

    def test_tensor_init_cuda(self):
        # This test assumes a CUDA device is available and the C bindings handle it.
        # If not, it might fail or behave like CPU.
        t = Tensor((1, 1), device="cuda")
        assert t.device == "cuda"
        assert t.c_tensor_ptr.contents.device == 1

    def test_tensor_data_property(self):
        t = Tensor((2, 2))
        # Fill data via direct C pointer access (simulating C computation)
        # In a real test, you'd use a creation op like FromData or an arithmetic op.
        flat_data = [1.0, 2.0, 3.0, 4.0]
        for i, val in enumerate(flat_data):
            t.c_tensor_ptr.contents.data.contents.data[i] = ctypes.c_float(val)

        expected_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        assert np.array_equal(t.data, expected_data)
        assert t.data.dtype == np.float32

    def test_tensor_grad_property(self):
        t = Tensor((2, 2), requires_grad=True)
        # Fill grad via direct C pointer access

        expected_grad = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        assert np.array_equal(t.grad, expected_grad)
        assert t.grad.dtype == np.float32

    def test_tensor_shape_property(self):
        t = Tensor((1, 2, 3))
        assert t.shape == (1, 2, 3)

    def test_tensor_strides_property(self):
        t_2x3 = Tensor((2, 3))
        # Strides for (2,3) should be (3,1)
        assert t_2x3.strides == (3, 1)

        t_1x2x3 = Tensor((1, 2, 3))
        # Strides for (1,2,3) should be (6,3,1)
        assert t_1x2x3.strides == (6, 3, 1)
        
        t_4 = Tensor((4,))
        # Strides for (4,) should be (1,)
        assert t_4.strides == (1,)

    """
    def test_tensor_ndim_property(self):
        t_1d = Tensor((5,))
        assert t_1d.ndim == 1
        t_3d = Tensor((2, 3, 4))
        assert t_3d.ndim == 3
    """

    def test_tensor_T_property(self):
        t = Tensor((2, 3))
        t_T = t.T # This should create a new Tensor with a Transpose LazyBuffer
        assert isinstance(t_T, Tensor)
        assert t_T.shape == (3, 2)
        assert t_T._lazy_buffer is not None
        assert isinstance(t_T._lazy_buffer.op, Transpose)
        assert t_T._lazy_buffer.prev[0] is t # The original tensor is a previous node

        # Realize to get the actual transposed data
        # For this to work, the C `c_transpose` function must be correctly implemented.
        # Let's manually put some data into 't'
        t_flat_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        for i, val in enumerate(t_flat_data):
            t.c_tensor_ptr.contents.data.contents.data[i] = ctypes.c_float(val)
        
        # Manually verify transpose output shape and data
        expected_transposed_data = np.array([[1., 4.], [2., 5.], [3., 6.]], dtype=np.float32)
        assert np.array_equal(t_T.realize().data, expected_transposed_data)


    def test_tensor_numel(self):
        t = Tensor((2, 3, 4))
        assert t.numel() == 2 * 3 * 4
        t2 = Tensor((5,))
        assert t2.numel() == 5
        #t3 = Tensor(()) # Scalar tensor
        #assert t3.numel() == 1


    def test_tensor_del(self):
        t = Tensor((2,2))
        # Store initial pointer value
        c_ptr_ref = t.c_tensor_ptr

        # Manually track if c_tfree is called (requires mocking c_tfree if not testing live C lib)
        # For now, we assume c_tfree is correctly implemented in the C library
        # and that Python's `del` triggers __del__ and thus c_tfree.
        
        # Before deletion, the pointer should be valid
        assert is_c_ptr_valid(c_ptr_ref)
        
        # After deletion, the Python object is gone, and c_tfree should be called
        del t
        # We can't directly check `c_ptr_ref.contents` as `c_ptr_ref` is a local Python variable
        # and might still hold a value, but the C memory should be freed.
        # A proper test would involve hooking c_tfree to record calls.
        
        # For now, we'll rely on memory leak detection tools or the C implementation being correct.
        # If the C library has assertions for double-free, we'd catch issues here.
        # This test primarily ensures __del__ doesn't crash.


class TestLazyBufferCore:

    def test_lazy_buffer_init(self):
        out_tensor = Tensor((2,2))
        mock_op = Add() # Use a real op
        prev_tensors = [Tensor((2,2))]
        forward_kwargs = {"scalar_val": 5.0}
        backward_ctx = ctypes.cast(ctypes.pointer(ctypes.c_float(5.0)), ctypes.c_void_p)
        
        lb = LazyBuffer(
            out=out_tensor,
            op=mock_op,
            prev=prev_tensors,
            forward_kwargs=forward_kwargs,
            backward_ctx=backward_ctx
        )

        assert lb.out is out_tensor
        assert lb.op is mock_op
        assert lb.prev == prev_tensors
        assert lb.forward_kwargs == forward_kwargs
        assert lb.backward_ctx == backward_ctx
        assert not lb._realized
        assert lb._topo_sorted is None

    def test_lazy_buffer_topo_sort_simple(self):
        a = Tensor((2,2), requires_grad=True)
        b = Tensor((2,2), requires_grad=True)
        c = a + b # c = Add.create_node(a, b)
        d = c.relu() # d = ReLU.create_node(c)

        # The topo sort for d should process dependencies before d itself.
        # The result should contain c's LazyBuffer then d's LazyBuffer.
        sorted_buffers = d._lazy_buffer.topo_sort()
        
        assert len(sorted_buffers) == 2
        assert c._lazy_buffer in sorted_buffers
        assert d._lazy_buffer in sorted_buffers
        assert sorted_buffers.index(c._lazy_buffer) < sorted_buffers.index(d._lazy_buffer)

    def test_lazy_buffer_topo_sort_complex(self):
        a = Tensor((2,2), requires_grad=True)
        b = Tensor((2,2), requires_grad=True)
        c = a * b
        d = c + a # 'd' depends on 'c' and 'a'. 'c' depends on 'a' and 'b'.
        e = d.exp()
        
        sorted_buffers = e._lazy_buffer.topo_sort()
        
        assert len(sorted_buffers) == 3 # c_lb, d_lb, e_lb
        assert c._lazy_buffer in sorted_buffers
        assert d._lazy_buffer in sorted_buffers
        assert e._lazy_buffer in sorted_buffers

        # Order must be dependencies first
        #assert sorted_buffers.index(c._lazy_buffer) < sorted_buffers.index(d._lazy_buffer)
        #assert sorted_buffers.index(d._lazy_buffer) < sorted_buffers.index(e._lazy_buffer)

        # `a` and `b` are base Tensors, not LazyBuffers, so they won't appear in `sorted_buffers` list.
        # But their realization (or access to their `c_tensor_ptr`) is implicit in `c` and `d` ops.

    def test_lazy_buffer_topo_sort_circular_dependency(self):
        # This scenario should generally be prevented by the design of LazyOp.create_node
        # which creates a new output tensor. However, if somehow manually created or
        # through a bug, a circular dependency could exist.
        
        # To simulate a circular dependency for this test:
        t1 = Tensor((1,), requires_grad=True)
        t2 = Tensor((1,), requires_grad=True)

        # Manually create LazyBuffers and enforce a cycle for testing topo_sort's detection
        op1 = Add()
        op2 = Add()

        # Create lazy_buffer objects
        lb1 = LazyBuffer(t1, op1, [], {}, None) # Will temporarily link prev later
        lb2 = LazyBuffer(t2, op2, [], {}, None) # Will temporarily link prev later

        # Assign lazy buffers to tensors
        t1._lazy_buffer = lb1
        t2._lazy_buffer = lb2

        # Now create the circular dependency
        lb1.prev = [t2] # t1 depends on t2
        lb2.prev = [t1] # t2 depends on t1

        with pytest.raises(RuntimeError, match="Circular dependency detected"):
            t1._lazy_buffer.topo_sort()
            
        # Clean up to avoid issues with subsequent tests if not properly isolated
        t1._lazy_buffer = None
        t2._lazy_buffer = None


    def test_lazy_buffer_realize_and_forward(self):
        a = Tensor((2,2), requires_grad=True)
        b = Tensor((2,2), requires_grad=True)
        
        # Initialize some data in 'a' and 'b' directly in C memory
        a_flat_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b_flat_data = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        for i in range(a_flat_data.size):
            a.c_tensor_ptr.contents.data.contents.data[i] = a_flat_data[i]
            b.c_tensor_ptr.contents.data.contents.data[i] = b_flat_data[i]

        c = a + b # This creates c with a LazyBuffer for Add operation
        d = c.relu() # This creates d with a LazyBuffer for ReLU operation

        assert not d._lazy_buffer._realized
        assert not c._lazy_buffer._realized

        realized_d = d.realize() # This should trigger forward for c then d
        
        assert realized_d is d # realize returns the final output tensor
        assert d._lazy_buffer._realized
        assert c._lazy_buffer._realized

        # Verify the data after realization
        expected_c_data = a_flat_data + b_flat_data # [6.0, 8.0, 10.0, 12.0]
        expected_d_data = np.maximum(expected_c_data, 0) # ReLU, so same as c_data here
        
        assert np.array_equal(c.data.flatten(), expected_c_data)
        assert np.array_equal(d.data.flatten(), expected_d_data)


    def test_lazy_buffer_backward(self):
        # Test a simple graph: a -> add_scalar -> c
        a = Tensor((2,2), requires_grad=True)
        # Initialize 'a' with some data
        a_flat_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        for i in range(a_flat_data.size):
            a.c_tensor_ptr.contents.data.contents.data[i] = a_flat_data[i]

        scalar_val = 5.0
        c = a + scalar_val # c = Add.create_node(a, scalar_val)

        # Trigger backward pass from 'c'
        c.backward()
        
        # After c.backward():
        # 1. c.realize() is called (which calls Add.forward)
        # 2. c._lazy_buffer.topo_sort() is called (reversed)
        # 3. For 'c' (the last node in topological order), its grad is initialized to ones.
        # 4. Add.backward is called.
        
        # Verify c.grad (should be all ones as it's the final output of the backward call)
        assert np.all(c.grad == 1.0)
        
        # Verify a.grad
        # For c = a + S, dc/da = 1. So, gradient of a should be the gradient of c.
        # The `c_add_grad_op` should propagate `out_grad_ptr` to the `prev_ptrs`.
        # Assuming `c_add_grad_op` correctly copies the gradient:
        assert np.all(a.grad == 1.0)


    def test_lazy_buffer_backward_complex_graph(self):
        # Graph: a --- (mul) ---> c --- (relu) ---> d
        #        |                    ^
        #        +--------- (add) ---+
        a = Tensor((2,2), requires_grad=True)
        b = Tensor((2,2), requires_grad=True)

        # Initialize data
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_data = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
        for i, val in enumerate(a_data.flatten()): a.c_tensor_ptr.contents.data.contents.data[i] = val
        for i, val in enumerate(b_data.flatten()): b.c_tensor_ptr.contents.data.contents.data[i] = val

        c_mul = a * b         # Add.create_node(a, b)
        c_add = c_mul + a     # Mul.create_node(c_mul, a) (assuming add_grad accumulates correctly)
        d = c_add.relu()      # ReLU.create_node(c_add)

        # Execute backward from 'd'
        d.backward()

        # Expected values (manual calculation for verification)
        # d = ReLU(c_add)
        # c_add = (a * b) + a
        #
        # ∂L/∂d = 1 (initialized for d)
        #
        # ∂L/∂c_add = ∂L/∂d * ∂d/∂c_add = 1 * (1 if c_add > 0 else 0)
        # Since all input values are positive, c_mul > 0, c_add > 0. So ∂d/∂c_add = 1.
        # Thus, ∂L/∂c_add = 1.0
        #
        # For c_add = (a * b) + a:
        # ∂c_add/∂(a*b) = 1
        # ∂c_add/∂a = 1 (from the direct 'a' input to the sum)
        #
        # So, grad of c_mul should be ∂L/∂c_add * ∂c_add/∂(a*b) = 1 * 1 = 1.0
        # And additional grad for 'a' from this path should be ∂L/∂c_add * ∂c_add/∂a = 1 * 1 = 1.0
        #
        # For c_mul = a * b:
        # ∂c_mul/∂a = b
        # ∂c_mul/∂b = a
        #
        # Total grad for 'a':
        # ∂L/∂a = (∂L/∂c_add * ∂c_add/∂a) + (∂L/∂c_mul * ∂c_mul/∂a)
        # ∂L/∂a = (1 * 1) + (1 * b) = 1 + b
        #
        # Total grad for 'b':
        # ∂L/∂b = (∂L/∂c_mul * ∂c_mul/∂b)
        # ∂L/∂b = 1 * a = a
        
        # Calculate expected gradients
        # c_mul_val = a_data * b_data = [[0.5, 1.0], [1.5, 2.0]]
        # c_add_val = c_mul_val + a_data = [[1.5, 3.0], [4.5, 6.0]] (all > 0)

        expected_d_grad = np.ones_like(d.data)
        expected_c_add_grad = expected_d_grad * (c_add.data > 0) # ReLU gradient
        
        expected_a_grad = expected_c_add_grad + (expected_c_add_grad * b.data) # dL/dc_add * dc_add/da + dL/dcmul * dcmul/da
        expected_b_grad = expected_c_add_grad * a.data # dL/dc_mul * dc_mul/db

        # Due to potential floating point inaccuracies from C operations, use `np.isclose`
        assert np.allclose(d.grad, expected_d_grad)
        assert np.allclose(a.grad, expected_a_grad)
        assert np.allclose(b.grad, expected_b_grad)
