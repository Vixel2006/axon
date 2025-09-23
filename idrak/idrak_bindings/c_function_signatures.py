import ctypes
from .c_library_loader import tensor_lib
from .ctypes_definitions import CTensor, CDevice, CStorage

if tensor_lib:
    # Define the C function signatures
    tensor_lib.numel.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    tensor_lib.numel.restype = ctypes.c_int

    tensor_lib.compute_strides.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    tensor_lib.compute_strides.restype = ctypes.POINTER(ctypes.c_int)

    tensor_lib.smalloc.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_int
    ]
    tensor_lib.smalloc.restype = None

    tensor_lib.sfree.argtypes = [
        ctypes.POINTER(CStorage)
    ]
    tensor_lib.sfree.restype = None
 
    tensor_lib.gmalloc.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    tensor_lib.gmalloc.restype = None

    tensor_lib.gfree.argtypes = [ctypes.POINTER(CTensor)]
    tensor_lib.gfree.restype = None
    
    tensor_lib.tmalloc.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        CDevice,
        ctypes.c_bool,
    ]
    tensor_lib.tmalloc.restype = ctypes.POINTER(CTensor)

    tensor_lib.tfree.argtypes = [ctypes.POINTER(CTensor)]
    tensor_lib.tfree.restype = None

    tensor_lib.gmalloc.argtypes = []

    tensor_lib.zeros.argtypes = [
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.zeros.restype = None

    tensor_lib.ones.argtypes = [
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.ones.restype = None

    tensor_lib.randn.argtypes = [
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.randn.restype = None 

    tensor_lib.uniform.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float,
        ctypes.c_float,
    ]
    tensor_lib.uniform.restype = ctypes.POINTER(CTensor)

    tensor_lib.from_data.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    tensor_lib.from_data.restype = None

    tensor_lib.borrow.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CStorage),

    ]
    tensor_lib.borrow.restype = None

    tensor_lib.add_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.add_grad_op.restype = None

    tensor_lib.sub_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.sub_grad_op.restype = None

    tensor_lib.rsub_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.rsub_grad_op.restype = None

    tensor_lib.mul_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.mul_grad_op.restype = None

    tensor_lib.pow_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.pow_grad_op.restype = None

    tensor_lib.div_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.div_grad_op.restype = None

    tensor_lib.rdiv_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.rdiv_grad_op.restype = None

    tensor_lib.matmul_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.matmul_grad_op.restype = None

    tensor_lib.conv2d_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.conv2d_grad_op.restype = None

    tensor_lib.dot_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.dot_grad_op.restype = None

    tensor_lib.relu_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.relu_grad_op.restype = None

    tensor_lib.log_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.log_grad_op.restype = None

    tensor_lib.exp_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.exp_grad_op.restype = None

    tensor_lib.abs_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.abs_grad_op.restype = None

    tensor_lib.neg_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.neg_grad_op.restype = None

    # Reduction operatiosn grad
    tensor_lib.sum_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]

    tensor_lib.sum_grad_op.restype = None

    tensor_lib.mean_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]

    tensor_lib.mean_grad_op.restype = None

    tensor_lib.max_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]

    tensor_lib.max_grad_op.restype = None

    # Movement operations grad
    tensor_lib.concat_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.concat_grad_op.restype = None

    # Unary operations
    tensor_lib.relu_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.relu_op.restype = None

    tensor_lib.log_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.log_op.restype = None

    tensor_lib.exp_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.exp_op.restype = None

    tensor_lib.abs_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.abs_op.restype = None

    tensor_lib.neg_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.neg_op.restype = None

    # Reduction operations
    tensor_lib.max_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.c_int,
        ctypes.c_bool,
    ]
    tensor_lib.max_op.restype = None

    tensor_lib.mean_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.c_int,
        ctypes.c_bool,
    ]
    tensor_lib.mean_op.restype = None

    tensor_lib.sum_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.c_int,
        ctypes.c_bool,
    ]
    tensor_lib.sum_op.restype = None

    tensor_lib.sum_full_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.sum_full_op.restype = None

    tensor_lib.mean_full_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.mean_full_op.restype = None

    tensor_lib.max_full_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
   ]
    tensor_lib.max_full_op.restype = None

    tensor_lib.sum_full_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.sum_full_grad_op.restype = None

    tensor_lib.mean_full_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.mean_full_grad_op.restype = None

    tensor_lib.max_full_grad_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    tensor_lib.max_full_grad_op.restype = None

    # Binary operations
    tensor_lib.add_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.add_op.restype = None

    tensor_lib.sub_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.sub_op.restype = None

    tensor_lib.mul_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.mul_op.restype = None

    tensor_lib.div_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.div_op.restype = None

    tensor_lib.matmul_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    tensor_lib.matmul_op.restype = None

    tensor_lib.conv2d_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    tensor_lib.conv2d_op.restype = None

    tensor_lib.dot_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.dot_op.restype = None

    # Binary operations with scalars
    tensor_lib.add_scalar_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float,
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.add_scalar_op.restype = None

    tensor_lib.sub_scalar_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float,
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.sub_scalar_op.restype = None

    tensor_lib.rsub_scalar_op.argtypes = [
        ctypes.c_float,
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.rsub_scalar_op.restype = None

    tensor_lib.mul_scalar_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float,
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.mul_scalar_op.restype = None

    tensor_lib.div_scalar_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float,
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.div_scalar_op.restype = None

    tensor_lib.rdiv_scalar_op.argtypes = [
        ctypes.c_float,
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.rdiv_scalar_op.restype = None

    tensor_lib.pow_scalar_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float,
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.pow_scalar_op.restype = None

    # Movement operations
    tensor_lib.view_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    tensor_lib.view_op.restype = None

    tensor_lib.squeeze_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.c_int,
    ]
    tensor_lib.squeeze_op.restype = None

    tensor_lib.unsqueeze_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.c_int,
    ]
    tensor_lib.unsqueeze_op.restype = None

    tensor_lib.transpose_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.c_int,
        ctypes.c_int,
    ]
    tensor_lib.transpose_op.restype = None

    tensor_lib.expand_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.c_int),
    ]
    tensor_lib.expand_op.restype = None

    tensor_lib.broadcast_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    tensor_lib.broadcast_op.restype = None


    tensor_lib.concat_op.argtypes = [
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.POINTER(CTensor),
        ctypes.c_int,
        ctypes.c_int,
    ]
    tensor_lib.concat_op.restype = None


    # Optimizers
    tensor_lib.sgd.argtypes = [
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_float,
    ]
    tensor_lib.sgd.restype = None

    tensor_lib.adam.argtypes = [
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    tensor_lib.adam.restype = None

    tensor_lib.zero_grad.argtypes = [
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
    ]
    tensor_lib.zero_grad.restype = None

    tensor_lib.idrak_set_debug_mode.argtypes = [ctypes.c_int]
    tensor_lib.idrak_set_debug_mode.restype = None
