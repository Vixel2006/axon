import ctypes

# Define the C Tensor struct
class CTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("requires_grad", ctypes.c_bool),
        ("grad", ctypes.POINTER(ctypes.c_float)),
    ]


class CNode(ctypes.Structure):
    _fields_ = [
        ("out", ctypes.POINTER(CTensor)),
        ("prev", ctypes.POINTER(ctypes.POINTER(CTensor))),
        ("n_prev", ctypes.c_int),
        ("extras", ctypes.c_void_p),
        ("forward_fn", ctypes.c_void_p),
        ("backward_fn", ctypes.c_void_p),
    ]

BackwardFnType = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(CTensor),
    ctypes.POINTER(ctypes.POINTER(CTensor)),
    ctypes.c_int,
    ctypes.c_void_p,
)