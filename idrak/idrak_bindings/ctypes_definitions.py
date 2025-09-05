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

class Conv2DBackwardExtras(ctypes.Structure):
    _fields_ = [
        ("padding", ctypes.c_int),
        ("H_in", ctypes.c_int),
        ("W_in", ctypes.c_int),
        ("Kh", ctypes.c_int),
        ("Kw", ctypes.c_int),
        ("Sh", ctypes.c_int),
        ("Sw", ctypes.c_int),
        ("Hout", ctypes.c_int),
        ("Wout", ctypes.c_int),
    ]

BackwardFnType = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(CTensor),
    ctypes.POINTER(ctypes.POINTER(CTensor)),
    ctypes.c_int,
    ctypes.c_void_p,
)