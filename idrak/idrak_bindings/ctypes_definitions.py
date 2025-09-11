import ctypes

# Define the C SharedPtr struct
class CSharedPtr(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.POINTER(ctypes.c_float)),
        ("ref_counter", ctypes.c_int),
    ]

# Define the C Tensor struct
class CTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(CSharedPtr)),
        ("grad", ctypes.POINTER(CSharedPtr)),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("requires_grad", ctypes.c_bool),
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

class StackExtras(ctypes.Structure):
    _fields_ = [
        ("axis", ctypes.c_int),
    ]

class ConcatExtras(ctypes.Structure):
    _fields_ = [
        ("axis", ctypes.c_int),
    ]

class ClipExtras(ctypes.Structure):
    _fields_ = [
        ("min_val", ctypes.c_float),
        ("max_val", ctypes.c_float),
    ]

