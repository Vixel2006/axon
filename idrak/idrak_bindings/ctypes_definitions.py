import ctypes

# Define the C Dtype enum
class CDtype(ctypes.c_int):
    FLOAT32 = 0
    FLOAT64 = 1
    INT32 = 2

# Define the C Device enum
class CDevice(ctypes.c_int):
    CPU = 0
    CUDA = 1

# Define the C SharedPtr struct
class CSharedPtr(ctypes.Structure):
    _fields_ = [
        ("elems", ctypes.c_void_p),
        ("ref_counter", ctypes.c_uint),
        ("dtype", CDtype),
        ("device", CDevice),
    ]

# Define the C Tensor struct
class CTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(CSharedPtr)),
        ("grad", ctypes.POINTER(CSharedPtr)),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("ndim", ctypes.c_int),
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
        ("min_val", ctypes.c_double),
        ("max_val", ctypes.c_double),
    ]