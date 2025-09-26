import ctypes

CDevice = ctypes.c_int
CPU = 0
CUDA = 1

class CStorage(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("size", ctypes.c_int),
        ("counter", ctypes.c_int),
    ]

class CTensor(ctypes.Structure):
    pass

CTensor._fields_ = [
    ("data", ctypes.POINTER(CStorage)),
    ("grad", ctypes.POINTER(CStorage)),
    ("shape", ctypes.POINTER(ctypes.c_int)),
    ("strides", ctypes.POINTER(ctypes.c_int)),
    ("device", CDevice),
    ("ndim", ctypes.c_int),
    ("requires_grad", ctypes.c_bool),
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

class ClipExtras(ctypes.Structure):
    _fields_ = [
        ("min_val", ctypes.c_float),
        ("max_val", ctypes.c_float),
    ]
