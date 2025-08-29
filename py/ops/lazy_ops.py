from typing import Any, Tuple
from py.elnawah_bindings.ctypes_definitions import CTensor
from .lazy_base import LazyOp

from .lazy_ops.binary_ops import LazyAdd, LazySub, LazyRSub, LazyMul, LazyDiv, LazyRDiv, LazyMatMul
from .lazy_ops.unary_ops import LazyReLU, LazyLog, LazyExp, LazySoftmax, LazyAbs, LazyNeg
from .lazy_ops.reduction_ops import LazySum, LazyMean, LazyMax
from .lazy_ops.movement_ops import LazyView, LazyUnsqueeze, LazySqueeze, LazyTranspose, LazyExpand, LazyBroadcast