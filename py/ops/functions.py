from __future__ import annotations
import ctypes
from typing import Any, Tuple, Optional, List
import sys

from py.elnawah_bindings.ctypes_definitions import CTensor, BackwardFnType
from py.elnawah_bindings.c_library_loader import tensor_lib
from .base import Function

from .functions.binary_ops import Add, Sub, RSub, Mul, Div, RDiv, MatMul
from .functions.unary_ops import ReLU, Log, Exp, Softmax, Abs, Neg
from .functions.reduction_ops import Sum, Mean, Max
from .functions.movement_ops import View, Unsqueeze, Squeeze, Transpose, Expand, Broadcast