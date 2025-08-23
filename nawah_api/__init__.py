from .python.core.tensor import Tensor
from .python.api import pipe
from .python.nn import Sequential, layers, activations, loss
from .python.optimizers import SGD, Optimizer
from .python.data import Dataset

import cnawah as nw

DType = nw.DType
Device = nw.Device
DeviceType = nw.DeviceType
Tape = nw.Tape
relu = nw.relu
log = nw.log
exp = nw.exp
softmax = nw.softmax
conv2d = nw.conv2d
ones = nw.ones
zeros = nw.zeros
randn = nw.randn
uniform = nw.uniform
zeros_like = nw.zeros_like
ones_like = nw.ones_like
flatten = nw.flatten
cuda_synchronize = nw.cuda_synchronize

DataLoader = nw.DataLoader

__all__ = [
    "Tensor",
    "DType",
    "Device",
    "DeviceType",
    "Tape",
    "cuda_synchronize",
    "pipe",
    "relu",
    "ones",
    "zeros",
    "randn",
    "uniform",
    "zeros_like",
    "ones_like",
    "Sequential",
    "layers",
    "activations",
    "loss",
    "flatten",
    "Optimizer",
    "SGD",
    "Dataset",
    "DataLoader",
]
