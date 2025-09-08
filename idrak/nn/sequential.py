from .module import Module
from typing import Union, Callable
from .linear import Linear
from idrak.core import Tensor

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)
    
    def __getitem__(self, idx: int) -> Union[Module, Callable]:
        return self.layers[idx]
    
    def __setitem__(self, idx: int, module: Union[Module, Callable]):
        self.layers[idx] = module
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x.realize()
        return x
    
    @property
    def params(self):
        """Override the params property to handle the layers list"""
        params = []
        for layer in self.layers:
            if isinstance(layer, Module):
                params.extend(layer.params)
        return params
    
    @property
    def buffers(self):
        """Override the buffers property to handle the layers list"""
        buffers = []
        for layer in self.layers:
            if isinstance(layer, Module):
                buffers.extend(layer.buffers)
        return buffers


