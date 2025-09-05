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
        return x
    
