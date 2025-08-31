from .module import Module
from typing import Union, Callable
from .linear import Linear
from py.core import Tensor

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
    

if __name__ == "__main__":
    t = Tensor((2,2), [[2,3], [2,3]])
    model = Sequential(
        Linear(2,3, bias=False),
        Linear(3, 4, bias=False)
    )

    pred = model(t)

    pred.realize()

