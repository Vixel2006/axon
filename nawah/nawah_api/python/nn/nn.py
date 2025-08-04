from ..core import Tensor

class nn:
    def __init__(self, modules):
        self.modules = modules

    def __repr__(self):
        pass

    def __getitem__(self, idx):
        return self.modules[idx]

    def __setitem__(self, idx, module):
        self.modules[idx] = module

    def __add__(self, other):
        modules = self.modules.extend(other.modules)

        return nn(modules)
