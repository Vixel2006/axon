from abc import ABC, abstractmethod
import uuid

class Module(ABC):
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__ + "_" + str(uuid.uuid4())[:8]
        self.metadata = {}
        self._parameters = {}
        self._buffers = {}
        self._submodules = {}

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def add_parameter(self, name, tensor):
        self._parameters[name] = tensor

    def register_buffer(self, name, tensor):
        self._buffers[name] = tensor

    def add_module(self, name, module):
        self._submodules[name] = module

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._submodules.values():
            yield from module.parameters()

    def buffers(self):
        for buf in self._buffers.values():
            yield buf
        for module in self._submodules.values():
            yield from module.buffers()

    def state_dict(self):
        state = {f"{self.name}.params": self._parameters.copy(), f"{self.name}.buffers": self._buffers.copy()}
        for name, module in self._submodules.items():
            state.update(module.state_dict())
        return state

    def __repr__(self):
        return f"<Module {self.name} with {len(self._parameters)} params, {len(self._buffers)} buffers>"

