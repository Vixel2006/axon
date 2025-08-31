from abc import ABC, abstractmethod
from typing import Any

class Module(ABC):
    @abstractmethod
    def forward(self, x, *args, **kwargs):
        pass

    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        return self.forward(x, *args, **kwds)