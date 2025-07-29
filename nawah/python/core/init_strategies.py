from abc import ABC, abstractmethod

import numpy as np


class InitStrategy(ABC):
    @abstractmethod
    def init(self, shape: tuple, dtype: type = np.float32) -> np.ndarray:
        pass


class ZeroInit(InitStrategy):
    def init(self, shape: tuple, dtype: type = np.float32) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)


class OneInit(InitStrategy):
    def init(self, shape: tuple, dtype: type = np.float32) -> np.ndarray:
        return np.ones(shape, dtype=dtype)


class RandnInit(InitStrategy):
    def init(self, shape: tuple, dtype: type = np.float32) -> np.ndarray:
        return np.random.random(*shape).astype(dtype)
