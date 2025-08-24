from abc import ABC, abstractmethod
import cnawah as nw


class Dataset(ABC):
    def __getitem__(self):
        raise NotImplementedError("Get item not implemented yet.")

    def __len__(self):
        raise NotImplementedError("dataset length not implemented yet.")

    def size(self):
        size = 0
        attrs = self.__dict__.values()

        for attr in attrs:
            print(type(attr))
            if isinstance(attr, nw.Tensor):
                size += 1

        return size
