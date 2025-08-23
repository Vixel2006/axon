from abc import ABC, abstractmethod


class Dataset(ABC):
    def __getitem__(self):
        raise NotImplementedError("Get item not implemented yet.")

    def __len__(self):
        raise NotImplementedError("dataset length not implemented yet.")
