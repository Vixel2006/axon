from abc import ABC, abstractmethod


class Dataset(ABC):
    """
    Abstract base class for a dataset.
    All datasets should implement __len__ and __getitem__.
    """

    @abstractmethod
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, index):
        """
        Returns a single sample and its corresponding label.
        Should return a tuple: (data, label).
        """
        pass
