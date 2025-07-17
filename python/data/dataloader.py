import numpy as np

from ..core.tensor import Tensor
from .dataset import Dataset


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]

            batch_samples = [self.dataset[j] for j in batch_indices]

            batch_data_list, batch_labels_list = zip(*batch_samples)

            batch_data_np = np.stack(batch_data_list, axis=0)
            batch_labels_np = np.array(batch_labels_list)

            if batch_labels_np.ndim == 1:
                batch_labels_np = batch_labels_np.reshape(-1, 1)

            yield Tensor(batch_data_np), Tensor(batch_labels_np)
