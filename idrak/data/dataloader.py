from __future__ import annotations
from typing import Iterator, List, Optional, Sized
import math
import random
import numpy as np
from idrak.functions import from_data

class DataLoader:
    def __init__(self, dataset: Sized, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self) -> Iterator[List]:
        if self.shuffle:
            random.shuffle(self.indices)

        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            
            # Transpose the batch data
            transposed_batch = list(zip(*batch_data))
            
            stacked_tensors = []
            for items in transposed_batch:
                # Assuming all items in 'items' are Tensors and have a .data attribute (numpy array)
                # And assuming they have compatible shapes for stacking
                numpy_arrays = [item.data for item in items]
                stacked_numpy_array = np.stack(numpy_arrays, axis=0)
                
                # Get the shape and device from the first Tensor in the list
                first_tensor = items[0]
                stacked_tensor_shape = stacked_numpy_array.shape
                stacked_tensor_device = first_tensor.device
                stacked_tensor_requires_grad = first_tensor.requires_grad
                
                stacked_tensors.append(from_data(stacked_tensor_shape, stacked_numpy_array, device=stacked_tensor_device, requires_grad=stacked_tensor_requires_grad))
            
            yield tuple(stacked_tensors)

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)


    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)
