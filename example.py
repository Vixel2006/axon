import nawah_api as nw
import time

a = nw.Tensor.from_data(
    [[[1, 3, 4], [23, 4, 5]], [[13, 4, 5], [34, 34, 54]], [[3, 4, 5], [2, 3, 4]]]
)


class Dummyset(nw.Dataset):
    def __init__(self):
        self.features = nw.Tensor.from_data(
            [[13, 4, 5], [1, 3, 4], [3, 4, 5], [2, 4, 5], [13, 4, 5]]
        )

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx]


dataloader = nw.DataLoader(dataset=Dummyset(), batch_size=2)

print(next(dataloader))

b = nw.Tensor([2, 3])
print(b)
