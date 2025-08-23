import nawah_api as nw

a = nw.Tensor(
    [[[1, 3, 4], [23, 4, 5]], [[13, 4, 5], [34, 34, 54]], [[3, 4, 5], [2, 3, 4]]]
)


class Dummyset(nw.Dataset):
    def __init__(self):
        self.features = nw.Tensor(
            [[13, 4, 5], [1, 3, 4], [3, 4, 5], [2, 4, 5], [13, 4, 5]]
        )

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx]


dataloader = nw.DataLoader(dataset=Dummyset(), batch_size=1, shuffle=True)

for feature in dataloader:
    print(feature)
